import json
import os
import numpy as np
import paddle
import h5py
from copy import deepcopy
#from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from paddle.io import Dataset
from paddlenlp.data import Stack, Pad, Dict






GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def _fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, token_indexers, pad_ind=-1):
    """
    Turn a detection list into what we want: some text, as well as some tags.
    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_ind))

    text_field = [x[0] for x in new_tokenization_with_tags]
    tags = [x[1] for x in new_tokenization_with_tags]
    return bert_embs,tags

class VCR(Dataset):
    def __init__(self,split, mode,only_use_relevant_dets=True, add_image_as_a_box=True,embs_to_load='bert_da',conditioned_answer_choice=0):
        self.split = split
        self.mode = mode
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections" , flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        with open(os.path.join('data', '{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        if split  not in ('test','train','val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        if mode not in ('answer', 'rationale'):
            raise ValueError("split must be answer or rationale")

        self.token_indexers = {}
        with open(os.path.join('data', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)

        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}
        self.embs_to_load = embs_to_load
        self.h5fn = os.path.join('data/bert_feature', f'{self.embs_to_load}_{self.mode}_{self.split}.h5')
        print("Loading embeddings from {}".format(self.h5fn), flush=True)

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset  """
        kwargs_copy = {x: y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        # test = cls(split='test', **kwargs_copy)
        return train, val  #, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', mode='answer', **kwargs)] + [
            cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)
    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        """
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]
        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = deepcopy(self.items[index])

        #  Load questions and answers
        if self.mode == 'rationale':
            conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
            item['question'] += item['answer_choices'][conditioned_label]
        answer_choices = item['{}_choices'.format(self.mode)]
        dets2use, old_det_to_new_ind = self._get_dets_to_use(item)

        # Load in BERT,We'll get contextual representations of the context and the answer choices
        with h5py.File(self.h5fn, 'r') as h5:
            grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}

        # Essentially we need to condition on the right answer choice here, if we're doing QA->R. We will always
        # condition on the `conditioned_answer_choice.`
        condition_key = self.conditioned_answer_choice if self.split == "test" and self.mode == "rationale" else ""
        instance_dict = {}
        if 'endingonly' not in self.embs_to_load:
            questions_tokenized, question_tags = zip(*[_fix_tokenization(
                item['question'],
                grp_items[f'ctx_{self.mode}{condition_key}{i}'],
                old_det_to_new_ind,
                item['objects'],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1
            ) for i in range(4)])
            #print('======questions_tokenized',questions_tokenized)
            instance_dict['question'] = questions_tokenized
            instance_dict['question_tags'] = question_tags

        answers_tokenized, answer_tags = zip(*[_fix_tokenization(
            answer,
            grp_items[f'answer_{self.mode}{condition_key}{i}'],
            old_det_to_new_ind,
            item['objects'],
            token_indexers=self.token_indexers,
            pad_ind=0 if self.add_image_as_a_box else -1
        ) for i, answer in enumerate(answer_choices)])
        instance_dict['answers'] = answers_tokenized
        instance_dict['answer_tags'] = answer_tags
        if self.split != 'test':
            instance_dict['label'] = item['{}_label'.format(self.mode)]
        instance_dict['metadata'] = {'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
                                                   'img_fn': item['img_fn'],
                                                   'question_number': item['question_number']}


        ###Load image now and rescale it.
        while item['img_fn']!=item['img_fn'].replace("./", "/"):
            item['img_fn'] = item['img_fn'].replace("./", "/")

        image = load_image(os.path.join('data/vcr1images', item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        #load boxs
        while item['metadata_fn']!=item['metadata_fn'].replace("./", "/"):
            item['metadata_fn'] = item['metadata_fn'].replace("./", "/")
        with open(os.path.join('data/vcr1images','vcr1images', item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i])
                          for i in dets2use])
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        instance_dict['segms'] = segms
        instance_dict['objects'] = obj_labels

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = boxes
        #print('--------instance_dict',instance_dict)
        instance_dict['images'] = image
        return image , instance_dict

def pad_np_bert(x,max_len,emb_size,padding_value):
    new_arr = np.ones((max_len,emb_size), dtype = np.float32) * padding_value
    new_arr[:len(x)] = x
    return new_arr

def pad_np_tag(x,max_len,padding_value):
    new_arr = np.ones((max_len), dtype = np.float32) * padding_value
    new_arr[:len(x)] = x
    return new_arr
def pad_np_segms(x,max_len,padding_value):
    new_arr = np.ones((max_len,14,14), dtype = np.int32) * padding_value
    new_arr[:len(x)] = x
    return new_arr

def collate_fn(data, to_gpu=False):
    images, instances = zip(*data)
    question_len = []
    answer_len = []
    box_len = []
    segms_len = []
    for i in instances:
        question_len.append(max(len(j) for j in i['question_tags']))
        answer_len.append(max(len(j) for j in i['answer_tags']))
        segms_len.append(len(i['segms']))
        box_len.append(len(i['boxes'] ))
    max_question_len = max(question_len)
    max_answer_len = max(answer_len)
    max_box_len = max(box_len)
    max_segms_len = max(segms_len)
    for i in instances:
        i['question'] = [pad_np_bert(x,max_question_len,x.shape[1],0) for x in i['question']]
        i['answers'] = [pad_np_bert(x, max_answer_len,x.shape[1],0) for x in i['answers']]
        i['question_tags'] = [pad_np_tag(x,max_question_len,-2) for x in i['question_tags']]
        i['answer_tags'] = [pad_np_tag(x,max_answer_len,-2) for x in i['answer_tags']]
        i['question_mask'] = np.where(np.array(i['question_tags'])>-1 , 1 ,0)
        i['answer_mask'] = np.where(np.array(i['answer_tags']) > -1 , 1 ,0)
        i['boxes'] = pad_np_bert(i['boxes'],max_box_len,i['boxes'].shape[1],-1)
        i['box_mask'] = np.all( i['boxes'] > -1 , -1 ).astype(int)
        i['segms'] = pad_np_segms(i['segms'],max_segms_len,0)
    batchify_fn = Dict({'question': Stack(), 'question_tags': Stack(),
                         'answers':Stack(),'answer_tags' : Stack(),
                         'label' : Stack(),
                         'segms' : Stack(),'objects':Pad(pad_val = -1),
                         'boxes' : Stack(),'question_mask':Stack(),
                         'answer_mask':Stack(),'box_mask':Stack(),
                         'images':Stack()})
    td = batchify_fn(instances)
    return td


class VCRLoader(paddle.io.DataLoader):
    @classmethod
    def from_dataset(cls, data, batch_size=1, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset = data,
            batch_size = batch_size * num_gpus,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn= lambda x:collate_fn(x,to_gpu=False),
            drop_last = data.is_train,
            **kwargs
        )
        return loader

# if __name__ == '__main__':
#     train, val, test = VCR.splits()
#     #print(type(train))
#     # for i in range(len(train)):
#     #     res = train[i]
#     #     print(type(res))
#     #     print('===res',res)
#     #     print("done with {}".format(i))
#     NUM_GPUS = 1
#     num_workers = 1
#     loader_params = {'batch_size': 2 // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}
#
#     train_loader = VCRLoader.from_dataset(train, **loader_params)
#     for i in train_loader:
#         print('########'*30)
#         #print(train_loader[i])
















