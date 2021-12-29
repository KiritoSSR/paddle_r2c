"""
it's the resnet backbone
"""
import paddle
import paddle.nn as nn
from utils.Resnet50 import resnet50
from utils.Resnet50_imagenet import resnet50_imagenet
from paddle.vision.ops import RoIAlign
from utils.paddle_misc import Flattener ,pad_sequence
from paddle.nn import functional as F
import pickle
import numpy as np
USE_IMAGENET_PRETRAINED = True


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def _load_resnet(pretrained = True):
    backbone = resnet50(pretrained=False)
    with open('utils/torch_resnet50.pkl', 'rb') as f:
        param2 = pickle.load(f)
    backbone.set_state_dict(param2)
    return backbone



def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet50_imagenet(pretrained=False)
    with open('utils/torch_resnet50.pkl', 'rb') as f:
        param2 = pickle.load(f)
    backbone.set_state_dict(param2)
    return backbone

class SimpleDetector(nn.Layer):
    def __init__(self,pretrained=True, average_pool=True, semantic=True, final_dim=1024):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()
        backbone = _load_resnet_imagenet(pretrained=pretrained) if USE_IMAGENET_PRETRAINED else _load_resnet(
            pretrained=pretrained)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        for p_name, p in self.backbone.named_parameters():
            if 'weight'  in p_name or 'bias'  in p_name:
                p.stop_gradient = True

        self.roi_align = RoIAlign((7, 7) if USE_IMAGENET_PRETRAINED else (14, 14),
                                  spatial_scale=1 / 16)
        if semantic:
            self.mask_dims = 32
            self.object_embed = paddle.nn.Embedding(num_embeddings=81, embedding_dim=128)
            self.mask_upsample = paddle.nn.Conv2D(1, self.mask_dims, kernel_size=3,
                                                  stride=2 if USE_IMAGENET_PRETRAINED else 1,
                                                  padding=1, bias_attr=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        after_roi_align = [backbone.layer4]

        self.final_dim = final_dim
        if average_pool:
            after_roi_align += [nn.AvgPool2D(7, stride=1, exclusive=False), Flattener()]
        self.after_roi_align = paddle.nn.Sequential(*after_roi_align)
        self.obj_downsample = paddle.nn.Sequential(
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Linear(2048 + (128 if semantic else 0), final_dim, bias_attr=True,weight_attr=paddle.nn.initializer.XavierUniform()),
            paddle.nn.ReLU(),
        )
        self.regularizing_predictor = paddle.nn.Linear(2048, 81, bias_attr=True,weight_attr=paddle.nn.initializer.XavierUniform())

        pad_split = [0, 40, 0, 0]
        self.my_pad_split = paddle.nn.Pad2D(padding=pad_split)

    def forward(self,
                    images,
                    boxes,
                    box_mask,
                    classes,
                    segms
                    ):
            """
                   :param images: [batch_size, 3, im_height, im_width]
                   :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
                   :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
                   :return: object reps [batch_size, max_num_objects, dim]
                   """
            # [batch_size, 2048, im_height // 32, im_width // 32
            img_feats = self.backbone(images)
            img_feats.stop_gradient = True
            lengths = box_mask.sum(1).tolist()
            boxes_num = paddle.sum(box_mask, -1).astype('int32')
            boxes = boxes.reshape([boxes.shape[0] * boxes.shape[1], boxes.shape[2]])
            box_mask = box_mask.reshape([box_mask.shape[0] * box_mask.shape[1]])
            box_inds = paddle.to_tensor([i for i in range(len(box_mask)) if box_mask[i] != 0])

            assert box_inds.shape[0] > 0
            rois = paddle.fluid.layers.gather(boxes, box_inds)
            # Object class and segmentation representations
            roi_align_res = self.roi_align(img_feats, rois,boxes_num)
            if self.mask_upsample is not None:
                assert segms is not None
                segms = segms.reshape([segms.shape[0]*segms.shape[1],segms.shape[2],segms.shape[3]])
                segms_indexed =( paddle.fluid.layers.gather(segms, box_inds)-0.5).unsqueeze(1)
                segms_unsample = self.mask_upsample(segms_indexed)
                roi_align_res += paddle.concat((segms_unsample , paddle.zeros([segms_unsample.shape[0],roi_align_res.shape[1]-segms_unsample.shape[1],segms_unsample.shape[2],segms_unsample.shape[3]])),1)
            post_roialign = self.after_roi_align(roi_align_res)
            # Add some regularization, encouraging the model to keep giving decent enough predictions
            obj_logits = self.regularizing_predictor(post_roialign)
            classes = classes.reshape([classes.shape[0]*classes.shape[1]])
            obj_labels = paddle.fluid.layers.gather(classes, box_inds)
            cnn_regularization = F.cross_entropy(obj_logits, obj_labels)[None]

            feats_to_downsample = post_roialign if self.object_embed is None else paddle.concat(
                (post_roialign, self.object_embed(obj_labels)), -1)

            roi_aligned_feats = self.obj_downsample(feats_to_downsample)
            # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...

            output = paddle.split(roi_aligned_feats, num_or_sections=lengths, axis=0)
            for i in range(len(output)):
                output[i] = self.my_pad_split(output[i])[:max(lengths)].unsqueeze(0)
            output = paddle.concat(output,axis=0)
            return {
                'obj_reps_raw': post_roialign,
                'obj_reps': output,
                'obj_logits': obj_logits,
                'obj_labels': obj_labels,
                'cnn_regularization_loss': cnn_regularization
            }








