import numpy as np
import pandas as pd
import os
import re
import shutil
import paddle
import time
def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1 - start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i

class Flattener(paddle.nn.Layer):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        # print('!!!!!!!!!!!!!!!x.type',type(x))
        # print('!!!!!!!!!!!!!!!x.shape(0)', x.shape[0])
        # print('!!!!!!!!!!!!!!!x.reshape([x.size(0), -1])',x.reshape([x.shape[0], -1]).shape)
        return x.reshape([x.shape[0], -1])

def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    #output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    # print('!!!!!!!!!!!!!!!sequence',*sequence.shape[1:])
    output = paddle.zeros([len(lengths), max(lengths), *sequence.shape[1:]],dtype=sequence.dtype)
    # output =
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    # output = paddle.ones_like(output)
    print('+++++++++++++++++++++++++',output.shape,sequence.shape)
    return output

def extra_leading_dim_in_sequence(f, x, mask):
    return f(x.view(-1, *x.shape[2:]), mask.view(-1, mask.shape[2])).view(*x.shape[:3], -1)


def find_latest_checkpoint(serialization_dir):
    """
    Return the location of the latest model and training state files.
    If there isn't a valid checkpoint then return None.
    """
    have_checkpoint = (serialization_dir is not None and
                       any("model_state_epoch_" in x for x in os.listdir(serialization_dir)))

    if not have_checkpoint:
        return None

    serialization_files = os.listdir(serialization_dir)
    model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
    # Get the last checkpoint file.  Epochs are specified as either an
    # int (for end of epoch files) or with epoch and timestamp for
    # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
    found_epochs = [
        # pylint: disable=anomalous-backslash-in-string
        re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
        for x in model_checkpoints
    ]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split('.')
        if len(pieces) == 1:
            # Just a single epoch without timestamp
            int_epochs.append([int(pieces[0]), 0])
        else:
            # has a timestamp
            int_epochs.append([int(pieces[0]), pieces[1]])
    last_epoch = sorted(int_epochs, reverse=True)[0]
    if last_epoch[1] == 0:
        epoch_to_load = str(last_epoch[0])
    else:
        epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

    model_path = os.path.join(serialization_dir,
                              "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(serialization_dir,
                                       "training_state_epoch_{}.th".format(epoch_to_load))
    return model_path, training_state_path

def save_checkpoint(model, optimizer, serialization_dir, epoch, val_metric_per_epoch, is_best=None,
                    learning_rate_scheduler=None) -> None:
    """
    Saves a checkpoint of the model to self._serialization_dir.
    Is a no-op if self._serialization_dir is None.
    Parameters
    ----------
    epoch : Union[int, str], required.
        The epoch of training.  If the checkpoint is saved in the middle
        of an epoch, the parameter is a string with the epoch and timestamp.
    is_best: bool, optional (default = None)
        A flag which causes the model weights at the given epoch to
        be copied to a "best.th" file. The value of this flag should
        be based on some validation metric computed by your model.
    """
    if serialization_dir is not None:
        model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = model.module.state_dict() if isinstance(model, paddle.DataParallel) else model.state_dict()
        paddle.save(model_state, model_path)

        training_state = {'epoch': epoch,
                          'val_metric_per_epoch': val_metric_per_epoch,
                          'optimizer': optimizer.state_dict()
                          }
        if learning_rate_scheduler is not None:
            training_state["learning_rate_scheduler"] = \
                learning_rate_scheduler.lr_scheduler.state_dict()
        training_path = os.path.join(serialization_dir,"training_state_epoch_{}.th".format(epoch))
        # print('____________________',training_path)
        paddle.save(training_state, training_path)
        if is_best:
            print("Best validation performance so far. Copying weights to '{}/best.th'.".format(serialization_dir))
            shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))

def restore_best_checkpoint(model, serialization_dir):
    fn = os.path.join(serialization_dir, 'best.th')

    model_state = paddle.load(fn)
    assert os.path.exists(fn)
    if isinstance(model, paddle.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)


def restore_checkpoint(model, optimizer, serialization_dir, learning_rate_scheduler=None):
    """
    Restores a model from a serialization_dir to the last saved checkpoint.
    This includes an epoch count and optimizer state, which is serialized separately
    from  model parameters. This function should only be used to continue training -
    if you wish to load a model for inference/load parts of a model into a new
    computation graph, you should use the native Pytorch functions:
    `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``
    If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
    this function will do nothing and return 0.
    Returns
    -------
    epoch: int
        The epoch at which to resume training, which should be one after the epoch
        in the saved training state.
    """
    # print('------------------------serialization_dir',serialization_dir)
    latest_checkpoint = find_latest_checkpoint(serialization_dir)

    if latest_checkpoint is None:
        # No checkpoint to restore, start at 0
        return 0, []

    model_path, training_state_path = latest_checkpoint

    # Load the parameters onto CPU, then transfer to GPU.
    # This avoids potential OOM on GPU for large models that
    # load parameters onto GPU then make a new GPU copy into the parameter
    # buffer. The GPU transfer happens implicitly in load_state_dict.
    model_state = paddle.load(model_path)
    training_state = paddle.load(training_state_path)
    if isinstance(model, paddle.DataParallel):
        model.module.set_state_dict(model_state)
    else:
        model.set_state_dict(model_state)

    # idk this is always bad luck for me
    optimizer.set_state_dict(training_state["optimizer"])

    if learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        learning_rate_scheduler.lr_scheduler.set_state_dict(
            training_state["learning_rate_scheduler"])
    #move_optimizer_to_cuda(optimizer)

    # We didn't used to save `validation_metric_per_epoch`, so we can't assume
    # that it's part of the trainer state. If it's not there, an empty list is all
    # we can do.
    if "val_metric_per_epoch" not in training_state:
        print("trainer state `val_metric_per_epoch` not found, using empty list")
        val_metric_per_epoch: []
    else:
        val_metric_per_epoch = training_state["val_metric_per_epoch"]

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state["epoch"] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1
    return epoch_to_return, val_metric_per_epoch


def detokenize(array, vocab):
    """
    Given an array of ints, we'll turn this into a string or a list of strings.
    :param array: possibly multidimensional numpy array
    :return:
    """
    if array.ndim > 1:
        return [detokenize(x, vocab) for x in array]
    tokenized = [vocab.get_token_from_index(v) for v in array]
    return ' '.join([x for x in tokenized if x not in (vocab._padding_token)])

def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    total_params = 0
    total_params_training = 0
    name_list = []
    for p_name, p in model.named_parameters():
        #print( ' p_name, p',p_name, p.shape,type(np.asarray(p)),np.asarray(p))
        # if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
        st[p_name] = ([str(x) for x in  p.shape],np.asarray(p).size, p.stop_gradient)
        total_params += np.prod(np.asarray(p).size)
        name_list.append(p_name)
        if  not p.stop_gradient:
            total_params_training += np.prod(np.asarray(p).size)
    name_list = pd.DataFrame(name_list)
    name_list.to_csv('paddle_param_name.csv',index=False,header=False,sep='\t')
    pd.set_option('display.max_columns', None)
    shapes_df = pd.DataFrame([(p_name, '[{}]'.format(','.join(size)), prod, p_req_grad)
                              for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1])],
                             columns=['name', 'shape', 'size', 'requires_grad']).set_index('name')
    # shapes_df.to_csv('paddle_param_name.csv',index=False,header=False,sep='\t')
    print('\n {:.1f}M total parameters. {:.1f}M training \n ----- \n {} \n ----'.format(total_params / 1000000.0,
                                                                                        total_params_training / 1000000.0,
                                                                                        shapes_df.to_string()),
          flush=True)
    return shapes_df

def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def batch_iterator(seq, batch_size, skip_end=True):
    for b_start, b_end in batch_index_iterator(len(seq), batch_size, skip_end=skip_end):
        yield seq[b_start:b_end]

