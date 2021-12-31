import argparse
import os
import shutil
import json
import multiprocessing
import numpy as np
import pandas as pd
import paddle
import paddle.distributed as dist
from tqdm import tqdm
from dataloaders.vcr import VCR, VCRLoader
from utils.paddle_misc import time_batch
from utils.paddle_misc import time_batch, save_checkpoint, clip_grad_norm,restore_checkpoint, print_para, restore_best_checkpoint
from model.multiatt.model import AttentionQA
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)
#初始化并行环境
dist.init_parallel_env()


parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
    default='model/saves/flagship_answer'
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

args = parser.parse_args()
with open('./default.json', 'r') as f:
    Params = [json.loads(s) for s in f]
params = Params[0]
train, val = VCR.splits(mode='rationale' if args.rationale else 'answer')
NUM_GPUS = paddle.distributed.get_world_size()
NUM_CPUS = multiprocessing.cpu_count()

print('NUM_GPUS',NUM_GPUS)
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


num_workers = (4 * NUM_GPUS if NUM_CPUS >= 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 64 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
# test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100

model = AttentionQA(pool_question=True,
                    pool_answer=True)

for submodule in model.detector.backbone:
    if isinstance(submodule, paddle.nn.BatchNorm2D):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False

if NUM_GPUS > 1:
    model = paddle.DataParallel(model)
param_shapes = print_para(model)

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    if b%50==0:
        print(b)
    with paddle.no_grad():
        output_dict = model(batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch[4].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.5f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
