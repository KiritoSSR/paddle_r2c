import argparse
import os
import shutil
import json
import multiprocessing
import numpy as np
import pandas as pd
import paddle
from dataloaders.vcr import VCR, VCRLoader
from utils.paddle_misc import time_batch, save_checkpoint, clip_grad_norm,restore_checkpoint, print_para, restore_best_checkpoint
from model.multiatt.model import AttentionQA



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


NUM_GPUS = paddle.distributed.get_world_size()
NUM_CPUS = multiprocessing.cpu_count()
print('NUM_GPUS',NUM_GPUS)
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

num_workers = (4 * NUM_GPUS if NUM_CPUS >= 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 1 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}


#定义模型
model = AttentionQA(pool_question=True, pool_answer=True)

param_shapes = print_para(model)


#单个数据加载,将需要测试的数据的json单独提取出来，命名成val_one.jsonl
val_one = VCR.splits_predict(mode='rationale' if args.rationale else 'answer')
val_loader = VCRLoader.from_dataset(val_one, **loader_params)

restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with paddle.no_grad():
        output_dict = model(batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch[4].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.5f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
