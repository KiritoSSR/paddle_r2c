#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import json
import multiprocessing
import numpy as np
import pandas as pd
import paddle
import paddle.distributed as dist
from paddle.regularizer import L2Decay
from tqdm import tqdm
from dataloaders.vcr import VCR, VCRLoader
from utils.paddle_misc import time_batch, save_checkpoint,restore_checkpoint, print_para, restore_best_checkpoint
from model.multiatt.model import AttentionQA
from config import args
import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
#初始化多卡训练环境
dist.init_parallel_env()

with open('./default.json', 'r') as f:
    Params = [json.loads(s) for s in f]
params = Params[0]
train, val = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))
NUM_GPUS = paddle.distributed.get_world_size()
NUM_CPUS = multiprocessing.cpu_count()

print('NUM_GPUS',NUM_GPUS)
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

num_workers = (4 * NUM_GPUS if NUM_CPUS >= 32 else 2*NUM_GPUS)-1
num_workers = 30
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
# test_loader = VCRLoader.from_dataset(test, **loader_params)
ARGS_RESET_EVERY = args.args_reset_every
model = AttentionQA(input_dropout = args.input_dropout,
                    hidden_dim_maxpool = args.hidden_dim_maxpool,
                    pool_question=args.pool_question,
                    pool_answer=args.pool_answer)
#固定detector.backbone部分的权重
for submodule in model.detector.backbone:
    if isinstance(submodule, paddle.nn.BatchNorm2D):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.stop_gradient = True
#设置学习率的
scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.lr, factor=args.scheduler_factor,mode=args.scheduler_model,
                                                patience=args.scheduler_patience,verbose=args.scheduler_verbose,cooldown=args.scheduler_cooldown)
#进行梯度裁剪
clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_norm)
#和torch一样设置L2正则化策略
optimizer = paddle.optimizer.Adam(learning_rate=scheduler,
        parameters=[x for x in model.parameters() if not  x.stop_gradient],
        weight_decay=L2Decay(args.weight_decay),
        grad_clip=clip)
#在程序中断后可接着最新的epoch进行训练
if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)
#进行多卡训练
if NUM_GPUS > 1:
    model = paddle.DataParallel(model)

#paddle中没有找到正交初始化选项，因此加载torch的初始化
checkpoint = paddle.load('initial_param.pd')
model.set_state_dict(checkpoint)
#打印模型参数
param_shapes = print_para(model)
num_batches = 0
if start_epoch == 0:
    start_epoch = start_epoch + 1
for epoch_num in range(start_epoch,args.num_epoch + start_epoch):
    train_results = []
    norms = []
    model.train()
    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        output_dict = model(batch)
        loss = output_dict['loss'].mean() + output_dict['cnn_regularization_loss'].mean()
        loss.backward()
        num_batches += 1
        optimizer.step()
        optimizer.clear_grad()

        # train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
        #                                 'crl': output_dict['cnn_regularization_loss'].mean().item()}))
        # 单卡训练可以调取get_metrics函数，查看训练过程中的准确率
        train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                        'crl': output_dict['cnn_regularization_loss'].mean().item(),
                                        'accuracy': model.get_metrics(reset=(b % ARGS_RESET_EVERY) == 0)['accuracy'],
                                        }))
        if b % ARGS_RESET_EVERY == 0 and b > 0:
            logging.info('loss:%f', output_dict['loss'].mean().item())
            logging.info('crl:%f', output_dict['cnn_regularization_loss'].mean().item())

            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)

    print("---\nTRAIN EPOCH {:2d}:\n\n----".format(epoch_num))
    val_probs = []
    val_labels = []
    val_loss_sum = 0.0
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with paddle.no_grad():
            output_dict = model(batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch[4].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch[4].shape[0]
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    val_loss_avg = val_loss_sum / val_labels.shape[0]
    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))

    print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
              flush=True)
    if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - args.patience):
        print("Stopping at epoch {:2d}".format(epoch_num))
        break
    save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                        is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best，加载最好的模型参数
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
