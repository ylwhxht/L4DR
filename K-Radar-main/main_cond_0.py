'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta

import argparse


def init_dist_pytorch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default = 0 , type=int)
    parser.add_argument('--nproc_per_node',default = 0 , type=int)
    parser.add_argument('--nnode',default = 0 , type=int)
    parser.add_argument('--local_rank', default = -1 ,type=int)
    parser.add_argument('--tag', default = 'none',type=str)
    parser.add_argument('--cfg_file', default = 'default',type=str)
    args = parser.parse_args()
    if args.local_rank == -1:
        return -1, args.cfg_file, args.tag
    else:
        
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank % num_gpus)
        torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
            rank=args.local_rank,
            world_size=num_gpus
        )
        rank = dist.get_rank()
        return rank, args.cfg_file, args.tag

if __name__ == '__main__':
    rank, PATH_CONFIG, tag = init_dist_pytorch()
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='test', rank = rank, tag = tag)
    max_epoch = 0
    for epoch in range(100,0,-1):
        PATH_MODEL = '/mnt/32THHD/hx/K-Radar-main/logs/L4DR_PlusPlus/train_'+ tag + '/models/model_'+str(epoch)+'.pt'
        if os.path.exists(PATH_MODEL):
            max_epoch = epoch - 1
            break
    for epoch in range(max_epoch - 7, max_epoch + 1):
        PATH_MODEL = '/mnt/32THHD/hx/K-Radar-main/logs/L4DR_PlusPlus/train_'+ tag + '/models/model_'+str(epoch)+'.pt'
        pline.load_dict_model(PATH_MODEL)
        print('* Start resume, path_state_dict =  ', PATH_MODEL)
        pline.network.eval()
        pline.validate_kitti_conditional(epoch = epoch, list_conf_thr=[0.1,0.2,0.3], is_subset=False, is_print_memory=False)
