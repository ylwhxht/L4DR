import concurrent.futures as futures
import os
import pathlib
import re
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from skimage import io
def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
def get_label_anno(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    x = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    result_dis = [0,0,0]
    dis = np.linalg.norm(x,axis=1)
    for idx in range(len(result_dis)):
        result_dis[idx] += ((idx * 25 <  dis) & (dis<= (idx+1) * 25)).sum()
    return  result_dis

def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    final_result_dis = [0,0,0]
    for idx in tqdm(image_ids):
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        dis_num = get_label_anno(label_filename)
        for i in range(len(final_result_dis)):
            final_result_dis[i] += dis_num[i]
    print(final_result_dis)
    return annos

split_path = '/mnt/32THHD/hx/K-Radar-main/logs/PP_RLF/test_35e_0.1denoise_unnormal/test_kitti/epoch_30_total/0.3/heavysnow/val.txt'
val_ids = _read_imageset_file(split_path)
labels_dir = '/mnt/32THHD/hx/K-Radar-main/logs/PP_RLF/test_35e_0.1denoise_unnormal/test_kitti/epoch_30_total/0.3/heavysnow/gts'
gt_annos = get_label_annos(labels_dir, val_ids)