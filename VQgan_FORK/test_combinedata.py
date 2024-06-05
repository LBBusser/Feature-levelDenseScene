from collections import OrderedDict
import torch
import argparse
import sys
import random
import torchvision.transforms as trn
from tqdm import tqdm
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torchvision.transforms as trn
import pickle
from torchvision import transforms
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from image_transformations import RandomResizedCrop, RandomHorizontalFlip, Compose
from my_utils import set_device
import faiss
from enum import Enum
# from my_utils import denormalize_video, make_seg_maps
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
from torchvision.datasets.utils import download_url
from abc import ABC, abstractmethod
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from data_loader import NYUMemoryTasksDataLoader, CocoMemoryTasksDataLoader, CombinedMemoryTasksDataLoader





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2)
    argparser.add_argument('--img_size', type=int, help='img_size', default=504)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--dataset', type=str, default='MSCOCO')
    argparser.add_argument('--data_size', type=int, default=100)

    # COCO training / non-episodic image captioning
    argparser.add_argument('--num_epochs', type=int, help='epoch number for training', default=20)
    argparser.add_argument('--batch_size', type=int, help='batch size for few shot training', default=8)
    argparser.add_argument('--coco_annotations_path', type=str, default='/scratch-shared/combined_hbird/mscoco_hbird/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=5)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=2e-04)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=100)
    argparser.add_argument('--trained_model_name', type=str, default='default.pt')
    argparser.add_argument('--patch_size', type=int, default=14)
    #SCANN setup
    argparser.add_argument('--num_leaves', type=int, default=1)
    argparser.add_argument('--num_leaves_to_search', type=int, default=1)
    argparser.add_argument('--reorder', type=int, default=1800)
    argparser.add_argument('--num_neighbors', type=int, default=5)
    args = argparser.parse_args()


    min_scale_factor = 0.5
    max_scale_factor = 2.0
    brightness_jitter_range = 0.1
    contrast_jitter_range = 0.1
    saturation_jitter_range = 0.1
    hue_jitter_range = 0.1
    brightness_jitter_probability = 0.5
    contrast_jitter_probability = 0.5
    saturation_jitter_probability = 0.5
    hue_jitter_probability = 0.5
    input_size = args.img_size
    image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([trn.ColorJitter(hue=hue_jitter_probability)], p=hue_jitter_probability),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ])
    shared_train_transform = Compose([
            Resize(size=(input_size, input_size)),
            # RandomHorizontalFlip(probability=0.1),
        ])
    
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
            Resize(size=(input_size, input_size)),
        ])
    
    cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
    image_ids = sorted(os.listdir("/scratch-shared/combined_hbird/mscoco_hbird/train2017/"))
    train_idx, val_idx = train_test_split(np.arange(len(image_ids)), test_size=0.2, random_state=0)
    train_set_coco = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train',  batchsz=args.data_size, k_shot=5, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_train_transform, shared_train_transform),mode_idx=train_idx)
    val_set = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train', batchsz=10, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index,cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), mode_idx=val_idx)
    
    cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
    image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
    train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
    cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl'
    train_set_nyu = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = train_idx, batchsz=args.data_size, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
    val_set = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = val_idx, batchsz=10, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
    data_loaders = [train_set_coco, train_set_nyu]
    combined_data_loader = CombinedMemoryTasksDataLoader(data_loaders)
    for i in range(10):
        combined_data_loader[i]