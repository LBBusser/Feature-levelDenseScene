import yaml
import torch
from omegaconf import OmegaConf
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from collections import OrderedDict
import torch
import sys
import torchvision.transforms as trn
from tqdm import tqdm
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
from data_loader import *
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
import random
import json
import faiss
import argparse
from enum import Enum
# from my_utils import denormalize_video, make_seg_maps
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
import torchvision as tv
from abc import ABC, abstractmethod
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
import torch
sys.path.append('/home/lbusser/taming-transformers/')
from taming.models.vqgan import VQModel, GumbelVQ

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params, sane_index_shape= True)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()


def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  print(indices.shape)
  xrec = model.decode(z)
  return xrec


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--testk_spt', type=int, help='k shot for test support set', default=5)
    argparser.add_argument('--testk_qry', type=int, help='k shot for test query set', default=1)
    argparser.add_argument('--img_size', type=int, help='img_size', default=256)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--data_size', type=int, default=100)
    argparser.add_argument('--test_size', type=int, default=10)
    argparser.add_argument('--val_data_size', type=int, default=10)

    # COCO training / non-episodic image captioning
    argparser.add_argument('--num_epochs', type=int, help='epoch number for training', default=5)
    argparser.add_argument('--batch_size', type=int, help='batch size for few shot training', default=8)
    argparser.add_argument('--test_batch_size', type=int, help='batch size for few shot testing', default=1)
    argparser.add_argument('--coco_annotations_path', type=str, default='/scratch-shared/combined_hbird/mscoco_hbird/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=3)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=2e-04)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=100)
    argparser.add_argument('--trained_model_name', type=str, default='default')
    argparser.add_argument('--patch_size', type=int, default=16)
    args = argparser.parse_args()
    device = set_device()
    
    MODEL = 'dinov2_vitb14'
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
    #----------------------------TRAIN---------------------------------------
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
    
    # cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
    # image_ids = sorted(os.listdir("/scratch-shared/combined_hbird/mscoco_hbird/train2017/"))
    # train_idx, val_idx = train_test_split(np.arange(len(image_ids)), test_size=0.2, random_state=0)
    # train_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train',  setsz=args.data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_train_transform, shared_train_transform),mode_idx=train_idx)
    # val_set_COCO = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode = 'train', setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=input_size, cluster_index= cluster_index,cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform), mode_idx=val_idx)
    # cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
    # image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
    # train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
    # cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl'
    # train_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = train_idx, setsz=args.data_size, k_shot= args.k_spt, k_query=args.k_qry, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
    # val_set_NYU = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", mode_idx = val_idx, setsz=args.val_data_size, k_shot=args.k_spt, k_query=args.k_qry, resize=504, cluster_index= cluster_index_nyu, cluster_assignment= cluster_assignment,  transforms = (image_train_transform, shared_train_transform))
    
    # train_set = CombinedDataset([train_set_COCO, train_set_NYU])
    # val_set = CombinedDataset([val_set_COCO, val_set_NYU])

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                         pin_memory=True, drop_last=True)
    #------------------------------------------TEST------------------------------------
    test_coco_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_index.index')
    test_nyu_cluster_index = faiss.read_index('/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_index.index')
    
    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
                Resize(size=(input_size, input_size)),
            ]) 

    coco_test_set = CocoMemoryTasksDataLoader("/scratch-shared/combined_hbird/mscoco_hbird",'val', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    nyu_test_set = NYUMemoryTasksDataLoader("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", 'test', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_nyu_cluster_index, cluster_assignment='/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    kp_test_set = KeyPointMemoryTasksDataLoader("/scratch-shared/combined_hbird/mscoco_hbird", 'val', args.test_size, args.testk_spt, args.testk_qry, args.img_size, cluster_index= test_coco_cluster_index, cluster_assignment= '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl', transforms = (image_val_transform, shared_val_transform))
    test_set = CombinedDataset([coco_test_set, nyu_test_set, kp_test_set])

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    config_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/vqgan/configs/config.yaml"
    ckpt_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/vqgan/checkpoints/model.ckpt"

    config = load_config(config_path, display=False)
    model = load_vqgan(config, ckpt_path)
    with torch.no_grad():
      for i, (support_x, support_y, query_x, query_y) in enumerate(test_loader):
        query_y = query_y.squeeze(0)
        rec = reconstruct_with_vqgan(query_y, model)
        torch.save(query_y, 'query_y_test.pt')
        torch.save(rec, 'rec_test.pt')
        break