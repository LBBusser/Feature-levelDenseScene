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
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def create_colormap(num_classes=80):
  np.random.seed(10)
  colors = np.random.randint(0, 256, (num_classes+1, 3), dtype=np.uint8)
  colors[0] = [0, 0, 0]
  return colors

def apply_custom_colormap(mask, colormap):
    """
    Apply a custom colormap to a mask
    :param mask: numpy array of the mask [H, W]
    :param colormap: numpy array of shape [num_classes+1, 3]
    :return: Colored mask as a numpy array [H, W, 3]
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(colormap.shape[0]):
        color_mask[mask == i] = colormap[i]/255.0
    return torch.from_numpy(color_mask)

def find_closest_class(rgb_values, colormap):
    """
    Find the closest class for each RGB value in the predicted mask based on the colormap.
    
    :param rgb_values: numpy array of RGB values [H, W, 3]
    :param colormap: numpy array of the colormap [num_classes+1, 3]
    :return: Class map as a numpy array [H, W]
    """
    # Calculate the difference between the RGB values and each color in the colormap
    diff = np.linalg.norm(colormap - rgb_values[:, :, np.newaxis], axis=3)
    print(diff.shape)
    # Find the index of the minimum difference which corresponds to the closest class
    class_map = np.argmin(diff, axis=2)
    return class_map


if __name__ == "__main__":
    input_size = 504
    eval_spatial_resolution = input_size // 14
    # Define transformation parameters
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

    image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(hue=hue_jitter_probability)], p=hue_jitter_probability),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ])
    shared_train_transform = Compose([
            Resize(size=(input_size, input_size)),
            # RandomHorizontalFlip(probability=0.1),
        ])
    cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_index.index")
    # cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_100_cluster_results/cluster_index.index")
    cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_100_cluster_results/cluster_assignments.pkl'
    few_shot_ds = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="val", batchsz=100, k_shot=3, k_query=1, resize=504, cluster_index= cluster_index, cluster_assignment=cluster_assignment, transforms = (image_train_transform, shared_train_transform))
    # few_shot_ds = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="test", batchsz=10, k_shot=3, k_query=1, resize=504, cluster_index= cluster_index_nyu, cluster_assignment=cluster_assignment, transforms = (image_train_transform, shared_train_transform))
    dl = DataLoader(few_shot_ds, batch_size=1, shuffle=False)
    config_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/f8_config.yaml"
    ckpt_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/model.ckpt"
    colors = create_colormap()
    config = load_config(config_path, display=False)
    model = load_vqgan(config, ckpt_path)
    with torch.no_grad():
      for i, (support_x, support_y, query_x, query_y) in enumerate(dl):
        print(query_y[0].unique())
        # torch.save(query_y[0].squeeze().numpy(), 'query_test.pt')
        colored_mask = apply_custom_colormap(query_y.squeeze(), colors)
        print(colored_mask.unique())
        # torch.save(colored_mask.permute(2,0,1).unsqueeze(0),'query_test.pt')
        
        pred = reconstruct_with_vqgan(colored_mask.permute(2,0,1).unsqueeze(0), model) 
        out = find_closest_class((pred*255).squeeze().permute(1,2,0).numpy(), colors)
        print(np.unique(out))

        # torch.save(out, 'pred_test.pt')
        break