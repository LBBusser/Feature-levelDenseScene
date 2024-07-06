import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from torchvision.transforms import GaussianBlur
from typing import List
import io
import os, sys
import requests
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import glob
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import List
from torchvision.utils import draw_segmentation_masks
import cv2
from PIL import Image
import matplotlib
import numpy as np
import wandb
from torch import distributed as dist


def show_trainable_paramters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)



def process_attentions(attn_batch, spatial_res, threshold = 0.5, blur_sigma = 0.6):
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    # attns_processed = torch.cat(attns_group, dim = 0)
    attns_processed = sum(attn_batch[:, i] * 1 / attn_batch.size(1) for i in range(attn_batch.size(1)))
    attentions = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()



def preprocess(imgs):
    img_group = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = T.ToPILImage()(img.cpu())
        target_image_size = 224
        s = min(img.size)
        
        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        img_group.append(map_pixels(img))
    return torch.cat(img_group, dim = 0)



def cosine_scheduler(base_value: float, final_value: float, max_iters: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(max_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def set_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)
            

def localize_objects(input_img, cluster_map):

    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black"]
    ticks = np.unique(cluster_map.flatten()).tolist()

    dc = np.zeros(cluster_map.shape)
    for i in range(cluster_map.shape[0]):
        for j in range(cluster_map.shape[1]):
            dc[i, j] = ticks.index(cluster_map[i, j])

    colormap = matplotlib.colors.ListedColormap(colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 3))
    # plt.figure(figsize=(5,3))
    im = axes[0].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    cbar = fig.colorbar(im, ticks=range(len(colors)))
    axes[1].imshow(input_img)
    axes[2].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    axes[2].imshow(input_img, alpha=0.5)
    # plt.show(block=True)
    # plt.close()
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return np.asarray(Image.open(buffer))