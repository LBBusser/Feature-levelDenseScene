from collections import OrderedDict
import torch
import torchvision.transforms as trn
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import wandb
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from image_transformations import RandomResizedCrop, RandomHorizontalFlip, Compose
import random
import json
from enum import Enum
# from my_utils import denormalize_video, make_seg_maps
from torch.utils.data.distributed import DistributedSampler as DistributedSampler

# import video_transformations
import sys
import h5py
import shutil
import tarfile
import zipfile
import requests
from torchvision.datasets.utils import download_url
import torchvision as tv
from abc import ABC, abstractmethod
import scipy.io as sio
from torchvision.datasets import CIFAR10


project_name = "MSCOCO_hbird_eval_stuff"





torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)





class Dataset(torch.nn.Module):
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass
    
    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

class MSCOCODataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, shared_transform = None, subset_indices= None, task = "normal"):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of 'train' or 'val' to specify the split of dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.transform = transform
        self.shared_transform = shared_transform
        if self.task == "stuff":
            self.images_dir = os.path.join(root_dir, split + '2017')
            self.masks_dir = os.path.join(root_dir, "stuff_" + split + '2017_pixelmaps')
        elif self.task == "normal":
            self.images_dir = os.path.join(root_dir, split + '2017')
            self.masks_dir = os.path.join(root_dir, split + '2017')
        else:
            self.images_dir = os.path.join(root_dir, split + '2017')
            self.masks_dir = os.path.join(root_dir, split + '2017')
        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))
      

        if subset_indices is not None:
            self.images = [self.images[i] for i in subset_indices]
            self.masks = [self.masks[i] for i in subset_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        if self.transform:
            image = self.transform(image)
        if self.shared_transform:
            image, mask = self.shared_transform(image, mask)
        if self.task == "stuff":
            mask = mask - 91 #Because the stuff labels begin from 92. 182-91=    
        return image, mask
class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(
        self,
        root: str,
        mode: str = "train",
        rgb_transform=None,
        depth_transform=None,
        shared_transform = None,  
    ):
      
        super().__init__()
        self.mode = mode
        self.root = root
        if mode == "train":
            self.global_min = 18  
            self.global_max = 255  
        elif mode == "test":
            self.global_min = 713 
            self.global_max = 9986
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.shared_transform = shared_transform
      
        self.csv_file = os.path.join(self.root, f"data/nyu2_{self.mode}.csv")
    
        self.dataframe = pd.read_csv(self.csv_file)
        
    
    def __getitem__(self, idx: int):
        img_name = os.path.join(self.root, self.dataframe.iloc[idx, 0])
        depth_name = os.path.join(self.root, self.dataframe.iloc[idx, 1])

        image = Image.open(img_name)
        mask = np.array(Image.open(depth_name))
       
        if self.rgb_transform:
            image = self.rgb_transform(image)
        mask = self.normalize_depth_map(mask)
        if self.shared_transform:
            image, mask = self.shared_transform(image, mask)
    
        return image, mask
    
    def normalize_depth_map(self, depth_map):
        depth_array = np.array(depth_map)
        depth_array = depth_array.astype(np.float32)
        depth_array = (depth_array - self.global_min) / (self.global_max - self.global_min)
        return Image.fromarray(depth_array)
    
    def __len__(self):
        return len(self.dataframe)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                for part, transform in zip(
                    ["rgb", "seg13", "sn", "depth"],
                    [
                        self.rgb_transform,
                        self.seg_transform,
                        self.sn_transform,
                        self.depth_transform,
                    ],
                ):
                    if transform is None:
                        continue
                    path = os.path.join(self.root, f"{split}_{part}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        if self.rgb_transform is not None:
            download_rgb(self.root)
        if self.seg_transform is not None:
            download_seg(self.root)
        # if self.sn_transform is not None:
        #     download_sn(self.root)
        if True:
            download_depth(self.root)
        print("Done!")

class COCODataModule():
    """
    DataModule for MSCOCO dataset

    Args:
        batch_size (int): Batch size
        train_transform (torchvision.transforms): Transform for training set
        val_transform (torchvision.transforms): Transform for validation set
        test_transform (torchvision.transforms): Transform for test set (if applicable)
        dir (str): Path to dataset
        num_workers (int): Number of workers for dataloader
    """

    def __init__(self, batch_size, train_transform, val_transform, test_transform, dir="/scratch-shared/mscoco_hbird/", num_workers=0, task = "normal") -> None:
        self.batch_size = batch_size
        self.dir = dir
        self.task = task
        self.train_transform = train_transform['img']
        self.val_transform = val_transform['img']
        self.test_transform = test_transform['img']
        self.num_workers = num_workers
        self.shared_train_transform = train_transform['shared']

    def setup(self):
        subset_indices_train = list(range(2000))
        subset_indices_val  = list(range(1000))
        print("MSCOCO", self.task, "segmentation")
        self.train_dataset = MSCOCODataset(self.dir, split = "train", transform=self.train_transform, shared_transform=self.shared_train_transform, subset_indices=subset_indices_train, task = self.task)
        self.val_dataset = MSCOCODataset(self.dir, split = "val", transform=self.val_transform, shared_transform=self.shared_train_transform, subset_indices=subset_indices_val, task = self.task)
        # Test dataset setup can be added here if needed

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    # Add a get_test_dataloader method if you have a test dataset

    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    # Add a get_test_dataset_size method if you have a test dataset

    def get_module_name(self):
        return "COCODataModule"
    
    def get_num_classes(self):
        # Return the number of classes in the MSCOCO dataset (commonly 80 for object detection)
        if self.task == "stuff":
            return 92
        elif self.task == "normal":
            return 80
        return 80
    
class NYUv2DataModule():
    """ 
    DataModule for Pascal NYUv2 dataset

    Args:
        batch_size (int): batch size
        train_transform (torchvision.transforms): transform for training set
        val_transform (torchvision.transforms): transform for validation set
        test_transform (torchvision.transforms): transform for test set
        dir (str): path to dataset
        year (str): year of dataset
        split (str): split of dataset
        num_workers (int): number of workers for dataloader

    """

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/scratch-shared/NYUhbird/nyu_data", num_workers=0, download=False) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = train_transform["shared"]
        self.shared_val_transform = val_transform["shared"]
        self.shared_test_transform = test_transform["shared"]

    def setup(self):
        self.train_dataset = NYUv2(self.dir, mode="train", rgb_transform=self.image_train_transform,shared_transform=self.shared_train_transform)
        self.val_dataset = NYUv2(self.dir, mode="test", rgb_transform=self.image_val_transform,shared_transform=self.shared_val_transform)
        self.test_dataset = NYUv2(self.dir,mode="test", rgb_transform=self.image_test_transform ,shared_transform=self.shared_test_transform)
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "NYUv2DataModule"
    
    def get_num_classes(self):
        #only use for segmentation...
        return 9512
    
    
class VOCDataset(Dataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'JPEGImages')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'ImageSets/Segmentation/')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.return_masks:
            mask = Image.open(self.masks[index])
        if self.image_set == "val":
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                res = self.transforms(img, mask)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)



class PascalVOCDataModule():
    """ 
    DataModule for Pascal VOC dataset

    Args:
        batch_size (int): batch size
        train_transform (torchvision.transforms): transform for training set
        val_transform (torchvision.transforms): transform for validation set
        test_transform (torchvision.transforms): transform for test set
        dir (str): path to dataset
        year (str): year of dataset
        split (str): split of dataset
        num_workers (int): number of workers for dataloader

    """

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="/scratch-shared/tmp.xc2nBiDuTi/VOCdevkit/VOC2012/", num_workers=0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = train_transform["shared"]
        self.shared_val_transform = val_transform["shared"]
        self.shared_test_transform = test_transform["shared"]

    def setup(self):
        download = False
        if os.path.isdir(self.dir) == False:
            download = True
        self.train_dataset = VOCDataset(self.dir, image_set="train", transform=self.image_train_transform, target_transform=self.target_train_transform, transforms=self.shared_train_transform, return_masks=True)
        self.val_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_val_transform, target_transform=self.target_val_transform, transforms=self.shared_val_transform, return_masks=True)
        self.test_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_test_transform, target_transform=self.target_test_transform, transforms=self.shared_test_transform, return_masks=True)
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    
    def get_num_classes(self):
        return 21
    


class VideoDataModule():

    def __init__(self, name, path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers=0, world_size=1, rank=0):
        super().__init__()
        self.name = name
        self.path_dict = path_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_directory = self.path_dict["class_directory"]
        self.annotations_directory = self.path_dict["annotation_directory"]
        self.meta_file_path = self.path_dict["meta_file_path"]
        self.num_clip_frames = num_clip_frames
        self.sampling_mode = sampling_mode
        self.regular_step = regular_step
        self.num_clips = num_clips
        self.world_size = world_size
        self.rank = rank
        self.sampler = None
        self.data_loader = None

    
    def setup(self, transforms_dict):
        data_transforms = transforms_dict["data_transforms"]
        target_transforms = transforms_dict["target_transforms"]
        shared_transforms = transforms_dict["shared_transforms"]
        if self.name == "timetytvos":
            self.dataset = TimeTYVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "ytvos":
            self.dataset = YVOSDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        elif self.name == "kinetics":
            self.dataset = Kinetics(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        else:
            self.dataset = VideoDataset(self.class_directory, self.annotations_directory, self.sampling_mode, self.num_clips, self.num_clip_frames, data_transforms, target_transforms, shared_transforms, self.meta_file_path, self.regular_step)
        print(f"Dataset size : {len(self.dataset)}")
    
    def make_data_loader(self, shuffle=True):
        if self.world_size > 1:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, sampler=self.sampler)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)
    
    def get_data_loader(self):
        return self.data_loader
    

    
def test_mscoco_data_module(logger):
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

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])
    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])
        
    
    # image_train_transform = trn.Compose([trn.Resize((448, 448)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    # target_train_transform = trn.Compose([trn.Resize((448, 448), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    dataset = COCODataModule(batch_size=1, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms, task = "stuff")
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    # test_dataloader = dataset.get_test_dataloader()
    print(f"Train size : {len(dataset.train_dataset)}")
    print(f"Val size : {len(dataset.val_dataset)}")
    # print(f"Test size : {len(dataset.test_dataset)}")
    print(f"Train dataloader size : {len(train_dataloader)}")
    print(f"Val dataloader size : {len(val_dataloader)}")
    # print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y) in enumerate(val_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        y[y==165] = 0 #For stuff segmentation.
        ## log image
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i ==10:
            break


def test_pascal_data_module(logger):
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

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])

    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])
        
    
    # image_train_transform = trn.Compose([trn.Resize((448, 448)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    # target_train_transform = trn.Compose([trn.Resize((448, 448), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    dataset = PascalVOCDataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(f"Train size : {len(dataset.train_dataset)}")
    print(f"Val size : {len(dataset.val_dataset)}")
    print(f"Test size : {len(dataset.test_dataset)}")
    print(f"Train dataloader size : {len(train_dataloader)}")
    print(f"Val dataloader size : {len(val_dataloader)}")
    print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y) in enumerate(val_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        ## log image
        classes = torch.unique((y * 255).long())
        print(f"Number of classes : {classes}")
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i == 5:
            break

def test_nyuv2_data_module(logger):
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

    # Create the transformation
    image_train_transform = trn.Compose([
        # trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        # trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        # trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        # trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])
    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])
        
    
    # image_train_transform = trn.Compose([trn.Resize((448, 448)), trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    # target_train_transform = trn.Compose([trn.Resize((448, 448), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_transform}
    dataset = NYUv2DataModule(batch_size=4, train_transform=train_transforms, val_transform=train_transforms, test_transform=train_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(f"Train size : {len(dataset.train_dataset)}")
    print(f"Val size : {len(dataset.val_dataset)}")
    print(f"Test size : {len(dataset.test_dataset)}")
    print(f"Train dataloader size : {len(train_dataloader)}")
    print(f"Val dataloader size : {len(val_dataloader)}")
    print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y) in enumerate(train_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")

        ## log image
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i ==10:
            break

# def test_video_data_module(logger):
#     rand_color_jitter = video_transformations.RandomApply([video_transformations.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8)
#     data_transform_list = [rand_color_jitter, video_transformations.RandomGrayscale(), video_transformations.RandomGaussianBlur()]
#     data_transform = video_transformations.Compose(data_transform_list)
#     video_transform_list = [video_transformations.ClipToTensor()]
#     target_transform = video_transformations.Compose(video_transform_list)
#     video_transform = video_transformations.TimeTTransform([224, 96], [1, 4], [0.25, 0.05], [1., 0.25], 1, 0.01, 1)
#     num_clips = 1
#     batch_size = 8
#     num_workers = 4
#     num_clip_frames = 4
#     regular_step = 1
#     transformations_dict = {"data_transforms": video_transform, "target_transforms": target_transform, "shared_transforms": None}
#     prefix = "/ssdstore/ssalehi/dataset"
#     data_path = os.path.join(prefix, "train1/JPEGImages/")
#     annotation_path = os.path.join(prefix, "train1/Annotations/")
#     meta_file_path = os.path.join(prefix, "train1/meta.json")
#     path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
#     sampling_mode = SamplingMode.DENSE
#     video_data_module = VideoDataModule("timetytvos", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers)
#     video_data_module.setup(transformations_dict)
#     data_loader = video_data_module.get_data_loader()
#     logging_directory = "data_loader_log/"

#     if os.path.exists(logging_directory):
#         os.system(f'rm -r {logging_directory}')
#     os.makedirs(logging_directory)

#     for i, train_data in enumerate(data_loader):
#         datum, labels, annotations = train_data
#         print("===========================")
#         print("")
#         annotations = annotations.squeeze(1)
#         datum = datum.squeeze(1)
#         datum = denormalize_video(datum)
#         print((torch.unique(annotations)))
#         print(datum.shape)
#         print(annotations.shape)
#         # visualize_sampled_videos(datum, "data_loader_log/", f"test_{i}.avi")
#         # visualize_sampled_videos(annotations, "data_loader_log/", f"test_anotations_{i}.avi")
#         make_seg_maps(datum, annotations, logging_directory, f"test_seg_maps_{i}.avi")



    

if __name__ == "__main__":
    ## init wandb
    logger = wandb.init(project=project_name, group="data_loader", tags="NYUv2DataModule", job_type="eval")
    ## test data module
    # test_pascal_data_module(logger)
    ## finish wandb
    # logger.finish()
    test_mscoco_data_module(logger)
    # NYUv2(root="/somepath/NYUv2", download=True, 
    #   rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t)

    # get_file_path("/ssdstore/ssalehi/dataset/val1/JPEGImages/")
    # test_video_data_module(logger)


#--------------------------------------------------------------------------


# class NormalSubsetDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, subset_indices):
#         self.dataset = dataset
#         self.subset_indices = subset_indices

#     def __getitem__(self, index):
#         return self.dataset[self.subset_indices[index]]

#     def __len__(self):
#         return len(self.subset_indices)


# class Cifar10_Handler(Dataset):
#     def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
#         self.batch_size = batch_size
#         self.normal_classes = normal_classes
#         self.num_workers = num_workers
#         self.device = device
#         self.dataset_name = "Cifar10"
#         self.num_classes = 10
#         self.transform = transformations
#         self.val_transform = val_transformations
#         self.train_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=True, download=True, transform=self.transform)
#         self.test_dataset = CIFAR10(root="/ssdstore/ssalehi/cifar", train=False, download=True, transform=self.val_transform )
#         self.binarize_test_labels()
#         ## split the dataset to train and validation
#         normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
#         self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
#         print("Normal Subset Size: ", len(self.normal_dataset))
#         self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
#         self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
#         self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
#         self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#     def get_train_loader(self):
#         return self.train_loader

#     def get_test_loader(self):
#         return self.test_loader

#     def get_val_loader(self):
#         return self.val_loader

#     def get_train_dataset(self):
#         return self.train_dataset

#     def get_test_dataset(self):
#         return self.test_dataset

#     def get_val_dataset(self):
#         return self.val_dataset

#     def get_num_classes(self):
#         return self.num_classes

#     def get_dataset_name(self):
#         return self.dataset_name
    
#     def get_normal_classes(self):
#         return self.normal_classes

#     def get_normal_sebset_indices(self):
#         normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
#         normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
#         return normal_subset_indices, normal_subset
    
#     def binarize_test_labels(self):
#         for i, label in enumerate(self.test_dataset.targets):
#             if label in self.normal_classes:
#                 self.test_dataset.targets[i] = 0
#             else:
#                 self.test_dataset.targets[i] = 1
   
    



# class SamplingMode(Enum):
#     UNIFORM = 0
#     DENSE = 1
#     Full = 2
#     Regular = 3


# def get_file_path(classes_directory):
#     ## find all the folders and add all the files in the folders to a dict with keys are name of the file and values are the path to the file

#     folder_file_path = {} ## key is the directory_path and value are the files in the directory
#     for root, dirs, files in os.walk(classes_directory):
#         for file in sorted(files):
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 if root not in folder_file_path:
#                     folder_file_path[root] = []
#                 folder_file_path[root].append(file)
    
#     return dict(sorted(folder_file_path.items()))



# def make_categories_dict(meta_dict, name):
#     category_list = []
#     if "ytvos" in name:
#         video_name_list = meta_dict["videos"].keys()
#         for name in video_name_list:
#             obj_list = meta_dict["videos"][name]["objects"].keys()
#             for obj in obj_list:
#                 if meta_dict["videos"][name]["objects"][obj]["category"] not in category_list:
#                     category_list.append(meta_dict["videos"][name]["objects"][obj]["category"])
#         category_list = sorted(list(OrderedDict.fromkeys(category_list)))
#         category_ditct = {k: v+1 for v, k in enumerate(category_list)} ## zero is always for the background
#     return category_ditct



# def map_instances(data, meta, category_dict):
#     bs, fs, h, w = data.shape
#     for i, datum in enumerate(data):
#         for j, frame in enumerate(data):
#             objects = torch.unique(frame)
#             for k, obj in enumerate(objects):
#                 if int(obj.item()) == 0:
#                     continue
#                 frame[frame == obj] = category_dict[meta[str(int(obj.item()))]["category"]]
#     return data


# class VideoDataset(torch.utils.data.Dataset):
#     ## The data loader gets training sample and annotations direcotories, sampling mode, number of clips that is being sampled of each training video, number of frames in each clip
#     ## and number of labels for each training clip. 
#     ## Note that the number of annotations should be exactly similar to the number of frames existing in the training path.
#     ## Frame_transform is a function that transforms the frames of the video. It is applied to each frame of the video.
#     ## Target_transform is a function that transforms the annotations of the video. It is applied to each annotation of the video.
#     ## Video_transform is a function that transforms the whole video. It is applied to both frames and annotations of the video.
#     ## The same set of transformations is applied to the clips of the video.

#     def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
#         super().__init__()
#         self.train_dict = get_file_path(classes_directory)
#         self.train_dict_lenghts = {}
#         self.find_directory_length()
#         if (annotations_directory != "") and (os.path.exists(annotations_directory)):
#             self.train_annotations_dict = get_file_path(annotations_directory)
#             self.use_annotations = True
#         else:
#             self.use_annotations = False
#             print("Because there is no annotation directory, only training samples have been loaded.")
#         if (meta_file_directory is not None):
#             if (os.path.exists(meta_file_directory)):
#                 print("Meta file has been read.")
#                 file = open(meta_file_directory)
#                 self.meta_dict = json.load(file)
#             else:
#                 self.meta_dict = None
#                 print("There is no meta file.")
#         else:
#             print("Meta option is off.")
#             self.meta_dict = None
         
#         self.sampling_mode = sampling_mode
#         self.num_clips = num_clips
#         self.num_frames = num_frames
#         self.frame_transform = frame_transform
#         self.target_transform = target_transform
#         self.video_transform = video_transform
#         self.regular_step = regular_step
#         self.keys = list(self.train_dict.keys())
#         if self.use_annotations:
#             self.annotation_keys = list(self.train_annotations_dict.keys())
        
#     def __len__(self):
#         return len(self.keys)

    
#     def find_directory_length(self):
#         for key in self.train_dict:
#             self.train_dict_lenghts[key] = len(self.train_dict[key])

    
#     def read_clips(self, path, clip_indices):
#         clips = []
#         files = sorted(glob.glob(path + "/" + "*.jpg"))
#         if len(files) == 0:
#             files = sorted(glob.glob(path + "/" + "*.png"))
#         for i in range(len(clip_indices)):
#             images = []
#             for j in clip_indices[i]:
#                 # frame_path = path + "/" + f'{j:05d}' + ".jpg"
#                 frame_path = files[j]
#                 if not os.path.exists(frame_path):
#                     frame_path = path + "/" + f'{j:05d}' + ".png"
#                 if not os.path.exists(frame_path): ## This is for kinetics dataset
#                     frame_path = path + "/" + f'img_{(j + 1):05d}' + ".jpg" 
#                 if not os.path.exists(frame_path): ## This is for kinetics dataset
#                     frame_path = path + "/" + f'frame_{(j + 1):010d}' + ".jpg" 

#                 images.append(Image.open(frame_path))
#             clips.append(images)
#         return clips
    
    
#     def generate_indices(self, size, sampling_num):
#         indices = []
#         for i in range(self.num_clips):
#             if self.sampling_mode == SamplingMode.UNIFORM:
#                     if size < sampling_num:
#                         ## sample repeatly
#                         idx = random.choices(range(0, size), k=sampling_num)
#                     else:
#                         idx = random.sample(range(0, size), sampling_num)
#                     idx.sort()
#                     indices.append(idx)
#             elif self.sampling_mode == SamplingMode.DENSE:
#                     base = random.randint(0, size - sampling_num)
#                     idx = range(base, base + sampling_num)
#                     indices.append(idx)
#             elif self.sampling_mode == SamplingMode.Full:
#                     indices.append(range(0, size))
#             elif self.sampling_mode == SamplingMode.Regular:
#                 if size < sampling_num * self.regular_step:
#                     step = size // sampling_num
#                 else:
#                     step = self.regular_step
#                 base = random.randint(0, size - (sampling_num * step))
#                 idx = range(base, base + (sampling_num * step), step)
#                 indices.append(idx)
#         return indices
    

#     def read_batch(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
#         size = self.train_dict_lenghts[path]
#         # sampling_num = size if self.num_frames > size else self.num_frames
#         clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
#         sampled_clips = self.read_clips(path, clip_indices)
#         annotations = []
#         sampled_clip_annotations = []
#         if annotation_path is not None:
#             sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
#             if target_transformation is not None:
#                 for i in range(len(sampled_clip_annotations)):
#                     sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
#         if frame_transformation is not None:
#             for i in range(len(sampled_clips)):
#                 try:
#                     sampled_clips[i] = frame_transformation(sampled_clips[i])
#                 except:
#                     print("Error in frame transformation")
#         if video_transformation is not None:
#             for i in range(len(sampled_clips)):
#                 if len(sampled_clip_annotations) != 0:
#                     sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
#                 else:
#                     sampled_clips[i] = video_transformation(sampled_clips[i])
#         sampled_data = torch.stack(sampled_clips)
#         if len(sampled_clip_annotations) != 0:
#             sampled_annotations = torch.stack(sampled_clip_annotations)
#             if sampled_annotations.size(0) != 0:
#                 sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
#                 if sampled_annotations.shape[2] == 1:
#                     sampled_annotations = sampled_annotations.squeeze(2)
#         else:
#             sampled_annotations = torch.empty(0)
#         ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
#         return sampled_data, sampled_annotations


#     def read_batch_with_new_transforms(self, path, annotation_path=None, frame_transformation=None, target_transformation=None, video_transformation=None):
#         size = self.train_dict_lenghts[path]
#         # sampling_num = size if self.num_frames > size else self.num_frames
#         clip_indices = self.generate_indices(self.train_dict_lenghts[path], self.num_frames)
#         sampled_clips = self.read_clips(path, clip_indices)
#         annotations = []
#         sampled_clip_annotations = []
#         labels_dict = {}
#         if annotation_path is not None:
#             sampled_clip_annotations = self.read_clips(annotation_path, clip_indices)
#             if target_transformation is not None:
#                 for i in range(len(sampled_clip_annotations)):
#                     sampled_clip_annotations[i] = target_transformation(sampled_clip_annotations[i])
#         if frame_transformation is not None:
#             for i in range(len(sampled_clips)):
#                 # try:
#                 multi_crops, labels_dict = frame_transformation(sampled_clips[i])
#                 # except:
#                 #     print("Error in frame transformation")
#         if video_transformation is not None:
#             for i in range(len(sampled_clips)):
#                 if len(sampled_clip_annotations) != 0:
#                     sampled_clips[i], sampled_clip_annotations[i] = video_transformation(sampled_clips[i], sampled_clip_annotations[i])
#                 else:
#                     sampled_clips[i] = video_transformation(sampled_clips[i])
#         # sampled_data = torch.stack(sampled_clips)
#         if len(sampled_clip_annotations) != 0:
#             sampled_annotations = torch.stack(sampled_clip_annotations)
#             if sampled_annotations.size(0) != 0:
#                 sampled_annotations = (255 * sampled_annotations).type(torch.uint8) 
#                 if sampled_annotations.shape[2] == 1:
#                     sampled_annotations = sampled_annotations.squeeze(2)
#         else:
#             sampled_annotations = torch.empty(0)
#         ## squeezing the annotations to be in the shape of (num_sample, num_clips, num_frames, height, width)
#         return multi_crops, labels_dict, sampled_annotations
    

#     def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
#         # idx = 0  ## This is a hack to make the code work with the dataloader.
#         # idx = random.randint(0, 5)
#         video_path = self.keys[idx]
#         dir_name = video_path.split("/")[-1]
#         annotations = None
#         annotations_path = None
#         if (self.use_annotations):
#             annotations_path = self.annotation_keys[idx]
#             # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
#         data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
#         if self.meta_dict is not None:
#             category_dict = make_categories_dict(self.meta_dict, "davis")
#             meta_dict = self.meta_dict["videos"][dir_name]["objects"]
#             annotations = map_instances(annotations, meta_dict, category_dict)

#         return data, annotations



# class YVOSDataset(VideoDataset):

#     def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
#         super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        
#         self.category_dict = make_categories_dict(self.meta_dict, "ytvos")

#     def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
#         video_path = self.keys[idx]
#         dir_name = video_path.split("/")[-1]
#         annotations = None
#         annotations_path = None
#         if (self.use_annotations):
#             annotations_path = self.annotation_keys[idx]
#             # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
#         data, annotations = self.read_batch(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
#         if self.meta_dict is not None:
#             annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

#         # else:
#             # annotations = convert_to_indexed_RGB(annotations)
#         return data, annotations



# class TimeTYVOSDataset(VideoDataset):

#     def __init__(self, classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
#         super().__init__(classes_directory, annotations_directory, sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)
        
#         self.category_dict = make_categories_dict(self.meta_dict, "timetytvos")

#     def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
#         video_path = self.keys[idx]
#         dir_name = video_path.split("/")[-1]
#         annotations = None
#         annotations_path = None
#         if (self.use_annotations):
#             annotations_path = self.annotation_keys[idx]
#             # annotations = self.read_annotations(annotations_path, self.target_transform, indices=indices)
#         data, labels, annotations = self.read_batch_with_new_transforms(video_path, annotations_path, self.frame_transform, self.target_transform ,self.video_transform)   
#         if self.meta_dict is not None:
#             annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

#         # else:
#             # annotations = convert_to_indexed_RGB(annotations)
#         return data, labels, annotations


# class Kinetics(VideoDataset):

#     def __init__(self, classes_directory, sampling_mode, num_clips, num_frames, frame_transform=None, target_transform=None, video_transform=None, meta_file_directory=None, regular_step=1) -> None:
#         super().__init__(classes_directory, "", sampling_mode, num_clips, num_frames, frame_transform, target_transform, video_transform, meta_file_directory, regular_step=regular_step)

#     def __getitem__(self, idx): ### (B, num_clips, num_frames, C, H, W) returns None if the annotation flag is off. Be careful when loading the data.
#         video_path = self.keys[idx]
#         dir_name = video_path.split("/")[-1]
#         annotations = None
#         annotations_path = None
#         data, annotations = self.read_batch_with_new_transforms(video_path, None, self.frame_transform, self.target_transform ,self.video_transform)   
#         if self.meta_dict is not None:
#             annotations = map_instances(annotations, self.meta_dict["videos"][dir_name]["objects"], self.category_dict)

#         # else:
#             # annotations = convert_to_indexed_RGB(annotations)
#         return data, annotations
