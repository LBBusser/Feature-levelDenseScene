from collections import OrderedDict
import torch
import sys
import torchvision.transforms as trn
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from models import FeatureExtractorBeta as FeatureExtractor
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
import random
import json
from enum import Enum
# from my_utils import denormalize_video, make_seg_maps
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
from torchvision.datasets.utils import download_url
import torchvision as tv
from abc import ABC, abstractmethod
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
import torch



project_name = "test_hbird_data"





torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)





class Dataset_custom(torch.nn.Module):
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

class MSCOCODataset(Dataset_custom):
    def __init__(self, root_dir, split='train', transform=None, shared_transform = None, subset_indices= None, task = "normal", annotation_file = None, cluster_images = None, memory_bank = False):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of 'train' or 'val' to specify the split of dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.memory_bank = memory_bank
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.transform = transform
        self.shared_transform = shared_transform
        if self.memory_bank:
            same_cluster_masks = [path.replace('/train2017', '/train_masks2017').replace('.jpg', '.png') for path in cluster_images]
            self.images_dir = os.path.join(root_dir, split + '2017')
            self.masks_dir = os.path.join(root_dir, split + '_masks2017')
            self.images = sorted(cluster_images)
            self.masks = sorted(same_cluster_masks)
            
        else:
            if self.task == "stuff":
                self.images_dir = os.path.join(root_dir, split + '2017')
                self.masks_dir = os.path.join(root_dir, "stuff_" + split + '2017_pixelmaps')
                self.images = sorted(os.listdir(self.images_dir))
                self.masks = sorted(os.listdir(self.masks_dir))
            else:
                self.images_dir = os.path.join(root_dir, split + '2017')
                self.masks_dir = os.path.join(root_dir, split + '_masks2017')
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
        mask = Image.open(mask_name).convert('L')
        
        if self.transform:
            image = self.transform(image)
        if self.shared_transform:
            image, mask = self.shared_transform(image, mask)
        # if self.task == "stuff":
        #     mask = mask - 91 #Because the stuff labels begin from 92.  
        return image, mask, img_name
    
class NYUv2(Dataset_custom):

    def __init__(self, root_dir, split='train', transform=None, shared_transform = None, subset_index = None, cluster_images = None, memory_bank = False):
        self.memory_bank = memory_bank
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.shared_transform = shared_transform
        if self.split == "train":
            self.global_min = 18  
            self.global_max = 255  
        elif self.split == "test":
            self.global_min = 713 
            self.global_max = 9986
        if self.memory_bank:
            same_cluster_masks = [path.replace("_colors", "_depth") for path in cluster_images]
            self.images = sorted(cluster_images)
            self.masks = sorted(same_cluster_masks)
        self.csv_file = os.path.join(self.root_dir, f"data/nyu2_{self.split}.csv")
        self.dataframe = pd.read_csv(self.csv_file)
        if subset_index is not None:
            self.dataframe = pd.read_csv(self.csv_file, nrows=subset_index)
        else:
            self.dataframe = pd.read_csv(self.csv_file)
    
    def normalize_depth_map(self, depth_map):
        depth_array = np.array(depth_map)
        depth_array = depth_array.astype(np.float32)
        depth_array = (depth_array - self.global_min) / (self.global_max - self.global_min)
        return depth_array
    
    def __getitem__(self, idx: int):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        depth_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(depth_name)
        if self.transform:
            image = self.transform(image)
        if self.shared_transform:
            image, mask = self.shared_transform(image, mask)
        mask = self.normalize_depth_map(mask)
        return image, mask, img_name
    def __len__(self):
        if self.memory_bank:
            return len(self.images)
        return len(self.dataframe)


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

    def __init__(self, batch_size, train_transform, val_transform, dir="/scratch-shared/mscoco_hbird/", num_workers=0, task = "normal", annotation_dir = None, cluster_images = None, memory_bank = False) -> None:
        self.batch_size = batch_size
        self.dir = dir
        self.task = task
        self.train_transform = train_transform['train']
        self.val_transform = val_transform['val']
        self.num_workers = num_workers
        self.shared_train_transform = train_transform['shared']
        self.shared_val_transform = val_transform['shared']
        self.annotation_file = annotation_dir
        self.memory_bank = memory_bank
        self.cluster_images = cluster_images

    def setup(self):
        subset_indices_train = None
        subset_indices_val = None
        # subset_indices_train = list(range(175))
        subset_indices_val  = list(range(100))
        print("MSCOCO", self.task, "segmentation")
        print("Dataset loaded with memory bank mode: ", self.memory_bank)
        self.train_dataset = MSCOCODataset(self.dir, split = "train", transform=self.train_transform, shared_transform=self.shared_train_transform, subset_indices=subset_indices_train, task = self.task, annotation_file = self.annotation_file, cluster_images = self.cluster_images, memory_bank = self.memory_bank)
        self.val_dataset = MSCOCODataset(self.dir, split = "val", transform = self.val_transform, shared_transform=self.shared_val_transform, subset_indices=subset_indices_val, task = self.task, annotation_file = self.annotation_file, cluster_images = None, memory_bank = False)
        # Test dataset setup can be added here if needed

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=False)
    
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
        elif self.task == "keypoint": #experimental
            return 255
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

    def __init__(self, batch_size, train_transform, val_transform, dir="/scratch-shared/nyu_hbird/nyu_data/", num_workers=0, cluster_images = None, memory_bank = False) -> None:
        self.batch_size = batch_size
        self.dir = dir
        self.train_transform = train_transform['train']
        self.val_transform = val_transform['val']
        self.num_workers = num_workers
        self.shared_train_transform = train_transform['shared']
        self.shared_val_transform = val_transform['shared']
        self.memory_bank = memory_bank
        self.cluster_images = cluster_images


    def setup(self):
        subset_index_train = None
        subset_index_val = None
        subset_index_train = 360
        # subset_index_val  = 20
        print("NYUv2 depth estimation task")
        print("Dataset loaded with memory bank mode: ", self.memory_bank)
        self.train_dataset = NYUv2(self.dir, split="train", transform=self.train_transform,shared_transform=self.shared_train_transform, subset_index=subset_index_train, cluster_images = self.cluster_images, memory_bank = self.memory_bank)
        self.val_dataset = NYUv2(self.dir, split="test", transform=self.val_transform,shared_transform=self.shared_val_transform,subset_index=subset_index_val, cluster_images = None, memory_bank = False)
        print(f"Train size : {len(self.train_dataset)}")
        print(f"Val size : {len(self.val_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,drop_last=False)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_module_name(self):
        return "NYUv2DataModule"
    
    def get_num_classes(self):
        #only use for segmentation...
        return 0
    
    
class VOCDataset(Dataset_custom):

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
    
class CocoTasksDataLoader(Dataset):
    """
    This is DataLoader for episodic training on COCO dataset
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: contains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.
    """

    def __init__(self, data_path, mode, batchsz, n_way, k_shot, k_query, resize, num_imgs_supp = 5, num_imgs_quer = 2, transforms = None):
        self.resize = resize
        self.device = device
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.num_imgs_supp = num_imgs_supp
        self.num_imgs_quer = num_imgs_quer
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.transform_train = transforms[0]
        self.transform_shared = transforms[1]
        print('DATA SETTINGS: %s, batchsz:%d, %d-way, %d-shot, %d-query, resize:%d, number images support:%d, number images query:%d' % (mode, batchsz, n_way, k_shot,
                                                                                          k_query, resize, num_imgs_supp, num_imgs_quer))
        print("We choose %d number of different support images and %d number of different images for query. We choose %d patches per image because we have %d-shot for support and %d patches per image because %d-shot for query" % (self.num_imgs_supp, self.num_imgs_quer, self.k_shot//self.num_imgs_supp, self.k_shot,self.k_query//self.num_imgs_quer, self.k_query))
        self.path = data_path  # image path
        self.mode = mode
   

        self.cls2img_data = self.load_cls2imgs("/home/lbusser/hbird_scripts/hbird_eval/data/class_to_image_ids.pkl")
        self.cls_num = 80
        self.focus_cls = []
        self.create_batch(self.batchsz)

    def load_cls2imgs(self, filename):
            """
            load image to class data from file
            """
            with open(filename, 'rb') as f:
                return pickle.load(f)
            
    def create_batch(self, batchsz):
        """
        create batch for meta-learning. Only collects the image ids with the randomly selected classes.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        # Creating of tasks; batchsz is the num. of iterations when sampling from the task distribution
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(range(1,self.cls_num+1), self.n_way, replace=False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select number of images for each class
                available_indices = list(range(len(self.cls2img_data[cls])))

                selected_imgs_idx = np.random.choice(available_indices, self.num_imgs_supp + self.num_imgs_quer, False)
                np.random.shuffle(selected_imgs_idx) #shuffle the ids
                indexDtrain = np.array(selected_imgs_idx[:self.num_imgs_supp])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.num_imgs_supp:])  # idx for Dtest
                support_x.append(np.array(self.cls2img_data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.cls2img_data[cls])[indexDtest].tolist()) #same for images filenames for Dtest
                available_indices = [idx for idx in available_indices if idx not in selected_imgs_idx]
                # if self.repeats > 0: 
                #     for _ in range(self.repeats):
                #         selected_repeat_idx = np.random.choice(available_indices, self.num_imgs_supp, False)
                #         np.random.shuffle(selected_repeat_idx)
                #         indexDtrain = np.array(selected_imgs_idx[:self.num_imgs_supp])
                #         support_x.append(np.array(self.cls2img_data[cls])[indexDtrain].tolist())
                #         available_indices = [idx for idx in available_indices if idx not in selected_repeat_idx]
            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)
            
            #########################################################################################
            # support_x and query_x now contains selected img_ids that contain the chosen class(es) #
            #########################################################################################
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append set to current sets
            self.focus_cls.append(selected_cls) # shape: (batchsz, n_way)


    def load_and_transform(self, img_name, mask_name, transform_img, transform_shared):
        """Load and transform an image and its mask."""
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        if transform_img:
            image = transform_img(image)
        if transform_shared:
            image, mask = transform_shared(image, mask)
        image = image.unsqueeze(0)
        mask = mask.squeeze(0)  # Assuming mask is initially [1, H, W]
        return image, mask

    def extract_patches_and_features(self, mask, current_focus_cls, num_select=1):
        """Extract valid patches and their features."""
        patches = mask.reshape(self.resize//14, 14, self.resize//14, 14).permute(0, 2, 1, 3).reshape(-1, 14, 14)
        valid_patches = np.any(np.isin(patches, current_focus_cls), axis=(1, 2))
        relevant_indices = np.where(valid_patches)[0].tolist()
        selected_indices = np.random.choice(relevant_indices, num_select, replace=False).tolist()
        # selected_patches = patches[selected_indices]
        # with torch.no_grad():
        #     features, _, _ = model.forward_features(image)
        # features = features.cpu().squeeze(0)
        # selected_features = features[selected_indices]
        # is_focus_cls = np.isin(selected_patches, current_focus_cls)
        # is_focus_cls = torch.from_numpy(is_focus_cls).to(device=selected_patches.device)
        # selected_patches[~is_focus_cls] = 0
        return selected_indices
    
    def update_tensors(self, image, mask,  tensor_x, tensor_y, selected_idxs, selected_idx, idx):
        """Update support_x/query_x and support_y/query_y tensors."""
        tensor_x[idx] = image
        tensor_y.append(mask)
        selected_idxs.append(selected_idx)
        idx+=1
        return idx
    
    def get_img_mask_names(self, img_id):
        """Generate image and mask file names."""
        img_name = f"{self.path}/{self.mode}2017/{int(img_id):012d}.jpg"
        mask_name = f"{self.path}/{self.mode}_masks2017/{int(img_id):012d}.png"
        return img_name, mask_name
    
    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_x = torch.FloatTensor(self.setsz//(self.k_shot//self.num_imgs_supp), 3, self.resize, self.resize)
        support_y = []
        query_x = torch.FloatTensor(self.querysz//(self.k_query//self.num_imgs_quer), 3, self.resize, self.resize)
        query_y = []
        current_focus_cls = np.array(self.focus_cls[index])
        selected_idxs_supp = []
        selected_idxs_quer = []
        # image path files  f"/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"
        flatten_support_x = [f"{self.path}/{self.mode}2017/{int(img_id):012d}.jpg"
                             for sublist in self.support_x_batch[index] for img_id in sublist]
        flatten_query_x = [f"{self.path}/{self.mode}2017/{int(img_id):012d}.jpg"
                           for sublist in self.query_x_batch[index] for img_id in sublist]
        idx = 0
        for sublist in self.support_x_batch[index]: #shape (n_way, num_imgs_supp)
            for img_id in sublist:
                img_name, mask_name = self.get_img_mask_names(img_id)
                image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
                selected_idx_supp = self.extract_patches_and_features(mask, current_focus_cls, self.k_shot//self.num_imgs_supp)
                idx = self.update_tensors(image, mask, support_x, support_y, selected_idxs_supp, selected_idx_supp, idx)
        idx=0
        for sublist in self.query_x_batch[index]: #shape (n_way, num_imgs_quer)
            for img_id in sublist:
                img_name, mask_name = self.get_img_mask_names(img_id)
                image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
                selected_idx_quer = self.extract_patches_and_features(mask, current_focus_cls, self.k_query//self.num_imgs_quer)
                idx = self.update_tensors(image, mask,query_x, query_y, selected_idxs_quer, selected_idx_quer, idx)
        return support_x, support_y, flatten_support_x, query_x, query_y, flatten_query_x, selected_idx_supp, selected_idx_quer

    def __len__(self):
        return self.batchsz
    
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
    shared_train_transform = Compose([
            RandomResizedCrop(size=(504, 504), scale=(min_scale_factor, max_scale_factor)),
            # RandomHorizontalFlip(probability=0.1),
        ])

    image_val_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])
    shared_transform = Compose([
        RandomResizedCrop(size=(448, 448), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])
    shared_val_transform = Compose([
            Resize(size=(504, 504)),
        ])

        # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"train": image_train_transform, "shared":shared_train_transform}
    val_transforms = {"val": image_val_transform , "shared": shared_val_transform}
    
    dataset = COCODataModule(batch_size=1, train_transform=train_transforms, val_transform=val_transforms, annotation_dir = "/scratch-shared/mscoco_hbird/annotations")
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    # test_dataloader = dataset.get_test_dataloader()
    # print(f"Train size : {len(dataset.train_dataset)}")
    # print(f"Val size : {len(dataset.val_dataset)}")
    # # print(f"Test size : {len(dataset.test_dataset)}")
    # print(f"Train dataloader size : {len(train_dataloader)}")
    # print(f"Val dataloader size : {len(val_dataloader)}")
    # print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y, name) in enumerate(train_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        print(torch.unique(y[0]))
        # y[y==165] = 0 #For stuff segmentation.
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
    input_size = 504
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

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([transforms.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([transforms.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([transforms.ColorJitter(hue=hue_jitter_probability)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])

    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])

    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])

    # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"train": image_train_transform, "shared": shared_train_transform}
    val_transforms = {"val": image_val_transform , "shared": shared_val_transform}
    dataset = NYUv2DataModule(batch_size=1, train_transform=train_transforms, val_transform=val_transforms)
    dataset.setup()
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    # print(f"Train size : {len(dataset.train_dataset)}")
    # print(f"Val size : {len(dataset.val_dataset)}")
    # print(f"Test size : {len(dataset.test_dataset)}")
    # print(f"Train dataloader size : {len(train_dataloader)}")
    # print(f"Val dataloader size : {len(val_dataloader)}")
    # print(f"Test dataloader size : {len(test_dataloader)}")
    for i, (x, y,_) in enumerate(train_dataloader):
        print(f"Train batch {i} : {x.shape}, {y.shape}")
        ## log image
        y.to(device)
        logger.log({"train_batch": [wandb.Image(x[0]), wandb.Image(y[0])]})
        if i ==10:
            break



    

if __name__ == "__main__":
    ## init wandb
    # logger = wandb.init(project=project_name, group="data_loader", tags="MSCOCODataModule", job_type="eval")
    ## test data module
    # test_pascal_data_module(logger)
    ## finish wandb
    # logger.finish()
    # test_nyuv2_data_module(logger)
    # NYUv2(root="/somepath/NYUv2", download=True, 
    #   rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t)
    MODEL = "dinov2_vitb14"
    device = 'cuda'
    input_size = 504
    eval_spatial_resolution = input_size // 14
    vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
    feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution = eval_spatial_resolution, d_model = 768)
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

    # get_file_path("/ssdstore/ssalehi/dataset/val1/JPEGImages/")
    # test_video_data_module(logger)
    few_shot_ds = CocoTasksDataLoader(data_path="/scratch-shared/mscoco_hbird", mode="train", batchsz=5, n_way=5, k_shot=10, k_query=2, resize=504, transforms = (image_train_transform, shared_train_transform))
    few_shot_ds[0]
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
