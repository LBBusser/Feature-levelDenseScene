from collections import OrderedDict
import torch
import sys
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
import random
import json
import faiss
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

    def __init__(self, batch_size, train_transform, val_transform, dir="mscoco_hbird/", num_workers=0, task = "normal", annotation_dir = None, cluster_images = None, memory_bank = False) -> None:
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

    def __init__(self, batch_size, train_transform, val_transform, dir="nyu_data/", num_workers=0, cluster_images = None, memory_bank = False) -> None:
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
        # subset_index_train = 360
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

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir="VOCdevkit/VOC2012/", num_workers=0) -> None:
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
     

class NYUMemoryTasksDataLoader(Dataset):
    """
    This is DataLoader for episodic training on NYUv2 depth dataset using the memory bank approach
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: contains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.
    """

    def __init__(self, data_path, mode, setsz, k_shot, k_query, resize, cluster_index, cluster_assignment, transforms = None, mode_idx = None):
        self.resize = resize
        self.batchsz = setsz  # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.cluster_index = cluster_index
        self.transform_train = transforms[0]
        self.transform_shared = transforms[1]
        print('DATA SETTINGS: %s, task number:%d, %d-shot, %d-query, resize:%d' % (mode, setsz, k_shot,
                                                                                          k_query, resize))
        self.path = data_path  # image path
        # self.mode_idx = mode_idx
        self.mode = mode
        self.mode_idx = mode_idx
        self.images_dir = os.path.join(self.path, "nyu2_"+ self.mode)
        self.csv_file = os.path.join(self.path, f"nyu2_{self.mode}.csv")
        self.dataframe = pd.read_csv(self.csv_file)
        if mode_idx is not None:
            self.image_paths = self.dataframe.iloc[mode_idx,0].values.tolist()
        else:
            self.image_paths = self.dataframe.iloc[:,0].values.tolist()
        self.cluster_assignment = self.load_cluster_assignment(cluster_assignment)
        self.create_batch(self.batchsz)
        
    def load_cluster_assignment(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def normalize_depth_map(self, depth_map):
        if self.mode == "train":
            self.global_min = 18  
            self.global_max = 255  
        elif self.mode == "test":
            self.global_min = 713 
            self.global_max = 9986
        depth_array = np.array(depth_map)
        depth_array = depth_array.astype(np.float32)
        depth_array = (depth_array - self.global_min) / (self.global_max - self.global_min)
        return depth_array

    def save_tensor_as_image(self, tensor, file_path, mask=False):
        """
        Save a tensor as an image file.
        :param tensor: Tensor to save.
        :param file_path: Path where the image will be saved.
        """
        # Normalize the tensor to the range [0, 1]
           # Normalize the tensor to the range [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Convert tensor to numpy array with shape (H, W, C)
        image = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8 type and save image
        image = (image * 255).astype('uint8')
        Image.fromarray(image).save(file_path)
        print("Saved to", file_path)
    
    def create_batch(self, setsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        print("Initialising NYU batch creation for few shot learning...")
        
        for b in tqdm(range(setsz)):
            support_x = []
            query_x = []
            
            current_query = np.random.choice(self.image_paths, self.k_query, False) #randomly select k_query images from train set
            for _ in range(len(current_query)):  # We iterate over the number of selected queries
                while True:  # Start an indefinite loop that we'll break out of once conditions are met
                    selected = current_query[_]  # Get the current query image
                    selected_id = "/".join(selected.split("/")[-2:])
                    self.image_paths.remove(selected)
                    if self.mode == 'train':
                        full_selected = os.path.join(self.path,"nyu2_"+self.mode,selected_id)
                    elif self.mode =='test':
                        full_selected = os.path.join(self.path,selected_id)
                
                    cluster_id = self.cluster_assignment.get(full_selected) # Get the cluster assignment of the image
                  
                    selected_imgs = [image_id for image_id, cid in self.cluster_assignment.items() if cid == cluster_id]  # Get the other images in the same cluster
                      # Remove the image itself to prevent duplicates in support and query
                    
                    selected_imgs.remove(full_selected)
          
                    np.random.shuffle(selected_imgs)
                    if len(selected_imgs) >= self.k_shot:
                        selected_imgs = np.random.choice(selected_imgs, self.k_shot, False)  # Ensure even distribution
                        break  # Conditions are met, break out of the while loop
                    else:
                        print("Selecting new query, because support set too small!")
                        # Ensure that the new selection does not repeat previously selected images
                        new_query = np.random.choice(self.image_paths, 1, False)[0]
                        current_query[_] = new_query
                        continue  # Restart the while loop with the new selection
                support_x.extend(selected_imgs)
            query_x.append(current_query)
            
            np.random.shuffle(support_x)
            np.random.shuffle(query_x)
            ##########################################################################################
            # support_x and query_x now contains selected img_ids that are similar to query image(s) #
            ##########################################################################################
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.extend(query_x)  # append set to current sets
    

    def load_and_transform(self, img_name, mask_name, transform_img, transform_shared):
        """Load and transform an image and its mask."""
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        if transform_img:
            image = transform_img(image)
        if transform_shared:
            image, mask = transform_shared(image, mask)
        mask = self.normalize_depth_map(mask)
        image = image.unsqueeze(0)
        mask = mask  # Assuming mask is initially [1, H, W]
        return image, mask
    
    def update_tensors(self, image, mask,  tensor_x, tensor_y, selected_idxs, selected_idx, idx):
        """Update support_x/query_x and support_y/query_y tensors."""
        tensor_x[idx] = image
        tensor_y.append(mask)
        selected_idxs.append(selected_idx)
        idx+=1
        return idx
    
    def get_img_mask_names(self, img_path):
        """Generate image and mask file names."""
        if self.mode =='train':
            img_name = img_path
            mask_name = img_path[:-4] + '.png'
        elif self.mode == 'test':
            img_name = img_path
            mask_name = img_path.replace('colors', 'depth')
        return img_name, mask_name
    
    def __getitem__(self, index):
        """
        index means index of the batch, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_x = torch.FloatTensor(self.k_shot*self.k_query, 3, self.resize, self.resize)
        support_y = torch.zeros((self.k_shot*self.k_query, 3, self.resize,self.resize),dtype = torch.float)
        query_x = torch.FloatTensor(self.k_query, 3, self.resize, self.resize)
        query_y = torch.zeros((self.k_query, 3, self.resize,self.resize),dtype = torch.float)
        save_dir = os.path.join("support_images", self.mode, f"batch_{index}")
        os.makedirs(save_dir, exist_ok=True)
        # image path files  f"/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"

        for i,img_path in enumerate(self.support_x_batch[index]): #shape (k_shot)
            img_name, mask_name = self.get_img_mask_names(img_path)
            image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
            # patches = mask.reshape(self.resize//14, 14, self.resize//14, 14).permute(0, 2, 1, 3).reshape(-1, 14, 14)
            support_y[i]= torch.from_numpy(mask).repeat(3,1,1)
            support_x[i] = image

            # self.save_tensor_as_image(image, os.path.join(save_dir, f"support_image_{i}.jpg"))
            # self.save_tensor_as_image(mask, os.path.join(save_dir, f"support_mask_{i}.jpg"), mask=True)


        for i, img_id in enumerate(self.query_x_batch[index]): #shape (k_query)
            img_name, mask_name = self.get_img_mask_names(img_path)
            image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
            # patches = mask.reshape(self.resize//14, 14, self.resize//14, 14).permute(0, 2, 1, 3).reshape(-1, 14, 14)
            query_y[i]= torch.from_numpy(mask).repeat(3,1,1)
            query_x[i] = image
        return support_x, support_y, query_x, query_y

    def __len__(self):
        return self.batchsz

class CocoMemoryTasksDataLoader(Dataset):
    """
    This is DataLoader for episodic training on COCO dataset using the memory bank approach
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: contains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.
    """

    def __init__(self, data_path, mode, setsz, k_shot, k_query, resize, cluster_index, cluster_assignment, transforms = None, mode_idx= None, panoptic = False):
        self.resize = resize
        self.batchsz = setsz  # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.cluster_index = cluster_index
        self.panoptic = panoptic
        self.mode_idx = mode_idx
        if transforms is not None:
            self.transform_train = transforms[0]
            self.transform_shared = transforms[1]
        print('DATA SETTINGS: %s, task number:%d, %d-shot, %d-query, resize:%d' % (mode, setsz, k_shot,
                                                                                          k_query, resize))
        self.path = data_path  # image path
        self.mode = mode
        self.images_dir = os.path.join(self.path, self.mode + '2017')
        self.masks_dir = os.path.join(self.path, self.mode + '_masks2017')
        if self.panoptic:
            self.masks_dir = '/home/lbusser/annotations/panoptic/panoptic_val2017'
        if self.mode_idx is not None:
            self.images = sorted(os.listdir(self.images_dir))
            self.images = [self.images[idx] for idx in self.mode_idx]
            self.masks = sorted(os.listdir(self.masks_dir))
            self.masks = [self.masks[idx] for idx in self.mode_idx]
        else:
            self.images = sorted(os.listdir(self.images_dir))
            self.masks = sorted(os.listdir(self.masks_dir))
    
        self.cluster_assignment = self.load_cluster_assignment(cluster_assignment)
        self.create_batch(self.batchsz)
    
    def load_cluster_assignment(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
          
    def create_batch(self, setsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        print("Initialising COCO batch creation for few shot meta learning...")
        for b in tqdm(range(setsz)):
            support_x = []
            query_x = []
            current_query = np.random.choice(self.images, self.k_query, False) #randomly select k_query images from train set
            current_query = [self.images_dir +"/"+ img_id for img_id in current_query]
            for _ in range(len(current_query)):  # We iterate over the number of selected queries
                while True:  # Start an indefinite loop that we'll break out of once conditions are met
                    selected = current_query[_]  # Get the current query image
                    selected_id = selected.split("/")[-1]
                   
                    self.images.remove(selected_id)
                    cluster_id = self.cluster_assignment.get(selected)  # Get the cluster assignment of the image
                    selected_imgs = [image_id for image_id, cid in self.cluster_assignment.items() if cid == cluster_id]  # Get the other images in the same cluster
                    # Remove the image itself to prevent duplicates in support and query
                    selected_imgs.remove(selected)
                    np.random.shuffle(selected_imgs)
                    if len(selected_imgs) >= self.k_shot:
                        selected_imgs = np.random.choice(selected_imgs, self.k_shot, False) 
                        break  # Conditions are met, break out of the while loop
                    else:
                        print("Selecting new query, because support set too small!")
                        # Ensure that the new selection does not repeat previously selected images
                        new_query = np.random.choice([img for img in self.images if img not in current_query], 1)[0]
                        current_query[_] = self.images_dir + '/' + new_query
                        continue  # Restart the while loop with the new selection
                support_x.extend(selected_imgs)
            query_x.append(current_query)
            
            np.random.shuffle(support_x)
            np.random.shuffle(query_x)
            ##########################################################################################
            # support_x and query_x now contains selected img_ids that are similar to query image(s) #
            ##########################################################################################
                
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.extend(query_x)  # append set to current sets

    
    def load_and_transform(self, img_name, mask_name, transform_img, transform_shared):
        """Load and transform an image and its mask."""
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        if transform_img:
            image = transform_img(image)
        if transform_shared:
            image, mask = transform_shared(image, mask)
        image = image.unsqueeze(0)
        mask = mask/255
        return image, mask
    
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
        if self.panoptic:
            mask_name = f"annotations/panoptic/panoptic_val2017/{int(img_id):012d}.png"
        return img_name, mask_name



    def save_tensor_as_image(self, tensor, file_path):
        """
        Save a tensor as an image file.
        :param tensor: Tensor to save.
        :param file_path: Path where the image will be saved.
        """
        # Normalize the tensor to the range [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Convert tensor to numpy array with shape (H, W, C)
        image = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8 type and save image
        image = (image * 255).astype('uint8')
        Image.fromarray(image).save(file_path)
        print("Saved to", file_path)


    def __getitem__(self, index):
        """
        index means index of the batch, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_x = torch.FloatTensor(self.k_shot*self.k_query, 3, self.resize, self.resize)
        support_y = torch.zeros((self.k_shot*self.k_query, 3, self.resize,self.resize),dtype = torch.float)
        query_x = torch.FloatTensor(self.k_query, 3, self.resize, self.resize)
        query_y = torch.zeros((self.k_query, 3, self.resize,self.resize),dtype = torch.float)
        save_dir = os.path.join("support_images", self.mode, f"batch_{index}")
        os.makedirs(save_dir, exist_ok=True)
        # image path files  f"/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"

        for i,img_id in enumerate(self.support_x_batch[index]): #shape (k_shot)
            img_id = img_id.split("/")[-1] 
            img_id = os.path.splitext(img_id)[0]
            img_name, mask_name = self.get_img_mask_names(img_id)
            image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
            # patches = mask.reshape(self.resize//14, 14, self.resize//14, 14).permute(0, 2, 1, 3).reshape(-1, 14, 14)
            support_y[i]= mask
            support_x[i] = image

        # Save images and masks
            self.save_tensor_as_image(image, os.path.join(save_dir, f"support_image_{i}.jpg"))
            self.save_tensor_as_image(mask, os.path.join(save_dir, f"support_mask_{i}.jpg"))

        for i, img_id in enumerate(self.query_x_batch[index]): #shape (k_query)
            img_id = img_id.split("/")[-1]
            img_id = os.path.splitext(img_id)[0]
            img_name, mask_name = self.get_img_mask_names(img_id)
            image, mask = self.load_and_transform(img_name, mask_name, self.transform_train, self.transform_shared)
            # patches = mask.reshape(self.resize//14, 14, self.resize//14, 14).permute(0, 2, 1, 3).reshape(-1, 14, 14)
            query_y[i]= mask
            query_x[i] = image

        return support_x, support_y, query_x, query_y

    def __len__(self):
        return self.batchsz

class CombinedDataset(Dataset):
    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.ratios = ratios
        self.total_ratio = sum(ratios)
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_ratios = [sum(ratios[:i+1]) for i in range(len(ratios))]

        # Calculate the total size based on ratios
        self.dataset_size = sum([size * ratio for size, ratio in zip(self.dataset_sizes, self.ratios)])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Normalize idx to range within total_ratio
        normalized_idx = idx % self.total_ratio

        # Determine which dataset to sample from
        for i, cumulative_ratio in enumerate(self.cumulative_ratios):
            if normalized_idx < cumulative_ratio:
                dataset_idx = i
                break

        # Calculate index within the chosen dataset
        dataset_ratio = self.ratios[dataset_idx]
        effective_size = self.dataset_sizes[dataset_idx] * dataset_ratio
        data_idx = (idx // self.total_ratio) % effective_size

        return self.datasets[dataset_idx][data_idx]

    
