import torch
# from models import FeatureExtractor
from models import FeatureExtractorBeta as FeatureExtractor
from data_loader import PascalVOCDataModule, COCODataModule, NYUv2DataModule
import torchvision.transforms as trn
import torch.nn.functional as F
import math
import time
from sklearn.decomposition import PCA
import os
import faiss
from eval_metrics import PredsmIoU
import numpy as np
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224
import timm
from tqdm import tqdm
import argparse
import csv
import pickle
import shutil


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Hummingbird Clustering", add_help=add_help)
    parser.add_argument("--patch_size", default=14, type=int, help="Patch size", choices=[14, 16])
    parser.add_argument("--arch", default='vitb', type=str, help="Architecture", choices=['vitg', 'vitl', 'vitb', 'vits'])
    parser.add_argument("--model_type", default='dinov2',choices = ['dino', 'dinov2'], type=str, help="Model type")
    parser.add_argument("--device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--model-device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--num-clusters", default=1000, type=int, help="Number of clusters", required=False)
    parser.add_argument("--dataset", default='MSCOCO', type=str, help="Dataset to cluster", required=False)
    return parser

args = get_args_parser(add_help=True).parse_args()
DEVICE = args.device
MODEL_TYPE = args.model_type
CLUSTERS = args.num_clusters
patch_size = args.patch_size
arch = args.arch
DATASET = args.dataset
MODEL_DEVICE = args.model_device
MODEL = f'{MODEL_TYPE}_{arch}{patch_size}'
DIR = '/home/lbusser/hbird_scripts/hbird_eval/data'
EXP_NAME = f'{DATASET}_{MODEL}_{CLUSTERS}_cluster_results'

EXP_DIR = os.path.join(DIR, EXP_NAME)
os.makedirs(EXP_DIR, exist_ok = True) 

class HummingbirdClustering():
    def __init__(self, feature_extractor, dataset_module, num_clusters, device):
        
        self.feature_extractor = feature_extractor
        self.dataset_module = dataset_module
       
        self.device = device
        self.num_clusters = num_clusters
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.d_model = self.feature_extractor.d_model
       
        self.cluster_assignments = {} #Key: image name, Value: cluster id
        self.all_image_paths = []
        self.train_loader = self.dataset_module.get_train_dataloader()
        print("CLUSTERING WITH CLUSTERS:", self.num_clusters)
        # self.accumulate_features()


        #####IF FEATURES ALREADY COLLECTED######  
        all_features_accumulated = self.load_features()
        self.all_image_paths, all_features_list = zip(*all_features_accumulated.items())
        _, I = self.clustering(np.array(all_features_list))
        cluster_assignments = {self.all_image_paths[i]: int(I[i][0]) for i in range(len(self.all_image_paths))}
        self.save_cluster_assignments(cluster_assignments, os.path.join(EXP_DIR, 'cluster_assignments.pkl'))

    def accumulate_features(self):
        all_features_accumulated = []
        with torch.no_grad():
                for i, (x, _, img_names) in enumerate(tqdm(self.train_loader)):
                    # print(f"batch {i} has been read at {time.ctime()}")
                    self.all_image_paths.extend(img_names)
                    x = x.to(self.device)
                    _, cls_features,_ = self.feature_extractor.forward_features(x) #features shape is [bs, num_patches_flattened, d_model] for cls is [bs, d_model]
                    cls_features = cls_features.detach().cpu()
                    all_features_accumulated.extend(cls_features)
                    # print(f"batch {i} has been processed at {time.ctime()}")
                    
        
        features_dict = dict(zip(self.all_image_paths, all_features_accumulated))
        self.save_features(features_dict, os.path.join(EXP_DIR, 'features_dict.pkl'))
        print("Length of all the processed features:",len(all_features_accumulated))
        _, I = self.clustering(np.array(all_features_accumulated))
        cluster_assignments = {self.all_image_paths[i]: int(I[i][0]) for i in range(len(self.all_image_paths))}
        self.save_cluster_assignments(cluster_assignments, os.path.join(EXP_DIR, 'cluster_assignments.pkl'))
        
    def clustering(self, features):
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        cluster = faiss.Clustering(d, self.num_clusters)
        cluster.verbose = True
        cluster.niter = 20
        print("clustering has started")
        cluster.train(features, index)
        _, I = index.search(features, 1) 
        faiss.write_index(index, os.path.join(EXP_DIR, 'cluster_index.index'))
        return cluster, I

    def save_features(self, features, filename):
        # This function saves the cluster assignments to a pickle file
        print('Saving features to {}'.format(filename))
        with open(filename, 'wb') as file:
            pickle.dump(features, file, protocol=pickle.HIGHEST_PROTOCOL)  
    
    def load_features(self, filename=f'/home/lbusser/hbird_scripts/hbird_eval/data/{DATASET}_dinov2_vitb14_1000_cluster_results/features_dict.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def save_cluster_assignments(self, cluster_assignments, filename):
        # This function saves the cluster assignments to a pickle file
        print('Saving cluster assignments to {}'.format(filename))
        with open(filename, 'wb') as file:
            pickle.dump(cluster_assignments, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cluster_assignments(filename='/home/lbusser/hbird_scripts/hbird_eval/data/dinov2_vitb14_1000_cluster_results/cluster_assignments.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        

if __name__ == "__main__":
    
    device = DEVICE
    # vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    if 'dinov2' in MODEL.lower():
        input_size = 504
        eval_spatial_resolution = input_size // 14
        vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
    elif 'dino' in MODEL.lower():
        input_size = 512
        eval_spatial_resolution = input_size // 16
        vit_model = torch.hub.load('facebookresearch/dino:main', MODEL)
    
    # vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    # vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        
    ########################
    task = "normal"
    ########################
    if arch == 'vitg':
        feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=eval_spatial_resolution, d_model=1536)
    elif arch == 'vitl':
        feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=eval_spatial_resolution, d_model=1024)
    elif arch == 'vitb':
        feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=eval_spatial_resolution, d_model=768)
    elif arch == 'vits':
        feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=eval_spatial_resolution, d_model=384)

    # Define transformation parameters
    min_scale_factor = 0.5
    max_scale_factor = 2.0
 

    # Create the transformation
    image_train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])

    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])

    image_val_transform = trn.Compose([ 
        trn.ToTensor(), 
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])

    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])
    
    # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"train": image_train_transform, "target": None, "shared": shared_train_transform}
    val_transforms = {"val": image_val_transform, "target": None , "shared": shared_val_transform}
    if DATASET == "MSCOCO":
            dataset = COCODataModule(batch_size=64, train_transform=train_transforms, val_transform=val_transforms, task=task, annotation_dir="/scratch-shared/mscoco_hbird/annotations")
    elif DATASET == "NYUv2":
            dataset = NYUv2DataModule(batch_size=64, train_transform=train_transforms, val_transform=val_transforms)
    else:
        raise ValueError("Invalid dataset name")
    dataset.setup()
    evaluator = HummingbirdClustering(feature_extractor, dataset, num_clusters=CLUSTERS, device=device)
    
    
    
    ###########
    ## TEST  ##
    ###########
    # Load the cluster assignments from the .pkl file
#     with open(f'/home/lbusser/hbird_scripts/hbird_eval/data/{DATASET}_dinov2_vitb14_{CLUSTERS}_cluster_results/cluster_assignments.pkl', 'rb') as f:
#         cluster_assignments = pickle.load(f)
#     val_loader = dataset.get_val_dataloader(batch_size=1)
#     loaded_index = faiss.read_index(f'/home/lbusser/hbird_scripts/hbird_eval/data/{DATASET}_dinov2_vitb14_{CLUSTERS}_cluster_results/cluster_index.index')
#     for i, (x, y, img_names) in enumerate(val_loader):
#         print(f"batch {i} has been read at {time.ctime()}")
#         bs = x.shape[0]
#         x = x.to(device)
#         y = y.to(device)
#         y = y.long()
#         features, _ = feature_extractor.forward_features(x)
#         flattened_features= features.reshape(bs,-1).detach().cpu()
#         flattened_features = flattened_features.numpy()
#         reduced_features = np.mean(flattened_features, axis=1)
#         _,I = loaded_index.search(reduced_features.reshape(-1,1), 1)
#         test_image_cluster_id = I[0][0]
#         corresponding_images = [image_path for image_path, assigned_cluster_id in cluster_assignments.items() if assigned_cluster_id == test_image_cluster_id]
#         print(corresponding_images)
#         print(img_names)
#         destination_dir = os.getcwd()

# # Copy each image to the destination directory
#         for image_path in corresponding_images:
#             # Extract the base filename to maintain the original file name
#             filename = os.path.basename(image_path)
#             destination_path = os.path.join(destination_dir, filename)
            
#             # Copy the file to the new location
#             shutil.copy(image_path, destination_path)
#         break