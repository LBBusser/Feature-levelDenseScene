import torch
# from models import FeatureExtractor
from models import FeatureExtractorBeta as FeatureExtractor
from data_loader import PascalVOCDataModule, COCODataModule
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


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Hummingbird Clustering", add_help=add_help)
    parser.add_argument("--patch_size", default=14, type=int, help="Patch size", choices=[14, 16])
    parser.add_argument("--arch", default='vitb', type=str, help="Architecture", choices=['vitg', 'vitl', 'vitb', 'vits'])
    parser.add_argument("--model_type", default='dinov2',choices = ['dino', 'dinov2'], type=str, help="Model type")
    parser.add_argument("--device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--model-device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--num-clusters", default=1000, type=int, help="Number of clusters", required=False)
    return parser

args = get_args_parser(add_help=True).parse_args()
DEVICE = args.device
MODEL_TYPE = args.model_type
CLUSTERS = args.num_clusters
patch_size = args.patch_size
arch = args.arch
MODEL_DEVICE = args.model_device
MODEL = f'{MODEL_TYPE}_{arch}{patch_size}'
DIR = '/home/lbusser/hbird_scripts/hbird_eval/data'
EXP_NAME = f'{MODEL}_cluster_results'

EXP_DIR = os.path.join(DIR, EXP_NAME)
os.makedirs(EXP_DIR, exist_ok = True) 

class HummingbirdClustering():
    def __init__(self, feature_extractor, dataset_module, num_clusters, device, num_components = 128):
        self.feature_extractor = feature_extractor
        self.dataset_module = dataset_module
        self.num_components = num_components
        self.device = device
        self.num_clusters = num_clusters
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.d_model = self.feature_extractor.d_model
        self.num_components = num_components
        self.kmeans = faiss.Kmeans(self.num_components, self.num_clusters, niter=20, verbose=True)
        self.cluster_assignments = {} #Key: image name, Value: cluster id
        self.online_cluster()
      

    def online_cluster(self, interval = 50):
        train_loader = self.dataset_module.get_train_dataloader()
        all_features_accumulated = []
        all_image_paths = []
        batch_counter = 0
        with torch.no_grad():
                for i, (x, y, img_names) in enumerate(tqdm(train_loader)):
                    print(f"batch {i} has been read at {time.ctime()}")
                    bs = x.shape[0]
                    all_image_paths.extend(img_names)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y = y.long()
                    features, _ = self.feature_extractor.forward_features(x) #features shape is [bs, num_patches_flattened, d_model]
                    flattened_features= features.reshape(bs,-1).detach().cpu()
                    flattened_features = flattened_features.numpy()
                    pca = PCA(n_components=self.num_components)
                    reduced_features = pca.fit_transform(flattened_features)
                    all_features_accumulated.extend(reduced_features)
                    batch_counter += 1
                    if batch_counter % interval == 0 or i == len(train_loader):
                        print(f"starting interval clustering at {time.ctime()} for batches {i-interval} to {i}")
                        if batch_counter == interval:
                            self.kmeans = self.initial_clustering(all_features_accumulated)
                        else:
                            self.kemans = self.incremental_clustering(self.kmeans, all_features_accumulated)

                        _, new_assignments = self.kmeans.index.search(all_features_accumulated, 1)
                        self.cluster_assignments.update(zip(all_image_paths, new_assignments.flatten()))
                        self.save_cluster_assignments_incrementally(self.cluster_assignments, os.path.join(EXP_DIR, 'cluster_assignments.pkl'))
                        all_features_accumulated = []
                    print(self.cluster_assignments)
                    print("Length of cluster_assignments:",len(self.cluster_assignments))
                    print(f"batch {i} has been processed at {time.ctime()}")
        


    def initial_clustering(self, features):
        print("initial clustering has started")
        self.kmeans.train(np.array(features).astype('float16'))
        return self.kmeans
                    

    def incremental_clustering(self, kmeans, features):
        num_clusters_current = kmeans.k
        print('Incremental clustering with {} clusters'.format(num_clusters_current))
            # Combine existing centroids with new data
        combined_data = np.vstack([kmeans.centroids, np.array(features).astype('float16')])
        
        # Re-cluster with the new number of clusters
        new_kmeans = faiss.Kmeans(self.num_components, num_clusters_current, niter=50, verbose=True)
        new_kmeans.train(combined_data)
        
        # Update the kmeans object and centroids for the next iteration
        self.kmeans = new_kmeans

        return self.kmeans
    
    def save_cluster_assignments_incrementally(self, cluster_assignments, filename):
        # This function saves the cluster assignments to a pickle file incrementally
        print('Saving cluster assignments to {}'.format(filename))
        with open(filename, 'wb') as file:
            pickle.dump(cluster_assignments, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cluster_assignments(filename='/home/lbusser/hbird_scripts/hbird_eval/data/dinov2_vitb14_cluster_results/cluster_assignments.pkl'):
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
    used_dataset = "MSCOCO"
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

    image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])
    # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
    train_transforms = {"train": image_train_transform, "target": None, "shared": shared_train_transform}
    val_transforms = {"val": image_val_transform, "target": None , "shared": shared_val_transform}
    if used_dataset == "MSCOCO":
            dataset = COCODataModule(batch_size=128, train_transform=train_transforms, val_transform=val_transforms, test_transform=val_transforms, task=task, annotation_dir="/scratch-shared/mscoco_hbird/annotations")
    else:
        dataset = PascalVOCDataModule(batch_size=64, train_transform=train_transforms, val_transform=val_transforms, test_transform=val_transforms)
    dataset.setup()
    evaluator = HummingbirdClustering(feature_extractor, dataset, num_clusters=CLUSTERS, device=device)
    