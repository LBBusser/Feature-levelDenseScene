import torch
# from models import FeatureExtractor
from models import FeatureExtractorBeta as FeatureExtractor
from data_loader import PascalVOCDataModule, COCODataModule, NYUv2DataModule
import torchvision.transforms as trn
import torch.nn.functional as F
import shutil
from matplotlib import pyplot as plt
import math
import scann
import time
import faiss
import os
from eval_metrics import PredsmIoU
import numpy as np
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224
import timm
from tqdm import tqdm
import argparse
import pickle
import random

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Hummingbird Evaluation", add_help=add_help)
    parser.add_argument("--mem_size", default=51200, type=int, help="Size of the memory")
    parser.add_argument("--neighbors", default=90, type=int, help="Number of neighbours")
    parser.add_argument("--patch_size", default=14, type=int, help="Patch size", choices=[14, 16])
    parser.add_argument("--arch", default='vitb', type=str, help="Architecture", choices=['vitg', 'vitl', 'vitb', 'vits'])
    parser.add_argument("--aug_epochs", default=1, type=int, help="Augmentation epochs")
    parser.add_argument("--model_type", default='dinov2',choices = ['dino', 'dinov2'], type=str, help="Model type")
    parser.add_argument("--device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--model-device", default='cuda', type=str, help="Device", required=False)
    parser.add_argument("--force-save", action='store_true', help="Force Saving memory again", required=False)

    parser.add_argument("--num_leaves", default=512, type=int, help="Number of leaves for ScaNN")
    parser.add_argument("--num_leaves_to_search", default=256, type=int, help="Number of leaves to search in ScaNN")
    parser.add_argument("--reorder", default=1800, type=int, help="Reorder for ScaNN")
    parser.add_argument("--evaluation_only", action='store_true', help="Evaluation only", required=False)
    parser.add_argument("--clusters", default=1000, type=int, help="Number of clusters")
    parser.add_argument("--dataset", default='MSCOCO', type=str, help="Dataset", choices=['MSCOCO', 'PascalVOC','NYUv2'])
    return parser

args = get_args_parser(add_help=True).parse_args()
DEVICE = args.device
MODEL_TYPE = args.model_type
MEM_SIZE = args.mem_size
NEIGHBOURS = args.neighbors
AUG_EPOCHS = args.aug_epochs
patch_size = args.patch_size
arch = args.arch
MODEL_DEVICE = args.model_device
CLUSTERS = args.clusters
DATASET = args.dataset
# ScaNN parameters
NUM_LEAVES = args.num_leaves
NUM_LEAVES_TO_SEARCH = args.num_leaves_to_search
REORDER = args.reorder
MODEL = f'{MODEL_TYPE}_{arch}{patch_size}'
FORCE_SAVE = args.force_save
eval_only = args.evaluation_only
DIR = '/home/lbusser/hbird_scripts/hbird_eval/data'
EXP_NAME = f'{MODEL}_{MEM_SIZE}_{NEIGHBOURS}_{AUG_EPOCHS}_{CLUSTERS}'
EXP_DIR = os.path.join(DIR, EXP_NAME)
os.makedirs(EXP_DIR, exist_ok = True) 
F_MEM = os.path.join(EXP_DIR, 'fts_memory_cluster.pt')
L_MEM = os.path.join(EXP_DIR, 'lbl_memory_cluster.pt')
cluster_index = f"/home/lbusser/hbird_scripts/hbird_eval/data/{DATASET}_dinov2_vitb14_{CLUSTERS}_cluster_results/cluster_index.index"



class HummingbirdEvaluation():
    def __init__(self, feature_extractor, num_neighbour, augmentation_epoch, memory_size, task, device, cluster_index, evaluation_only = False):
        self.feature_extractor = feature_extractor
        self.device = device
        self.task = task
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.num_neighbour = num_neighbour
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        
        self.cluster_index = cluster_index
        self.cluster_index = faiss.read_index(cluster_index)
        self.cluster_assignments = self.load_cluster_assignments()

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
        self.image_train_transform = trn.Compose([
            trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
            trn.RandomApply([transforms.ColorJitter(hue=hue_jitter_probability)], p=hue_jitter_probability),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])

        self.shared_train_transform = Compose([
            RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
            # RandomHorizontalFlip(probability=0.1),
        ])

        self.image_val_transform = trn.Compose([ trn.ToTensor(), trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
        self.shared_val_transform = Compose([
            Resize(size=(input_size, input_size)),
        ])

        # target_train_transform = trn.Compose([trn.Resize((224, 224), interpolation=trn.InterpolationMode.NEAREST), trn.ToTensor()])
        self.train_transforms = {"train": self.image_train_transform, "shared": self.shared_train_transform}
        self.val_transforms = {"val": self.image_val_transform , "shared": self.shared_val_transform}
        if DATASET == "MSCOCO":
                dataset = COCODataModule(batch_size=64, train_transform=self.train_transforms, val_transform=self.val_transforms, task=task, annotation_dir="/scratch-shared/mscoco_hbird/annotations")
        elif DATASET == "NYUv2":
                dataset = NYUv2DataModule(batch_size=64, train_transform=self.train_transforms, val_transform=self.val_transforms)
        dataset.setup()
        self.dataset_module = dataset
        self.incontext_evaluation()

    def load_cluster_assignments(self, filename=f'/home/lbusser/hbird_scripts/hbird_eval/data/{DATASET}_dinov2_vitb14_{CLUSTERS}_cluster_results/cluster_assignments.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    
    def create_NN(self):
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), self.num_neighbour, "dot_product").tree(
    num_leaves=NUM_LEAVES, num_leaves_to_search=NUM_LEAVES_TO_SEARCH, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=0.1).reorder(REORDER).build()

    def create_memory(self, relevant_images):
        if DATASET == "MSCOCO":
            data = COCODataModule(batch_size=64, train_transform=self.train_transforms, val_transform=self.val_transforms, task=task, annotation_dir="/scratch-shared/mscoco_hbird/annotations", cluster_images=relevant_images, memory_bank=True)
        elif DATASET =="NYUv2":
            data = NYUv2DataModule(batch_size=64, train_transform=self.train_transforms, val_transform=self.val_transforms, cluster_images=relevant_images, memory_bank=True)
        data.setup()
        data_loader = data.get_train_dataloader()
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.num_sampled_features = self.memory_size // (data.get_train_dataset_size() * self.augmentation_epoch)
        assert self.num_sampled_features <= 1296, "num_sampled_features exceeds the maximum allowed value of 1296. Please adjust the memory_size, train dataset size to correct this."
        self.feature_memory = torch.zeros((self.memory_size, self.feature_extractor.d_model))
        self.label_memory = torch.zeros((self.memory_size, self.dataset_module.get_num_classes()+1))
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        idx = 0
        with torch.no_grad():
            for j in range(self.augmentation_epoch):
                print(f"augmentation epoch {j} has started at {time.ctime()}")
                for i, (x, y,_) in enumerate(tqdm(data_loader)):
                    print(f"batch {i} has been read at {time.ctime()}")
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if DATASET == "MSCOCO":
                        y = y.long()
                    ###################
                    features, _,_ = self.feature_extractor.forward_features(x)  
                    features = features.to(self.device) # bs, num_patches_flatteded (36x36), d_model
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    patchified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    ###################
                    #FOR MSCOCO KEYPOINT or DEPTH
                    if DATASET == "NYUv2":
                        patchified_gts = patchified_gts.float() 
                        label = patchified_gts.mean(dim=3)
                        sampled_features, sampled_indices = self.sample_features_depth(features, patchified_gts)
                    ###################
                    #FOR MSCOCO SEGMENTATION
                    if DATASET == "MSCOCO":
                        num_classes = self.dataset_module.get_num_classes() + 1
                        one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()  ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size, num_classes)
                        label = one_hot_patch_gt.mean(dim=3) #Change the way labels are stored, maybe put raw labels in, no aggregate!
                        sampled_features, sampled_indices = self.sample_features(features, patchified_gts)  
                    ###################
                    normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=1, keepdim=True)
                    # self.overlay_sampled_locations_on_gt(y, sampled_indices)
                    label = label.flatten(1, 2)  
                    ## select the labels of the sampled features
                    sampled_indices = sampled_indices.to(self.device)    
                    ## repeat the label for each sampled feature
                    ###################
                    if DATASET == "MSCOCO":
                        label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1])) 
                    elif DATASET == "NYUv2":
                        label_hat = label.gather(1, sampled_indices)
                        label_hat = label_hat.unsqueeze(-1)
                    ###################
                    normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                    label_hat = label_hat.flatten(0, 1)
                    self.feature_memory[idx:idx+normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                    self.label_memory[idx:idx+label_hat.size(0)] = label_hat.detach().cpu()
                    idx += normalized_sampled_features.size(0)
                    # memory.append(normalized_sampled_features.detach().cpu())
                    print(f"batch {i} has been processed at {time.ctime()}")
                    
        # self.save_memory()

    def save_memory(self):
        print('-'*50)
        print('Saving to ',F_MEM)
        print('Saving to ',L_MEM)
        torch.save(self.feature_memory.cpu(), F_MEM)
        torch.save(self.label_memory.cpu(), L_MEM)

    def load_memory(self):
        if os.path.isfile(F_MEM) and os.path.isfile(L_MEM):
            print("Loading memory from ", F_MEM, L_MEM)
            self.feature_memory = torch.load(F_MEM).to(self.device)
            self.label_memory = torch.load(L_MEM).to(self.device)
            return True
        return False


    def sample_features_depth(self, features, patchified_gts):
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(tqdm(patchified_gts)): #loop over elements in batch
            num_patches = gt.shape[0] * gt.shape[1]  # 36 * 36
            patch_indices = torch.arange(num_patches)
            shuffled_indices = patch_indices[torch.randperm(num_patches)]
            selected_indices = shuffled_indices[:self.num_sampled_features]
            sampled_indices.append(selected_indices)
            feature = features[k] 
            samples = feature[selected_indices] 
            sampled_features.append(samples)
        
        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)
        return sampled_features, sampled_indices

    def sample_features(self, features, pathified_gts):
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(tqdm(pathified_gts)):
            class_frequency = self.get_class_frequency(gt)
            patch_scores = self.get_patch_scores(gt, class_frequency).flatten()
            zero_score_idx = torch.where(patch_scores == 0)
            none_zero_score_idx = torch.where(patch_scores != 0)
            patch_scores[zero_score_idx] = 1e6

            uniform_x = torch.rand(none_zero_score_idx[0].size(0))
            patch_scores[none_zero_score_idx] *= uniform_x
            feature = features[k]
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)
            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)
        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)
        return sampled_features, sampled_indices

    '''
    Vectorized version of get_class_frequency
    '''
    def get_class_frequency(self, gt):
        num_classes = self.dataset_module.get_num_classes()
        class_frequency = torch.zeros(num_classes)
        for i in range(num_classes):
            class_existence = (gt == i).any(dim=2)
            class_frequency[i] = class_existence.sum()
        return class_frequency


    def get_patch_scores(self, gt, class_frequency):
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(class_frequency.shape[0]):
                    if torch.sum(gt[i, j] == k) > 0:
                        patch_scores[i, j] += class_frequency[k]
        return patch_scores
    
    def patchify_gt(self, gt, patch_size):
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h//patch_size, patch_size, w//patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h//patch_size, w//patch_size, c*patch_size*patch_size)
        return gt  

    def overlay_sampled_locations_on_gt(self, gts, sampled_indices):
        """
        This function overlays the sampled locations on the ground truth 
        and saves the figure in temp folder. It is used to check if the
        sampling is done correctly. For better visualization, turn off the
        uniform sampling when calling the sample_features function.

        Args:
            gts (torch.Tensor): ground truth tensor of shape (bs, c, h, w)
            sampled_indices (torch.Tensor): sampled indices of shape (bs, num_sampled_features)
        """
        maps = torch.zeros_like(gts)
        ## downsample the map to eval_spatial_resolution
        maps = F.interpolate(maps.float(), size=(self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution), mode="nearest").long()
        for i in range(maps.shape[0]):
            map = maps[i]
            sampled_idx = sampled_indices[i]
            map = map.flatten()
            map[sampled_idx] = 1
            map = map.reshape(1, self.feature_extractor.eval_spatial_resolution, self.feature_extractor.eval_spatial_resolution)
            maps[i] = map
        
        maps = F.interpolate(maps.float(), size=(gts.shape[2], gts.shape[3]), mode="nearest").long()
        ## save figures of maps and gts together
        for i in range(maps.shape[0]):
            map = maps[i]
            gt = gts[i]
            plt.imshow(gt.detach().cpu().numpy().transpose(1, 2, 0))
            plt.imshow(map.detach().cpu().numpy().transpose(1, 2, 0), alpha=0.1)
            plt.savefig(f"temp/map_{i}.png")
            plt.close()

    def recall(self, x):
        query_features, _ = self.feature_extractor.get_intermediate_layer_feats(x)
 
    def cross_attention(self, q, k, v, beta=0.02):
        """
        Args: 
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)
        """
        d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2) ## (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta ## (bs, num_patches, num_sampled_features)
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat
    
    def find_nearest_key_to_query(self, q):
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs*num_patches, d_k)
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).to(self.device)
        neighbors = neighbors.flatten()

        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.num_neighbour, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.num_neighbour, -1)
        return key_features, key_labels


    def incontext_evaluation(self, max_i=10):
        print("In-context evaluation has started")
        
        val_loader = self.dataset_module.get_val_dataloader(batch_size=1)
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.feature_extractor = self.feature_extractor.to(MODEL_DEVICE)
        label_hats = []
        lables = []
        with torch.no_grad():
            for i, (x, y, _) in enumerate(tqdm(val_loader)):
                #######################
                metric = PredsmIoU(81, 81) # num classes for the dataset 21 for VOC and 81 for MSCOCO and 91 for MSCOCO stuff
                #######################   
                x = x.to(self.device)
                y = y.to(self.device)
                _, _, h, w = x.shape
                features, cls_features,_ = self.feature_extractor.forward_features(x)
                _, I  = self.cluster_index.search(np.array(cls_features.detach().cpu()), 1)
                same_cluster_images = [path for path, cluster_id in self.cluster_assignments.items() if cluster_id == I[0][0]]
                sample_cluster_images = random.sample(same_cluster_images, 5) if len(same_cluster_images) >= 5 else same_cluster_images
                print("Cluster size: ", len(same_cluster_images))
                self.create_memory(sample_cluster_images)
                #     destination_dir = "/home/lbusser/cluster_images_1"
                #     for image_path in same_cluster_images:
                #         shutil.copy(image_path, destination_dir)
                self.create_NN() # Likely have to adjust the parameters
                ## copy the data of features to another variable
                q = features.clone()
                q = q.detach().cpu().numpy()

                ##########################################################################################################
                # Use a learnable mapper that can map the raw patch labels to a prediction label and use that, so not CA #
                ##########################################################################################################

                key_features, key_labels = self.find_nearest_key_to_query(q)
                label_hat =  self.cross_attention(features, key_features, key_labels) ## (bs, num_patches, label_dim)
                bs, _, label_dim = label_hat.shape
                label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
                resized_label_hats =  F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)
                label_hats.append(cluster_map.detach().cpu())
                lables.append(y.detach().cpu())
            try:
                if DATASET == "MSCOCO":
                    lables = torch.cat(lables) 
                    label_hats = torch.cat(label_hats)
                    ########################
                    valid_idx = lables != 255 #for pascal voc
                    # valid_idx = lables != 165 #for stuff segmentation mscoco
                    ########################
                    valid_target = lables[valid_idx]
                    valid_cluster_maps = label_hats[valid_idx]
                    metric.update(valid_target, valid_cluster_maps)
                    jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
                    print(f"eval finished, miou: {jac}")
                    print(f"tp: {tp}")
                    print(f"fp: {fp}")
                    print(f"fn: {fn}")
                elif DATASET == "NYUv2":
                    lables = torch.cat(lables)
                    label_hats = torch.cat(label_hats)
                    print("CHECK FOR VALIDNESS", torch.unique(label_hats))
                    # Calculate the Mean Squared Error (MSE)
                    mse = torch.mean((lables - label_hats) ** 2)
                    # Calculate the Root Mean Squared Error (RMSE)
                    rmse = torch.sqrt(mse)
                    # Calculate the Mean Absolute Error (MAE)
                    mae = torch.mean(torch.abs(lables - label_hats))
                    # Calculate R-squared (R²)
                    label_mean = torch.mean(lables)
                    ss_tot = torch.sum((lables - label_mean) ** 2)
                    ss_res = torch.sum((lables - label_hats) ** 2)
                    r_squared = 1 - ss_res / ss_tot
                    # Print the metrics
                    print(f"Mean Squared Error (MSE): {mse.item()}")
                    print(f"Root Mean Squared Error (RMSE): {rmse.item()}")
                    print(f"Mean Absolute Error (MAE): {mae.item()}")
                    print(f"R-squared (R²): {r_squared.item()}")
            except Exception as e:
                print('There was an error, but it is omited for now')
                print(e)
                pass
            
            torch.save(lables, os.path.join(EXP_DIR, 'gts.pt'))
            torch.save(label_hats, os.path.join(EXP_DIR, 'few_shot_preds.pt'))
    

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
    evaluator = HummingbirdEvaluation(feature_extractor  ,num_neighbour=NEIGHBOURS, augmentation_epoch=AUG_EPOCHS, memory_size=MEM_SIZE, task = "normal", device=device, cluster_index= cluster_index, evaluation_only = eval_only)
    

    