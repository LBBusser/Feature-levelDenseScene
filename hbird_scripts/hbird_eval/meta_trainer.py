import os, sys
from copy import deepcopy
sys.path.append('/home/lbusser/taming-transformers/')

import numpy as np
import torchvision.transforms as trn
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from models import FeatureExtractorBeta as FeatureExtractor
from meta_learner import MetaLearner
from my_utils import *
import torch.optim as optim
import scann
import argparse
import faiss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader import CocoMemoryTasksDataLoader, NYUMemoryTasksDataLoader
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader

# PATH = str(Path.cwd().parent)
MODELS_PATH = "/home/lbusser/hbird_scripts/hbird_eval/data/models"


class MetaTrainer(nn.Module):
    """
    Adapted from https://github.com/dragen1860/MAML-Pytorch/blob/98a00d41724c133bd29619a2fb2cc46dd128a368/meta.py
    """

    def __init__(self, args, feature_extractor, mode = 'train'):
        super(MetaTrainer, self).__init__()
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.resize = args.img_size
        self.task_num = args.task_num
        self.num_leaves = args.num_leaves
        self.num_leaves_to_search = args.num_leaves_to_search
        self.reorder = args.reorder
        self.patch_size = args.patch_size
        self.num_neighbors = args.num_neighbors
        self.device = set_device()
        self.feature_extractor = feature_extractor
        self.model_name = args.trained_model_name
        # self.log_file_path = PATH + "/logs/log_{}.txt".format(experiment_id)
        self.mode = mode
        self.model = MetaLearner(self.resize, patch_size = self.patch_size)
        self.model.train()
        # Loading pre-trained model 
        self.feature_extractor.to(self.device)
        self.model.to(self.device)
        #Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


    def create_NN(self):
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), self.num_neighbors, "dot_product").tree(
    num_leaves=self.num_leaves, num_leaves_to_search=self.num_leaves_to_search, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=0.1).reorder(self.reorder).build()


    def find_nearest_key_to_query(self, q):
        bs, num_patches, dim = q.shape
        reshaped_q = q.reshape(bs*num_patches, dim)
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors)#.to(self.device)
        neighbors = neighbors.flatten()

        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.num_neighbors, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.num_neighbors, -1)
        return key_features, key_labels
    
    def create_memory(self, x_spt, y_spt):
        with torch.no_grad():
           
            support_features, _,_ = self.feature_extractor.forward_features(x_spt) #shape support [setsz, num_patches , d_k]
            gt = self.patchify_gt(y_spt)
            gt = gt.flatten(0,1)
            normalized_sampled_features = support_features / torch.norm(support_features, dim=1, keepdim=True)
            normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
            self.feature_memory = normalized_sampled_features.detach().cpu()
            self.label_memory = gt.detach().cpu()

        return self.feature_memory, self.label_memory

    def patchify_gt(self, y_spt):
        sz, c, h, w = y_spt.shape
        gt = y_spt.reshape(sz, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(sz, h//self.patch_size, w//self.patch_size, c*self.patch_size*self.patch_size)
        gt = gt.flatten(1, 2)  
        # gt = gt.flatten(0, 1)
        return gt

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w] images that are selected from the cluster as support
        :param y_spt:   [b, setsz, 1, h, w] segmentation masks of the support images
        :param x_qry:   [b, querysz, c_, h, w] image(s) which make(s) up the query
        :param y_qry:   [b, querysz, 1, h, w] segmentation mask of the query image(s)
        :return:
        """
        task_num = x_spt.shape[0]
        losses = []
        
        for i in range(task_num):
            _, _ = self.create_memory(x_spt[i].to(self.device), y_spt[i].to(self.device))
            self.create_NN()
            
            #Pre-process data, extract the features most similar to selected query patch.
            with torch.no_grad():
                query_features, _, _ = self.feature_extractor.forward_features(x_qry[i].to(self.device)) #shape query [querysz, num_patches , d_k]
                # print("query_features shape", query_features.shape)
            q = query_features.clone()
            q = q.detach().cpu().numpy()
            #Patchify the labels in similar way to feature
            query_labels = self.patchify_gt(y_qry[i])
            #Find the most similar features and corresponding label to query patches
            support_features, support_labels = self.find_nearest_key_to_query(q)
            if self.mode=='train':
                #Select subset of patches
                subset_patches_size = 128
                selected_idx = np.random.choice(query_features.shape[1], subset_patches_size, replace=False)
                support_features = support_features[:, selected_idx, : , :]
                support_labels = support_labels[:, selected_idx, : , :]
                query_features = query_features[:,selected_idx, :]
                query_labels = query_labels[:, selected_idx, :]

                #Shape of features is [querysz, num_patches, num_neighbour, dim] and labels is [querysz, num_patches, num_neighbour, 14*14]
    
                out = self.model(support_features.flatten(0,1), support_labels, query_features.flatten(0,1), query_labels)
                out = out.reshape(-1, 81)
                # loss = F.l1_loss(out, query_labels.view(query_labels.size(0),-1).to(self.device), reduction= 'mean')
                loss = F.cross_entropy(out, query_labels.flatten().long().to(self.device))
                losses.append(loss)
            else:
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model(support_features.flatten(0,1), support_labels.flatten(0,1), query_features.flatten(0,1), query_labels.flatten(0,1))
                    pred = pred.reshape(-1, 81)
                    pred = F.softmax(pred, dim=1)
                    pred = pred.argmax(dim=1)

                    
        if self.mode == 'train':
            mean_loss = torch.stack(losses).mean()
            return mean_loss
        else:
            pred = pred.view(1, 36, 36, 14, 14)
            pred = pred.permute(0, 1, 3, 2, 4).contiguous()
            pred = pred.view(-1,1, 36 * 14, 36 * 14)
            return pred


    def save_model(self):
        torch.save({'model': self.model.state_dict()}, os.path.join(MODELS_PATH, "{}".format(self.model_name)))
        print("Model saved on path {}".format(MODELS_PATH))

    def load_model(self):
        model_dict = torch.load(MODELS_PATH + self.model_name, map_location=torch.device(self.device))
        self.model.load_state_dict(model_dict)
        print("Model loaded from {}".format(MODELS_PATH))


def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment_id', type=int, default=5)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=6)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2)
    argparser.add_argument('--img_size', type=int, help='img_size', default=504)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=5)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--dataset', type=str, default='coco')

    # COCO training / non-episodic image captioning
    argparser.add_argument('--coco_num_epochs', type=int, help='epoch number for training', default=100)
    argparser.add_argument('--coco_batch_size', type=int, help='batch size for training', default=8)
    argparser.add_argument('--coco_annotations_path', type=str, default='/home/lbusser/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=5)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for training', default=3e-05)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=1000)
    argparser.add_argument('--trained_model_name', type=str, default='default.pt')
    argparser.add_argument('--patch_size', type=int, default=14)
    #SCANN setup
    argparser.add_argument('--num_leaves', type=int, default=1)
    argparser.add_argument('--num_leaves_to_search', type=int, default=1)
    argparser.add_argument('--reorder', type=int, default=1800)
    argparser.add_argument('--num_neighbors', type=int, default=5)
    args = argparser.parse_args()

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
    cluster_index = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_index.index")
    # cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
    # get_file_path("/ssdstore/ssalehi/dataset/val1/JPEGImages/")
    # test_video_data_module(logger)
    few_shot_ds = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", batchsz=10, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index,cluster_assignment="/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl", transforms = (image_train_transform, shared_train_transform))
    # few_shot_ds = NYUMemoryTasksDataLoader(data_path="/scratch-shared/nyu_hbird/nyu_data/data/", mode="train", batchsz=10, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, transforms = (image_train_transform, shared_train_transform))
    train_set, val_set = train_test_split(few_shot_ds, test_size=0.2, random_state=0)
    eval_spatial_resolution = input_size // 14
    vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)

    feature_extractor = FeatureExtractor(vit_model)
    test = MetaTrainer(args, feature_extractor)
    dl_train = DataLoader(train_set, batch_size=2, shuffle=True, pin_memory=True)
    dl_val = DataLoader(val_set, batch_size=2, shuffle=True, pin_memory=True)
    for x_spt, y_spt, x_qry, y_qry in dl_val:
        out = test.forward(x_spt, y_spt, x_qry, y_qry)
    