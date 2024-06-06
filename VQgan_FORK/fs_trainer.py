import os, sys
import numpy as np
import torchvision.transforms as trn
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.nn import functional as F
from models import FeatureExtractorBeta as FeatureExtractor
sys.path.append('/home/lbusser/taming-transformers/')
from taming.models.vqgan import VQModel, GumbelVQ
from omegaconf import OmegaConf
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from autoregressive_model import *
from my_utils import *
import torch.optim as optim
import scann
import argparse
import pytorch_lightning as pl
import faiss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader import CocoMemoryTasksDataLoader, NYUMemoryTasksDataLoader
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader
import lightning as L
# PATH = str(Path.cwd().parent)
MODELS_PATH = "/home/lbusser/hbird_scripts/hbird_eval/data/models"
MODEL = 'dinov2_vitb14'


class MetaTrainer(L.LightningModule):
    """
    """

    def __init__(self, args):
        super(MetaTrainer, self).__init__()
        self.lr = args.lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.batch_size  = args.batch_size
        self.data_size = args.data_size
        self.num_epochs = args.num_epochs
        self.warm_up_steps = args.warm_up_steps
        self.resize = args.img_size
        config_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/vqgan/configs/config16.yaml"
        ckpt_path = "/home/lbusser/hbird_scripts/hbird_eval/data/models/vqgan/checkpoints/model16.ckpt"
        vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
        config = self.load_config(config_path)
        self.vq_model = self.load_vqgan(config, ckpt_path)
        self.feature_extractor = FeatureExtractor(vit_model)
        self.num_patches = (self.resize//14)**2
        self.vq_dim = self.resize//16
        self.feature_dim = self.feature_extractor.d_model
        self.model_name = args.trained_model_name
        self.ar_model = Autoregressive()
        # self.ar_model.igpt.transformer.wte.register_forward_hook(self.compute_norm_for_layer("wte"))
        # self.ar_model.mapper.register_forward_hook(self.compute_norm_for_layer("mapper"))
        # random_layer = 6
        # self.ar_model.igpt.transformer.h[random_layer].register_forward_hook(self.compute_norm_for_layer(f"layer {random_layer}"))
        #Freeze vq
        for param in self.vq_model.parameters():
            param.requires_grad = False
        #Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    # def compute_norm_for_layer(self, layer_name):
    #     def hook(module, input, output):
    #         if isinstance(output, tuple):
    #             output =output[0]
    #         norm = output.norm(2).item()  # Compute the L2 norm
    #         print(f"Norm of output at {layer_name}: {norm}")
    #     return hook
    
    def print_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None and ("wte" in name or "wpe" in name or "mapper" in name):
                grad_norm = param.grad.norm(2).item()
                print(f'Gradient Norm for {name}: {grad_norm}')
    
    def create_memory(self, x_spt, y_spt):
        bs, spt_sz, ch, h,w = x_spt.shape
      
        feature_memory = torch.zeros(bs, spt_sz , self.num_patches, self.feature_dim)
        label_memory = torch.zeros(bs, spt_sz, self.vq_dim**2)
        with torch.no_grad():
            for i in range(x_spt.shape[0]):
                support_features, _, _= self.feature_extractor.forward_features(x_spt[i]) #shape support_features:  [setsz, num_patches , d_k]
                _, _, [_, _, label_indices] = self.vq_model.encode(y_spt[i])
                # support_features = support_features.view(spt_sz, -1)
                label_indices = label_indices.flatten(1,2)
                feature_memory[i] = support_features
                label_memory[i] = label_indices
        return feature_memory.cuda(), label_memory.cuda()
    

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        """

        support_features, support_lbl_idx = self.create_memory(x_spt, y_spt)   
        with torch.no_grad():
            query_features, _, _= self.feature_extractor.forward_features(x_qry.squeeze(1)) #shape query [bs, num_patches , d_k]
            t, _, [_, _, qry_lbl_idx] = self.vq_model.encode(y_qry.squeeze(1))
            qry_lbl_idx = qry_lbl_idx.flatten(1,2)
        query_features = query_features.unsqueeze(1)
        loss = self.ar_model(support_lbl_idx.long(), qry_lbl_idx.long(), support_features, query_features)
        return loss
    
    def generate(self, x_spt, y_spt, x_qry, y_qry):
        support_features, support_lbl_idx = self.create_memory(x_spt, y_spt)
        with torch.no_grad():
            query_features,_,_= self.feature_extractor.forward_features(x_qry.squeeze(1))
            _,_, [_, _, qry_lbl_idx] = self.vq_model.encode(y_qry.squeeze(1))
        query_features = query_features.unsqueeze(1)
        out = self.ar_model.generate(support_lbl_idx.long(), support_features, query_features)
        print(out.shape)
        pred = self.vq_model.quantize.get_codebook_entry(indices = out, shape = (out.shape[0], self.vq_dim, self.vq_dim, 256))
        # Calculate and print accuracy
        accuracy = self.calculate_accuracy(out, qry_lbl_idx.flatten(1,2))
        print(f"Accuracy: {accuracy:.2f}%")
        out = self.vq_model.decode(pred)
        return out
    
    def training_step(self, batch, batch_idx):
        x_spt, y_spt, x_qry, y_qry = batch
        loss = self(x_spt, y_spt, x_qry, y_qry)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_spt, y_spt, x_qry, y_qry = batch
        val_loss = self(x_spt, y_spt, x_qry, y_qry)
        self.log("val_loss", val_loss, prog_bar=True, on_step= True, sync_dist = True)
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_spt, y_spt, x_qry, y_qry = batch
        out = self.generate(x_spt, y_spt, x_qry, y_qry)
        return {'pred': out, 'image': x_qry, 'mask': y_qry}


    def on_after_backward(self):
        # Log gradient norms after the backward pass
        self.print_gradients()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay = 1e-5)
        # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warm_up_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warm_up_steps,
            num_training_steps = (self.data_size/self.batch_size) * self.num_epochs
        )
        return [optimizer], [scheduler]
    
    def calculate_accuracy(self, out, qry_lbl_idx):
    # Ensure the dimensions match for comparison
        if out.shape != qry_lbl_idx.shape:
            raise ValueError(f"Shape mismatch: out shape {out.shape} and qry_lbl_idx shape {qry_lbl_idx.shape}")

        # Calculate accuracy
        correct_predictions = (out == qry_lbl_idx).sum().item()
        total_predictions = qry_lbl_idx.numel()
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, os.path.join(MODELS_PATH, "{}".format(self.model_name)))
        print("Model saved on path {}".format(MODELS_PATH))

    def load_model(self):
        model_dict = torch.load(MODELS_PATH + self.model_name, map_location=torch.device(self.device))
        self.model.load_state_dict(model_dict)
        print("Model loaded from {}".format(MODELS_PATH))

    def load_config(self, config_path):
        config = OmegaConf.load(config_path)
        return config

    def load_vqgan(self, config, ckpt_path=None, is_gumbel=False):
        if is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params, sane_index_shape= True)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment_id', type=int, default=5)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
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
    cluster_index_nyu = faiss.read_index("/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_index.index")
   
    image_paths = pd.read_csv("/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/nyu2_train.csv")
    train_idx, val_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state = 0)
    # few_shot_ds = CocoMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/mscoco_hbird", mode="train", batchsz=10, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index,cluster_assignment="/home/lbusser/hbird_scripts/hbird_eval/data/MSCOCO_dinov2_vitb14_1500_cluster_results/train/cluster_assignments.pkl", transforms = (image_train_transform, shared_train_transform))
    few_shot_ds = NYUMemoryTasksDataLoader(data_path="/scratch-shared/combined_hbird/nyu_hbird/nyu_data/data/", mode="train", batchsz=100, k_shot=3, k_query=2, resize=504, cluster_index= cluster_index_nyu, cluster_assignment = '/home/lbusser/hbird_scripts/hbird_eval/data/NYUv2_dinov2_vitb14_500_cluster_results/cluster_assignments.pkl', mode_idx = train_idx, transforms = (image_train_transform, shared_train_transform))
    eval_spatial_resolution = input_size // 14
    vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)

    feature_extractor = FeatureExtractor(vit_model)
    test = MetaTrainer(args, feature_extractor)
    dl_train = DataLoader(few_shot_ds, batch_size=2, shuffle=True, pin_memory=True)
    for x_spt, y_spt, x_qry, y_qry in tqdm(dl_train):
        out = test.forward(x_spt, y_spt, x_qry, y_qry)
    