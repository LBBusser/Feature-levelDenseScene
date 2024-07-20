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
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader import CocoMemoryTasksDataLoader, NYUMemoryTasksDataLoader
from image_transformations import Compose, RandomResizedCrop, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader
import lightning as L
# PATH = str(Path.cwd().parent)
MODELS_PATH = "data/models"
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
        config_path = "data/models/vqgan/configs/config16.yaml"
        ckpt_path = "data/models/vqgan/checkpoints/model16.ckpt"
        vit_model = torch.hub.load('facebookresearch/dinov2', MODEL)
        config = self.load_config(config_path)
        self.vq_model = self.load_vqgan(config, ckpt_path)
        self.feature_extractor = FeatureExtractor(vit_model)
        self.num_patches = (self.resize//14)**2
        self.vq_dim = self.resize//16
        self.feature_dim = self.feature_extractor.d_model
        self.model_name = args.trained_model_name
        self.ar_model = Autoregressive()
    
        #Freeze vq
        for param in self.vq_model.parameters():
            param.requires_grad = False
        #Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def print_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                if grad_norm >= 2:
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
        support_features, support_lbl_idx = self.create_memory(x_spt, y_spt)   
        with torch.no_grad():
            query_features, _, _= self.feature_extractor.forward_features(x_qry.squeeze(1)) #shape query [bs, num_patches , d_k]
            t, _, [_, _, qry_lbl_idx] = self.vq_model.encode(y_qry.squeeze(1))
            qry_lbl_idx = qry_lbl_idx.flatten(1,2)
        query_features = query_features.unsqueeze(1)
        loss = self.ar_model(support_lbl_idx.long(), qry_lbl_idx.long(), support_features, query_features)
        return loss
    
    def generate(self, x_spt, y_spt, x_qry, y_qry, perplex = False):
        total_mae = 0
        batch = 0
        support_features, support_lbl_idx = self.create_memory(x_spt, y_spt)
        with torch.no_grad():
            query_features,_,_= self.feature_extractor.forward_features(x_qry.squeeze(1))
            # _,_, [_, _, qry_lbl_idx] = self.vq_model.encode(y_qry.squeeze(1))
        query_features = query_features.unsqueeze(1)
        out = self.ar_model.generate(support_lbl_idx.long(), support_features, query_features, perplex=perplex)
        pred = self.vq_model.quantize.get_codebook_entry(indices = out, shape = (out.shape[0], self.vq_dim, self.vq_dim, 256))
        # Calculate and print accuracy
        # accuracy = self.calculate_accuracy(out, qry_lbl_idx.flatten(1,2))
        # print(f"Accuracy: {accuracy:.2f}%")
        out = self.vq_model.decode(pred)
        return out
    
    def training_step(self, batch, batch_idx):
        x_spt, y_spt, x_qry, y_qry = batch
        loss = self(x_spt, y_spt, x_qry, y_qry)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_spt, y_spt, x_qry, y_qry = batch
        val_loss = self(x_spt, y_spt, x_qry, y_qry)
        self.log("val_loss", val_loss, prog_bar=True, on_step= False, sync_dist = True)
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0, perplex = True):
        x_spt, y_spt, x_qry, y_qry = batch
        out = self.generate(x_spt, y_spt, x_qry, y_qry, perplex = perplex)
        return {'pred': out, 'image': x_qry, 'mask': y_qry}


    def on_after_backward(self):
        # Log gradient norms after the backward pass
        self.print_gradients()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
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

    
