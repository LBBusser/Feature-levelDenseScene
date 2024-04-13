import math
from typing import Optional
import numpy as np
from tqdm import tqdm
import torch
import faiss
from models import FeatureExtractorBeta as FeatureExtractor
from torch import nn
from torch.nn import functional as F
from my_utils import set_device
import scann
from transformers import ImageGPTModel, ImageGPTConfig, DeiTConfig, DeiTModel


class MetaLearner(nn.Module):
    def __init__(self, resize, patch_size = 14, model_dim = 768):
        super(MetaLearner, self).__init__()
        self.device = set_device()
        self.resize = resize
        self.patch_size = patch_size
        self.is_multi_gpu = True if torch.cuda.device_count() > 1 else False
        self.model_dim = model_dim
        self.label_dim = self.patch_size**2
        
        self.config = ImageGPTConfig(vocab_size = 1024, n_positions= 4096 ,n_embd = self.model_dim, n_head = 8, n_layer = 4)
        # self.config = DeiTConfig(image_size = 504, patch_size = 14, num_hidden_layers=4, num_attention_heads=8)
        self.igpt = ImageGPTModel(self.config)
        # self.deit = DeiTModel(self.config)
        self.label_emb = nn.Linear(self.label_dim, self.model_dim)
        self.logit_layer = nn.Linear(self.model_dim, self.label_dim*81)
        self.igpt.to(self.device)
        self.separator_feature = nn.Parameter(torch.randn(1,1,self.model_dim))
        self.separator_label = nn.Parameter(torch.randn(1,1,self.model_dim))

    def forward(self, supp_features, supp_labels, qry_features, qry_labels):
        bs, seq_len_sp, dim = supp_features.shape #here bs is number of query images
        supp_labels = supp_labels.to(self.device)
        qry_labels = qry_labels.to(self.device)
        supp_features = supp_features.to(self.device)
        qry_features = qry_features.view(bs,1,dim).to(self.device)

        label_proj_supp = self.label_emb(supp_labels.reshape(-1, self.label_dim).float())
        label_proj_supp = label_proj_supp.view(bs, seq_len_sp, dim)
        label_proj_qry = self.label_emb(qry_labels.reshape(-1,self.label_dim).float())
        label_proj_qry = label_proj_qry.view(bs,1, dim)
    
        combined_input = self.combine_input(supp_features, label_proj_supp, qry_features, label_proj_qry) #shape [bs, seq_len, dim]
        out = self.igpt(inputs_embeds = combined_input)

        pred = out.last_hidden_state[:,-2,:].squeeze() # Prediction of query_labels
    
        # out = self.logit_layer(pred)
        # allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert bytes to GB
        # print(f"GPU memory allocated: {allocated:.2f} GB")
        pred = self.logit_layer(pred)
        return pred

    def combine_input(self, supp_features, supp_labels, query_features, query_labels):
        bs, seq_len, _ = supp_features.shape
        separator_feature_expanded = self.separator_feature.expand(bs, seq_len, self.model_dim)
        separator_label_expanded = self.separator_label.expand(bs, seq_len, self.model_dim)

        # Prepare query separators, expanded across the batch
        query_separator_feature = self.separator_feature.expand(bs, 1, self.model_dim)
        query_separator_label = self.separator_label.expand(bs, 1, self.model_dim)

        # Interleave supp_features and supp_labels with corresponding separators
        # The pattern is feature -> separator_feature -> label -> separator_label
        # Achieve this by concatenating along a new dimension and then reshaping
        interleaved_support = torch.cat([
            supp_features.unsqueeze(2),
            separator_feature_expanded.unsqueeze(2),
            supp_labels.unsqueeze(2),
            separator_label_expanded.unsqueeze(2)
        ], dim=2).view(bs, seq_len * 4, self.model_dim)

        # Concatenate the query part at the end of each sequence
        query_part = torch.cat([
            query_features, query_separator_feature, query_labels, query_separator_label
        ], dim=1)

        # Finally, concatenate the support sequences with the query sequences for each batch
        combined_sequence = torch.cat([interleaved_support, query_part], dim=1)

        return combined_sequence
    

