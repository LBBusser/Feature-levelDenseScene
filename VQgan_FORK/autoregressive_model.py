import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from my_utils import set_device
from transformers import ImageGPTModel, ImageGPTConfig, DeiTConfig, DeiTModel, ImageGPTForCausalImageModeling
from dino2gpt_mapper import *

class Autoregressive(nn.Module):
    def __init__(self):
        super(Autoregressive, self).__init__()
        self.bos_token = 16384
        self.eos_token = 16385
        self.n_embd = 512
        # self.config = ImageGPTConfig(vocab_size = 16386, n_positions= 2500 ,n_embd = self.n_embd, n_head = 8, n_layer = 24)
        # self.config = DeiTConfig(image_size = 504, patch_size = 14, num_hidden_layers=4, num_attention_heads=8)
        
        # self.igpt = ImageGPTForCausalImageModeling(self.config)
        ####################
        self.igpt = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-small')
        self.igpt.config.max_position_embeddings = 2500
        self.igpt.config.vocab_size = 16386
        self.igpt.transformer.wte = nn.Embedding(16386, self.igpt.config.hidden_size)
        self.igpt.transformer.wpe = torch.nn.Embedding(16386, self.igpt.config.hidden_size)
        self.igpt.lm_head = torch.nn.Linear(self.igpt.config.hidden_size, 16386)
       
        # Initialize weights similar to Image GPT
        nn.init.xavier_normal_(self.igpt.transformer.wte.weight)
        nn.init.xavier_normal_(self.igpt.transformer.wpe.weight)
        nn.init.xavier_uniform_(self.igpt.lm_head.weight)
        if self.igpt.lm_head.bias is not None:
            nn.init.zeros_(self.igpt.lm_head.bias)
   
       ####################
        self.mapper = DINO2GPT(768, self.n_embd)

    def forward(self, support_tokens, query_tokens, support_features, query_features, max_new_tokens = 196):
        #Map the features to output of w2e size.
        #Get the output of the w2e from the model.
        #Create the visual sentence with bos and eos tokens (interchange features and labels, like before), pass that through the model.
        bos_embedding = self.igpt.transformer.wte(torch.full((support_tokens.shape[0], 1), self.bos_token, dtype=torch.long).cuda())
        eos_embedding = self.igpt.transformer.wte(torch.full((support_tokens.shape[0], 1), self.eos_token, dtype=torch.long).cuda())
        support_embeddings = self.igpt.transformer.wte(support_tokens)
        query_embeddings = self.igpt.transformer.wte(query_tokens)
        support_features_proj = self.mapper(support_features)
        query_features_proj = self.mapper(query_features)
 
        visual_sentence = self.create_sentences(support_features_proj, support_embeddings, query_features_proj, query_embeddings).flatten(1,2)  
        #maybe do without bos and eos
        input = torch.cat([bos_embedding, visual_sentence[:,:-max_new_tokens], eos_embedding, visual_sentence[:,-max_new_tokens:]], dim=1)
        targets = torch.full((input.shape[0],input.shape[1]), -100)
        targets[:,-max_new_tokens:] = query_tokens
        out = self.igpt(inputs_embeds = input, labels = targets.cuda())        
        return out.loss

    def normalize_and_scale_features(self, features, label_norms):
        # Compute the norms of the features
        feature_norms = features.norm(p=2, dim=-1, keepdim=True)
        
        # Normalize the features
        normalized_features = features / (feature_norms + 1e-8)
        avg_label_norm = label_norms.mean()
        # Scale normalized features to match the target norm
        scaled_features = normalized_features * avg_label_norm
        
        return scaled_features
    
    def create_sentences(self, supp_features, supp_labels, query_features, query_labels=None):
        bs, spt_sz, num_patches, dim = supp_features.shape
        # Interleave supp_features and supp_labels with corresponding separators
        # The pattern is feature1 -> label1 -> feature2 -> label2
        scaled_supp_features = self.normalize_and_scale_features(supp_features, supp_labels.norm(p=2, dim=-1, keepdim=True))
        scaled_query_features = self.normalize_and_scale_features(query_features, supp_labels.norm(p=2, dim=-1, keepdim=True))
        
        # supp_label_norms = supp_labels.norm(p=2, dim=-1, keepdim=True)
        # print("Norms of support labels:", supp_label_norms.squeeze(-1))
        # print("Norms of scaled support features:", scaled_supp_features.norm(p=2, dim=-1))
        visual_sentences = []
        for i in range(spt_sz):
            visual_sentences.append(scaled_supp_features[:,i])
            visual_sentences.append(supp_labels[:,i])

        visual_sentences = torch.stack(visual_sentences, dim=1)
        visual_sentences = torch.cat([visual_sentences, scaled_query_features], dim=1)
        if query_labels is not None:
            visual_sentences = torch.cat([visual_sentences, query_labels.unsqueeze(1)], dim=1)
        return visual_sentences
    
    def generate(self, support_tokens, support_features, query_features, max_new_tokens = 196):
        bos_embedding = self.igpt.transformer.wte(torch.full((support_tokens.shape[0], 1), self.bos_token, dtype=torch.long).cuda())
        eos_embedding = self.igpt.transformer.wte(torch.full((support_tokens.shape[0], 1), self.eos_token, dtype=torch.long).cuda())
        support_embeddings = self.igpt.transformer.wte(support_tokens)
        support_features_proj = self.mapper(support_features)
        query_features_proj = self.mapper(query_features)
 
        visual_sentence = self.create_sentences(support_features_proj, support_embeddings, query_features_proj).flatten(1,2)  
        input = torch.cat([bos_embedding, visual_sentence, eos_embedding], dim=1)
        generated = input
        generated_tokens = []
        for _ in range(max_new_tokens):
            outputs = self.igpt(inputs_embeds=generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            next_token_embeds = self.igpt.transformer.wte(next_token)
            generated = torch.cat((generated, next_token_embeds), dim=1)
            generated_tokens.append(next_token)
        return torch.stack(generated_tokens, dim=1).squeeze()

