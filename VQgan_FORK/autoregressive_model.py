import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from my_utils import set_device
from transformers import ImageGPTModel, ImageGPTConfig, ImageGPTForCausalImageModeling
from dino2gpt_mapper import *

class Autoregressive(nn.Module):
    def __init__(self):
        super(Autoregressive, self).__init__()
        
        self.n_embd = 512
        # self.config = ImageGPTConfig(vocab_size = 16386, n_positions= 2500 ,n_embd = self.n_embd, n_head = 8, n_layer = 24)
        # self.igpt = ImageGPTForCausalImageModeling(self.config)
        ####################
        self.igpt = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-small')
        self.igpt.config.max_position_embeddings = 2500
        self.igpt.config.vocab_size = 16384
        self.igpt.transformer.wte = nn.Embedding(16384, self.igpt.config.hidden_size)
        self.igpt.transformer.wpe = nn.Embedding(self.igpt.config.max_position_embeddings, self.igpt.config.hidden_size)
        self.igpt.lm_head = nn.Linear(self.igpt.config.hidden_size, 16384)
       
        # Initialize weights similar to Image GPT
        nn.init.normal_(self.igpt.transformer.wte.weight, std = 0.02)
        nn.init.normal_(self.igpt.transformer.wpe.weight, std = 0.02)
        nn.init.normal_(self.igpt.lm_head.weight, std = 0.02)
        if self.igpt.lm_head.bias is not None:
            nn.init.zeros_(self.igpt.lm_head.bias)
       ####################
        self.mapper = DINO2GPT(768, self.n_embd)


    def forward(self, support_tokens, query_tokens, support_features, query_features, max_new_tokens = 196):
        support_embeddings = self.igpt.transformer.wte(support_tokens)
        query_embeddings = self.igpt.transformer.wte(query_tokens)
        support_features_proj = self.mapper(support_features)
        query_features_proj = self.mapper(query_features)
 
        visual_sentence = self.create_sentences(support_features_proj, support_embeddings, query_features_proj, query_embeddings)
        # print(visual_sentence.shape)
        targets = torch.full((visual_sentence.shape[0],visual_sentence.shape[1]), -100)
        # print(targets.shape)
        targets[:,-max_new_tokens:] = query_tokens
        out = self.igpt(inputs_embeds = visual_sentence, labels = targets.cuda())        
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
        
        visual_sentences = []
        for i in range(spt_sz):
            combined_features = torch.cat([scaled_supp_features[:,i], supp_labels[:,i]], dim=1)
            visual_sentences.append(combined_features)
        visual_sentences = torch.stack(visual_sentences, dim=1).view(bs, -1, dim)
        visual_sentences = torch.cat([visual_sentences, scaled_query_features.squeeze(1)], dim=1)
      
        if query_labels is not None:
            visual_sentences = torch.cat([visual_sentences, query_labels], dim=1)
        return visual_sentences
    
    def generate(self, support_tokens, support_features, query_features, max_new_tokens=196, perplex=False):
        support_embeddings = self.igpt.transformer.wte(support_tokens)
        support_features_proj = self.mapper(support_features)
        query_features_proj = self.mapper(query_features)
    
        visual_sentence = self.create_sentences(support_features_proj, support_embeddings, query_features_proj) 
        generated = visual_sentence
        generated_tokens = []
        
        if perplex:
            log_likelihoods = []
        
        for _ in range(max_new_tokens):
            outputs = self.igpt(inputs_embeds=generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            next_token_embeds = self.igpt.transformer.wte(next_token)
            
            if perplex:
                log_likelihood = F.log_softmax(next_token_logits, dim=-1)
                selected_log_likelihood = torch.gather(log_likelihood, 1, next_token)
                log_likelihoods.append(selected_log_likelihood)
            
            generated = torch.cat((generated, next_token_embeds), dim=1)
            generated_tokens.append(next_token)
        
        generated_tokens = torch.stack(generated_tokens, dim=1).squeeze()
        
        if perplex:
            log_likelihoods = torch.cat(log_likelihoods, dim=1)
            avg_log_likelihood = torch.mean(log_likelihoods)
            perplexity = torch.exp(-avg_log_likelihood)
            print("PERPLEXITY SCORE:", perplexity)
        
        return generated_tokens
    
