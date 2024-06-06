import torch
import torch.nn as nn
import torch.nn.functional as F

class DINO2GPT(nn.Module):
    def __init__(self, dinov2_feature_dim, wte_size, hidden_dim=256):

        super(DINO2GPT, self).__init__()
        # Normalization layer
        self.norm0 = nn.LayerNorm(dinov2_feature_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Fully connected layers
        self.fc1 = nn.Linear(dinov2_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, wte_size)
        # Dropout layer
        # self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        self.activation = nn.GELU()
        self._init_weights()

    def forward(self, x):
        bs, seq_size, num_patches, dim = x.shape
        # First fully connected layer with ReLU activation and dropout
        x = self.norm0(x)
        x = self.activation(self.fc1(x))
        # x = self.dropout(x)
        x = self.norm(x)
        # Second fully connected layer with ReLU activation and dropout
        x = self.activation(self.fc2(x))
        x = self.norm2(x)
        # x = self.dropout(x)
        # x = self.norm2(x)
        # Final layer to output indices for the VQGAN codebook
        x = self.fc3(x)
        # x = torch.argmax(x, dim=-1)
        # Optionally, you can apply softmax to get a probability distribution over the codebook indices
        # x = F.softmax(x, dim=-1)
        return x
    
    def _init_weights(self):
        # Initialize weights similar to transformer architectures
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std = 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
