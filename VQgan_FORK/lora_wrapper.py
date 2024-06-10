import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAConv1DWrapper(nn.Module):
    """
    A wrapper module that applies LORA to the weights of a convolutional layer.
    """

    def __init__(self, module: nn.Module, rank: int):
        """
        Initializes the LoRAConv1DWrapper instance.

        Parameters:
            module (nn.Module): The base module whose weights are to be adapted.
            rank (int): The rank for the low-rank matrices A and B. If set to 0, LoRA is effectively disabled.
        """
        super().__init__()
        if rank < 0:
            raise ValueError("Rank must be a non-negative integer")

        self.base_module = module

        out_features, in_features = self.base_module.weight.shape

        self.lora_rank = rank
        if self.lora_rank > 0:
            self.W_A = nn.Parameter(
                torch.zeros((self.lora_rank, in_features)),
                requires_grad=True)
            self.W_B = nn.Parameter(
                torch.zeros((out_features, self.lora_rank)),
                requires_grad=True)

            # self.print_trainable_parameters()

            # freeze the base module's parameters, only focus on updating lora weights
            self.base_module.weight.requires_grad = False
            if self.base_module.bias is not None:
                self.base_module.bias.requires_grad = False
        else:
            print(f"Creating LoRAConv1DWrapper with no rank adaptation: rank {self.lora_rank}")

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes or resets the parameters of the LoRA matrices A and B to their default values.
        This method typically mirrors the initialization logic of the base module.
        """
        if self.lora_rank > 0:
            # initialize A matrix
            nn.init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
            # initialize B matrix to 0
            nn.init.zeros_(self.W_B)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the base module and the additional parameters added by LoRA.
        """
        base_params = sum(p.numel() for p in self.base_module.parameters())
        lora_params = sum(p.numel() for p in [self.W_A, self.W_B])

        print(f"Trainable parameters in base module: {base_params}")
        print(f"Trainable parameters in LoRA (base module frozen): {lora_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the LoRAConv1DWrapper, applying low-rank adaptations to the base module's weights.

        Parameters:
            x (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output of the module after applying the low-rank adapted forward pass.
        """
        if self.lora_rank > 0:
            # Compute the base module's forward pass with adapted weights
            # print(self.W_A.shape)
            # print(self.W_B.shape)
            adapted_weight = self.base_module.weight + self.W_B @ self.W_A
            return F.linear(x, adapted_weight.T, self.base_module.bias)
        else:
            # Perform a standard forward pass using the base module's original weights and bias
            return F.linear(x, self.base_module.weight, self.base_module.bias)