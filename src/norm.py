from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        RMSNorm
        rms = mean(x^2)
        x = weights * x / √(rms + eps)
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        y = x / rms * self.weight
        return y