from torch import nn
import torch

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): 输入张量的维度
        eps (float): 数值稳定的eps
    """
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
        """
        RMSNorm 前向过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: Norm操作后的张量，和输入张量相同shape。
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        y = x / rms * self.weight
        return y