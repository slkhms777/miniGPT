import torch
from torch import nn

class RoPE(nn.Module):
    """旋转位置编码"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: 每个头的维度
            max_seq_len: 最大序列长度
            base: 旋转基数, 控制频率衰减速度
        """
        super().__init__()
        self.dim = dim
        
        # 预计算角度: θ_i = base^(-2i/dim) for i in [0, dim / 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        t = torch.arange(max_seq_len)
        
        # 外积
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim/2]
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.tensor, torch.tensor]:
        """
        返回指定长度的 cos 和 sin，用于旋转
        Args:
            x: 输入张量，用于获取 device
            seq_len: 当前序列长度
        Returns:
            cos, sin: [1, 1, seq_len, dim/2]
        """
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsequeeze(1).to(x.device),
            self.sin_cached[:seq_len].unsqueeze(0).unsequeeze(1).to(x.device),            
        )