import torch
from torch import nn
from .attention import GQA
from .mlp import MLP
from .norm import RMSNorm
from .rope import precompute_freq_cis
from omegaconf import DictConfig
from typing import Optional
class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层，使用 Pre-LN 结构。

    Args:
        cfg: 模型参数配置。

    Attributes:
        norm1: 注意力前的 RMSNorm。
        attn: 分组查询注意力（GQA）。
        norm2: FFN 前的 RMSNorm。
        ffn: SwiGLU MLP。
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQA(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = MLP(cfg.d_model, cfg.intermediate_dim)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Transformer Decoder Layer 前向
        
        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, seq_len, d_model)。
            start_pos (int): 推理时的起始位置（用于 KV Cache）。
            freqs_cis (torch.Tensor): 旋转频率的复数。
            mask (Optional[torch.Tensor]): 因果掩码。

        Returns:
            torch.Tensor: 输出张量，形状 (batch_size, seq_len, d_model)。
        """
        x = x + self.attn(self.norm1(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
        return x
        
if __name__ == "__main__":
    # 测试DecoderLayer
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 4
    intermediate_dim = 16
    num_kv_heads = 2
    max_cache_batch = 4
    max_seq_len = 8
    
    decoder_layer = DecoderLayer(d_model, num_heads, intermediate_dim, num_kv_heads, max_cache_batch, max_seq_len)
    
    x = torch.randn(batch_size, seq_len, d_model)
    start_pos = 0
    freqs_cis = precompute_freq_cis(hidden_dim=d_model//num_heads, base=50000.0, max_seq_len=max_seq_len)
    mask = None
    out = decoder_layer(x, start_pos, freqs_cis[:seq_len], mask)
    print("Output shape:", out.shape)