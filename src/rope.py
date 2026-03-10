import torch
from typing import Tuple

def precompute_freq_cis(hidden_dim, base, max_seq_len):
    # 预计算角度: θ_i = base^(-2i/hidden_dim) for i in [0, hidden_dim / 2)
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
    t = torch.arange(max_seq_len)
    # 外积
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, hidden_dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor) : [batch_size, seq_len, num_heads, hidden_dim]
        k (torch.Tensor) : [batch_size, seq_len, num_kv_heads, hidden_dim]
        freq_cis (torch.Tensor) : [seq_len, hidden_dim // 2]
    Returns:
        Tuple[torch.Tensor, torch.Tensor]
    """
    q = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2)) # [batch_size, seq_len, heads, hidden_dim // 2]
    k = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(1) # [seq_len, 1, hidden_dim // 2]
    
    q_embed = torch.view_as_real(q * freqs_cis).flatten(-2) # 广播出一个batch维度 [batch_size, seq_len, heads, hidden_dim]
    k_embed = torch.view_as_real(k * freqs_cis).flatten(-2)
    
    return q_embed, k_embed
    