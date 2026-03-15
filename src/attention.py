from torch import nn
import torch
from typing import Optional
from .rope import apply_rope, precompute_freq_cis

def repeat_kv(x: torch.Tensor, n_rep: int):
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, hidden_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, hidden_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, hidden_dim)

class GQA(nn.Module):
    """
    Group Query Attention (GQA) Layer.

    Attributes:
        d_model (int): 模型维度/宽度
        head_dim (int): 每个注意力头的维度
        num_heads (int): q头数
        num_kv_heads (int): kv头数
        dropout (float): 丢弃率
        max_cache_batch (int): 最大缓存batch
        max_seq_len (int): 最大上下文长度
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.num_heads
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.dropout = cfg.dropout
        self.max_cache_batch = cfg.max_cache_batch
        self.max_seq_len = cfg.max_seq_len
        
        assert self.d_model % self.num_heads == 0, f"模型维度 {self.d_model} 不能被注意力头数 {self.num_heads} 整除"
        assert self.num_heads % self.num_kv_heads == 0, f"Q 的头数 {self.num_heads} 不能被 KV 头数 {self.num_kv_heads} 整除"
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        
        self.n_rep = self.num_heads // self.num_kv_heads
        self.softmax_scale = self.head_dim ** -0.5
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # 延迟初始化：只保存 shape，不分配显存
        self._cache_shape = (self.max_cache_batch, self.max_seq_len, self.num_kv_heads, self.head_dim)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        
    def _ensure_cache_initialized(self, device, dtype):
        """延迟初始化 KV Cache，仅在第一次推理时调用"""
        if self.k_cache is None:
            self.k_cache = torch.zeros(self._cache_shape, dtype=dtype, device=device)
            self.v_cache = torch.zeros(self._cache_shape, dtype=dtype, device=device)
                
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        GQA 前向过程
        
        Args:
            x : 输入张量，Shape:[batch, seq_len, dim]
            start_pos : 当前输入序列的起始位置，同时也是cache中写入的起始位置
            freqs_cis : 预计算的旋转位置编码，Shape:[max_seq_len, head_dim//2]
            mask : 掩码张量，Shape:[1, 1, seq_len, total_seq_len]，如-inf的上三角矩阵，用于遮挡未来位置
        Returns:
            torch.Tensor : 输出张量，Shape与输入相同
        """
        B, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        
        # 投影到QKV空间
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分头 reshape
        q = q.reshape(B, seq_len, self.num_heads, self.head_dim) # [B, seq_len, H, dim]
        k = k.reshape(B, seq_len, self.num_kv_heads, self.head_dim) # [B, seq_len, kvH, dim]
        v = v.reshape(B, seq_len, self.num_kv_heads, self.head_dim) # [B, seq_len, kvH, dim]
        
        # 旋转位置编码
        q, k = apply_rope(q, k, freqs_cis)
        if not self.training:
            self._ensure_cache_initialized(x.device, x.dtype)
            # 写入cache
            self.k_cache[:B, start_pos:end_pos] = k
            self.v_cache[:B, start_pos:end_pos] = v
            
            # 读取缓存
            k = self.k_cache[:B, :end_pos] # [B, T, kvH, dim]
            v = self.v_cache[:B, :end_pos] # [B, T, kvH, dim]
            
            
        
        # 把头数的维度放到seq_len维之前
        k = k.transpose(1, 2) # [B, num_kv_heads, T, hidden_dim]
        v = v.transpose(1, 2)
        q = q.transpose(1, 2) # [B, num_heads, seq_len, hidden_dim], seq_len可能是1
        
        # GQA：重复KV匹配Q的头数
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep) # [B, num_heads, T, hidden_dim]
        
        # 注意力计算
        attn_scores = (q @ k.transpose(-2, -1)) * self.softmax_scale # [B, H, seq_len, T]
        
        # 掩码
        if mask is not None:
            attn_scores = attn_scores + mask  # mask包含-inf的上三角
        
        attn_weights = torch.softmax(attn_scores, dim=-1) # [B, H, seq_len, T]
        attn_weights = self.attn_dropout(attn_weights) 
        
        # out = attn_weights @ v   # [B, H, seq_len, head_dim]
        out = torch.einsum('bhst,bhtd->bhsd', attn_weights, v)
        out = out.transpose(1, 2).reshape(B, seq_len, self.d_model)
        
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        
        return out
        
if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    attn = GQA(d_model=64, num_heads=8, num_kv_heads=4, max_cache_batch=2, max_seq_len=8)
    hidden_dim = attn.head_dim
    freqs_cis = precompute_freq_cis(hidden_dim=hidden_dim, base=50000.0, max_seq_len=8) # [max_seq_len, head_dim]
    print("freqs_cis", freqs_cis)
    out = attn(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
    print(out.shape)