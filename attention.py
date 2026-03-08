from torch import nn
import math
import torch

class  CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"模型维度 {d_model} 不能被注意力头数 {num_heads} 整除"
        
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.atten_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 因果掩码缓存（用于加速，避免每次都生成）
        self.register_buffer("causal_mask", None)
    
    def apply_rope(self, x, cos, sin):
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            cos/sin: [1, 1, seq_len, head_dim//2] （来自 RoPE 类）
        Return:
            y: [batch, num_heads, seq_len, head_dim]
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin # [batch, num_heads, seq_len, head_dim//2]
        y2 = x1 * sin + x2 * cos
        y = torch.stack(y1, y2, dim=-1).flatten(-2) # 
        return y
        
    def forward(self, x: torch.Tensor, rope=None):
        """
        Args:
            x : 输入张量，Shape:[batch, seq_len, dim]
            rope : 旋转位置编码. cos, sin: [1, seq_len, 1, dim/2]
        """
        B,T,_ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2) # [B, H, T, dim]
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2) # [B, H, T, dim]
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2) # [B, H, T, dim]
        
        # 旋转位置编码
        if rope is not None:
            cos, sin = rope
            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)
        
        # 注意力计算
        atten_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, T, T]
        
        # 下三角掩码
        if self.causal_mask is None or self.causal_mask.size(-1) < T:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            self.register_buffer("causal_mask", mask.bool())
        # 使用缓存的mask（如果T小于之前缓存的长度）
        atten_scores = atten_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
         
        atten_weights = torch.softmax(atten_scores, dim=-1) # [B, H, T, T]
        atten_weights = self.atten_dropout(atten_weights) 
        
        out = atten_weights @ v # [B, H, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        
        return out