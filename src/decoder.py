import torch
from torch import nn
from .attention import GQA
from .mlp import MLP
from .norm import RMSNorm
from .rope import precompute_freq_cis

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_dim, 
        num_kv_heads, max_cache_batch, max_seq_len, dropout=0.1):
            
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQA(d_model, num_heads, num_kv_heads, max_cache_batch, max_seq_len, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MLP(d_model, intermediate_dim)
    
    def forward(self, x, start_pos, freqs_cis, mask):
        # Pre-LN 结构
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