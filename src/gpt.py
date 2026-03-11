from torch import nn
import torch
from .emb import TextEmbedding
from .decoder import DecoderLayer
from .rope import precompute_freq_cis

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_kv_heads, 
        intermediate_dim, num_layers, base, dropout=0.1, device="cpu"):
        
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.embedding = TextEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model
        ).to(device)

        self.blocks = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                num_kv_heads=num_kv_heads,
                max_cache_batch=32, # 这里假设最大batch size为32，可以根据实际情况调整
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]).to(device)
        self.ln_final = nn.LayerNorm(d_model).to(device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device)
        self.register_buffer("freqs_cis", precompute_freq_cis(d_model//num_heads, base, max_seq_len), persistent=False)
    
    def forward(self, x, start_pos=0):
        """
        Args: 
            x, Shape [batch_size, seq_len]
        Return:
            logits, Shape [batch_size, seq_len, vocab_size] 不进行softmax，便于cross-entropy loss计算
        """
        
        h = self.embedding(x) # [batch_size, seq_len, d_model]
        b, seq_len = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len].to(h.device)
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device).triu_(1)

        for layer in self.blocks:
            h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask) 
        # h = self.ln_final(h)[:, -1]  # 只取最后一个位置的输出进行预测，Shape [batch_size, d_model]
        logits = self.lm_head(h) # [batch_size, vocab_size]
        
        return logits
        