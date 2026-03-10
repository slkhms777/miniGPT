from numpy.random.mtrand import f
from torch import nn
import torch
from src import TextEmbedding
from src import DecoderLayer
import hydra
from omegaconf import DictConfig
from src import precompute_freq_cis

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_kv_heads, 
        intermediate_dim, num_layers, base, dropout=0.1, device="cpu"):
        
        super().__init__()
        
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
    
    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        """
        Args: 
            x, Shape [batch_size, seq_len]
        Return:
            y, Shape [batch_size, seq_len + 1]
            new_kv: (new_k, new_v) 更新后的缓存，训练时为None
        """
        
        h = self.embedding(x) # [batch_size, seq_len, d_model]
        b, seq_len = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device).triu_(1)

        for layer in self.blocks:
            h = layer(h, start_pose=0, freqs_cis=freqs_cis, mask=mask) # 训练时不使用缓存和掩码
        
        logits = self.lm_head(self.ln_final(h))
        
        y = torch.softmax(logits, dim=-1)
        
        return y
        
@hydra.main(config_path="configs", config_name="gpt", version_base=None)
def main(cfg: DictConfig):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    gpt = GPT(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads, 
        num_kv_heads=cfg.num_kv_heads,
        intermediate_dim=cfg.intermediate_dim, 
        num_layers=cfg.num_layers, 
        base=cfg.base, 
        dropout=cfg.dropout,
        device=device
    ).to(device)
    gpt.eval()
    
    x = torch.randint(0, 100000, (4,128)).to(device)
    y = gpt(x)
    print(x.shape) # torch.Size([4, 128])
    print(y.shape) # torch.Size([4, 128, 100277])
    
if __name__ == "__main__":
    main()