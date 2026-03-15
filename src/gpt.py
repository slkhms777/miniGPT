from torch import nn
import torch
from .emb import TextEmbedding
from .decoder import DecoderLayer
from .rope import precompute_freq_cis
from omegaconf import DictConfig

class GPT(nn.Module):
    """
    基于 Transformer Decoder-only 架构的 GPT/LLM 模型实现。
    
    Attributes:
        embedding (TextEmbedding): 词嵌入层（包含词嵌入 + 位置编码）
        blocks (nn.ModuleList): Decoder 层列表，每层包含自注意力和 FFN
        ln_final (nn.LayerNorm): 最终的 Layer Normalization
        lm_head (nn.Linear): 语言模型头，将隐藏状态映射到词表空间（无偏置）
        freqs_cis (torch.Tensor): 预计算的 RoPE 旋转频率复数 [max_seq_len, head_dim//2]
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = cfg.device
        
        self.embedding = TextEmbedding(
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model
        ).to(self.device)

        self.blocks = nn.ModuleList([
            DecoderLayer(cfg)
            for _ in range(cfg.num_layers)
        ]).to(self.device)
        self.ln_final = nn.LayerNorm(cfg.d_model).to(cfg.device)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False).to(cfg.device)
        self.register_buffer("freqs_cis", precompute_freq_cis(cfg.d_model//cfg.num_heads, cfg.base, cfg.max_seq_len), persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        GPT/LLM 的前向过程
        
        Args: 
            x (torch.Tensor): Shape [batch_size, seq_len]
            start_pos (int): rope的起始pos
        Return:
            torch.Tensor: logits, Shape [batch_size, seq_len, vocab_size] 不进行softmax，便于cross-entropy loss计算
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
        