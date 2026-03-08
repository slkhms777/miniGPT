from typing import Union
import torch
from torch import nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size=50257, max_context_len=128, d_model=64):
        """
        Args:
            vocab_size: 词表大小，50257 (GPT-2)
            max_context_len: 最大上下文长度
            d_model: 模型隐藏层大小
        """
        super().__init__()
        self.max_context_len = max_context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
    
    def forward(self, tokens: Union[torch.Tensor, list[int]]) -> torch.Tensor:
        """
        Args:
            tokens : 输入的tokens. Shape:[batch_size, seq_len]
        Return:
            x : 返回词嵌入和位置编码嵌入后的张量. Shapex:[batch_size, seq_len, d_model] 嵌入向量
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens)
        B, T = tokens.size()
        assert T <= self.max_context_len, f"输入序列长度{T}超过最大长度{self.max_context_len}"
        
        x = self.token_emb(tokens)
        return x
        
        
if __name__ == "__main__":
    text = "hello world!"
    from tokenizer import Tokenizer
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    print(tokens)
    tokens = [tokens]
    text_embedding = TextEmbedding()
    text_embed = text_embedding(tokens)
    print(text_embed)
