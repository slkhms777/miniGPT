from typing import Union
import torch
from torch import nn

class TextEmbedding(nn.Module):
    """Token 嵌入层，将 token ID 映射为向量。"""
    def __init__(self, vocab_size: int = 50257, max_seq_len: int = 128, d_model: int = 64):
        """
        Args:
            vocab_size (int): 词表大小，50257 (GPT-2)
            max_seq_len (int): 最大上下文长度
            d_model (int): 模型隐藏层大小
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
    
    def forward(self, tokens: Union[torch.Tensor, list[int]]) -> torch.Tensor:
        """
        Args:
            tokens (Union[torch.Tensor, list[int]]): 输入的tokens. Shape:[batch_size, seq_len]
        Return:
            torch.Tensor : 返回词嵌入和位置编码嵌入后的张量. Shapex:[batch_size, seq_len, d_model] 嵌入向量
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        B, T = tokens.size()
        assert T <= self.max_seq_len
        return self.token_emb(tokens)
        
        
if __name__ == "__main__":
    text = "hello world!"
    from tokenizer import Tokenizer
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    print(tokens)
    tokens = [tokens, tokens]
    text_embedding = TextEmbedding()
    text_embed = text_embedding(tokens)
    print(text_embed.shape)
    print(text_embed)
