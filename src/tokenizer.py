import torch
import tiktoken
from typing import Dict

class Tokenizer():
    def __init__(self):
        # self.enc = tiktoken.get_encoding("cl100k_base")
        self.enc = tiktoken.get_encoding("gpt2") # 100277
        self.vocab_size = self.enc.n_vocab 
        self.eot_token = self.enc.eot_token  # <|endoftext|> = 50256
        self.pad_token = self.enc.eot_token  # GPT-2 没有专门的 pad token，使用 eot_token 作为填充
    
    def encode(
        self,
        text:str,
        max_length:int = 1025,
        truncation:bool = False,
        padding:bool = False,
        allowed_special={"<|endoftext|>"},
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text: 输入文本
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
            allowed_special: 允许的特殊 token
        
        Returns:
            dict: {
                "input_ids": tensor [max_length],
                "valid_mask": tensor [max_length],
                "length": int (实际长度)
            }
        """
        # tokenize
        tokens = self.enc.encode(text, allowed_special=allowed_special)
        
        # 截断
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        length = len(tokens)
        
        valid_mask = [1] * length
        
        # 填充
        if padding and max_length is not None and  length < max_length:
            tokens += [self.pad_token] * (max_length - length)
            valid_mask += [0] * (max_length - length)
        
        # 转换为 tensor
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.long),
            "length": length
        }
    
    def encode_batch(
        self,
        texts:list[str],
        max_length:int = 1025,
        allowed_special={"<|endoftext|>"},
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts: 输入文本列表
            max_length: 最大长度
            allowed_special: 允许的特殊 token
        
        Returns:
            dict: {
                "input_ids": tensor [batch_size, max_length],
                "valid_mask": tensor [batch_size, max_length],
                "lengths": tensor [batch_size] (截断后的实际长度)
            }
        """
        batch_input_ids = []
        batch_valid_mask = []
        batch_lengths = []
        
        # max_length不能为None，因为批处理时需要统一长度
        assert max_length is not None, "批处理时必须指定 max_length"
        
        truncation = True  # 批处理时默认启用截断
        padding = True  # 批处理时默认启用填充
        # 保证每个返回的序列都被截断或填充到 max_length
        for text in texts:
            encoding = self.encode(text, max_length, truncation, padding, allowed_special)
            batch_input_ids.append(encoding["input_ids"])
            batch_valid_mask.append(encoding["valid_mask"])
            batch_lengths.append(encoding["length"])
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "valid_mask": torch.stack(batch_valid_mask),
            "lengths": torch.tensor(batch_lengths, dtype=torch.long)
        }
    
    def decode(self, tokens:list[int]) -> str:
        """
        Args:
            tokens (list[int]): 离散的token indices
        Returns:
            text (str): 输出的句子
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        text = self.enc.decode(tokens)
        return text
        
    def decode_batch(self, batch_tokens:torch.Tensor) -> list[str]:
        """
        Args:
            batch_tokens (torch.Tensor): [batch_size, seq_len]
        Returns:
            list[str]: 输出的句子列表
        """
        sentences = []
        for tokens in batch_tokens:
            sentence = self.decode(tokens)
            sentences.append(sentence)
        return sentences
        
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokens = tokenizer.encode("Hello world!<|endoftext|>",1000)
    batch = [
        "Hello world!<|endoftext|>",
        "This is a test.<|endoftext|>"
    ]
    batch_tokens = tokenizer.encode_batch(batch, max_length=1000)
    print(tokens)
    print(batch_tokens["input_ids"].shape)
    # sentence = tokenizer.detokenize([15496, 995])
    # print(sentence)

    