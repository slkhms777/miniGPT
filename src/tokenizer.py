import torch
import tiktoken

class Tokenizer():
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab # 50257
        self.eot_token = self.enc.eot_token  # <|endoftext|> = 50256
        
    def tokenize(self, text:str, allowed_special={"<|endoftext|>"}) -> list[int]:
        """ 
        Args:
            text (int): 输入的句子
            allowed_special (str): 允许使用的特殊token
        Returns:
            token_indices (list[int])
        """
        token_indices = self.enc.encode(text, allowed_special=allowed_special)
        return token_indices
        
    def detokenize(self, tokens:list[int]) -> str:
        """
        支持list或tensor
        
        Args:
            tokens (list[int]): 离散的token indices
        Returns:
            text (str): 输出的句子
        """
        if isinstance(tokens, torch.Tensor):
            tokens.tolist()
        text = self.enc.decode(tokens)
        return text
        
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Hello world!<|endoftext|>")
    print(tokens)
    sentence = tokenizer.detokenize([15496, 995])
    print(sentence)

    