import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from src import Tokenizer
from typing import Dict

class TinyStoriesDataset(Dataset):
    """基础 Parquet Dataset"""
    def __init__(self, parquet_path: str = None, tokenizer: Tokenizer = None, 
        max_length:int = 1024):
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 读取 Parquet
        table = pq.read_table(parquet_path, columns=["text"])
        self.texts = table.column("text").to_pylist()
        print(f"Loaded {len(self.texts)} stories from {parquet_path}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoded_info: Dict[str, torch.Tensor] = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True, 
        )
        """
        Returns:
        dict: {
            “input_ids”: tensor [batch_size, max_length],
            “attention_mask”: tensor [batch_size, max_length],
            “lengths”: tensor [batch_size] 
        }
        """
        print(f"Encoded ids shape: {encoded_info["input_ids"].shape}")
        return encoded_info

def get_story_dataloader(parquet_path="datasets/tinystories_train.parquet", tokenizer=None, batch_size=8, max_length=1024, shuffle=True, num_workers=4):
    dataset = TinyStoriesDataset(parquet_path, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# 使用示例
if __name__ == "__main__":
    
    # 打印PYTHON_PATH
    import os
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    
    # 初始化 tokenizer
    tokenizer = Tokenizer()
    
    # 创建 Dataset
    dataset = TinyStoriesDataset(
        "datasets/tinystories_train.parquet",
        tokenizer,
        max_length=1000  # TinyStories 故事短，512 足够
    )
    
    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,      # 多进程加载
        pin_memory=True,    # 加速 GPU 传输
    )
    
    # 测试
    batch = next(iter(loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Sample: {tokenizer.decode(batch['input_ids'][0][:50])}")