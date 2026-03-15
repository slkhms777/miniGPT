import torch
from src import Tokenizer
from src import GPT
from utils import get_story_dataloader
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import os

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # 初始化 tokenizer
    tokenizer = Tokenizer()
    # 创建 DataLoader
    dataloader = get_story_dataloader(
        "datasets/tinystories_train.parquet",
        tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_seq_len,
        shuffle=True,
        num_workers=4
    )
    # 创建模型
    model = GPT(cfg)
    # 创建优化器
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    for epoch in range(cfg.num_epochs):
        model.train()
        tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for b_i, batch in enumerate(tqdm_bar):
            input_ids = batch["input_ids"].to(cfg.device) # [batch_size, max_seq_len]
            
            # 前向传播
            inputs = input_ids[:, :-1]  # 输入序列，去掉最后一个 token # [batch_size, seq_len]
            labels = input_ids[:, 1:]   # 目标序列，去掉第一个 token # [batch_size, seq_len]
            logits = model(inputs)
            
            # 计算损失
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
                labels.reshape(-1)  # [batch_size * seq_len]
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()
            
            if b_i % cfg.log_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                with open(f"checkpoints/epoch_{epoch}_log.txt", "a") as f:
                    f.write(f"Batch {b_i}, Loss: {loss.item():.6f}\n")
            
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")
    
if __name__ == "__main__":
    train()