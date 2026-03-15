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
    
    model = GPT(cfg)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True
    )
    # test( tokenizer, model, cfg, max_new_tokens=200)
    # return
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
        texts = test(tokenizer, model, cfg, 500)
        # 保存到checkpoints目录下
        with open(f"checkpoints/epoch_{epoch}_samples.txt", "w") as f:
            for text in texts:
                f.write(text + "\n")
                
def sample(logits: torch.Tensor, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5) #防止除以0
    probs: torch.Tensor = torch.softmax(logits, dim=-1)  # [batch, vocab_size]
    # 标准 multinomial sampling (不可导，仅用于推理)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def test(tokenizer : Tokenizer = None, model: GPT = None, cfg:DictConfig = None, max_new_tokens: int=200, temperature: float = 1.0):
    # 测试模型生成文本
    eos_id = cfg.eot_token
    model.eval()
    with torch.no_grad():
        prompts = ["Once upon a time", "One day", "In a faraway land"]
        prompt_tokens: list[torch.Tensor]= [
            tokenizer.encode(prompt, max_length=None, truncation=False, padding=False)["input_ids"].to(cfg.device)
            for prompt in prompts
        ]
        batch = len(prompt_tokens)
        prompt_lens = [len(t) for t in prompt_tokens]
        max_prompt_len = max(prompt_lens)
        total_len = min(cfg.max_seq_len, max_new_tokens + max_prompt_len)
        tokens = torch.full((batch, total_len), -1, dtype=torch.long, device=cfg.device)
        for i, t in enumerate(prompt_tokens):
            # tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=cfg.device)
            tokens[i, :len(t)] = t
        prev_pos = 0
        finished = torch.tensor([False] * len(prompt_tokens), device=cfg.device)
        prompt_mask = tokens != -1
        for cur_pos in range(min(prompt_lens), total_len):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos) # [batch, cur_pos-prev_pos, vocab_size]
            logits = logits[:, -1, :]   # 只取最后一个位置: [batch, vocab_size]
            if temperature > 0:
                next_token = sample(logits, temperature)  
            else:
                next_token = logits.argmax(dim=-1) 
            # 如果当前位置是prompt的一部分，则使用原始token，否则使用模型生成的token
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token) 
            tokens[:, cur_pos] = next_token
            # 如果生成了eot_token且当前位置不是prompt的一部分，则标记为finished。
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            prev_pos = cur_pos
            if finished.all():
                break
        
        return_list = []
        for i, toks in enumerate(tokens.tolist()):
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            text = tokenizer.decode(toks)
            # print(f"Prompt: {prompts[i]}\nGenerated: {text}\n")
            return_list.append(text)
        return return_list
            
    
if __name__ == "__main__":
    train()