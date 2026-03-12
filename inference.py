import torch
from src import Tokenizer
from src import GPT
from omegaconf import DictConfig
import hydra


def sample(logits: torch.Tensor, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5) #防止除以0
    probs: torch.Tensor = torch.softmax(logits, dim=-1)  # [batch, vocab_size]
    # 标准 multinomial sampling (不可导，仅用于推理)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def generate(cfg: DictConfig):
    tokenizer = Tokenizer()
    
    model = GPT(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads, 
        num_kv_heads=cfg.num_kv_heads,
        intermediate_dim=cfg.intermediate_dim, 
        num_layers=cfg.num_layers, 
        base=cfg.base, 
        dropout=cfg.dropout,
        device=cfg.device
    )
    eos_id = cfg.eot_token
    temperature = cfg.temperature
    max_new_tokens = cfg.max_new_tokens
    model.load_state_dict(torch.load(cfg.best_model_path, map_location=cfg.device))
    model.eval()
    with torch.no_grad():
        # prompts = ["Once upon a time", "One day", "In a faraway land"]
        prompts = cfg.prompts
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
        
        for i, toks in enumerate(tokens.tolist()):
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            text = tokenizer.decode(toks)
            print(f"Prompt: {prompts[i]}\nGenerated: {text}\n\n{'='*50}\n")

    
if __name__ == "__main__":
    generate()