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


def generate(cfg: DictConfig, prompts: list[str]):
    tokenizer = Tokenizer()
    model = GPT(cfg)
    eos_id = cfg.eot_token
    temperature = cfg.temperature
    max_new_tokens = cfg.max_new_tokens
    model.load_state_dict(torch.load(cfg.best_model_path, map_location=cfg.device))
    model.eval()
    with torch.no_grad():
        # prompts = ["Once upon a time", "One day", "In a faraway land"]
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
        
        return_texts = []
        for i, toks in enumerate(tokens.tolist()):
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            text = tokenizer.decode(toks)
            return_texts.append(text)
        return return_texts

@hydra.main(config_path="configs", config_name="config", version_base=None)
def chat(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("🖋️  GPT Story Generator")
    print("=" * 60)
    
    # 终端交互选择模式
    while True:
        print("\n请选择运行模式:")
        print("1. 批量生成 (Batch Mode) - 使用经典童话开头自动生成故事")
        print("2. 交互式创作 (Interactive Mode) - 输入自定义开头续写故事")
        choice = input("请输入选项 (1 或 2): ").strip()
        
        if choice == "1":
            mode = "batch"
            break
        elif choice == "2":
            mode = "interactive"
            break
        else:
            print("❌ 无效输入，请重新选择")
    
    # Mode 1: 批量生成童话故事
    if mode == "batch":
        # 经典童话故事开头的prompts
        prompts = [
            "Once upon a time, in a dark enchanted forest,",
            "Long long ago, there lived a kind-hearted dragon who",
            "In a kingdom far far away, a little princess discovered",
            "Once upon a midnight dreary, while the little owl pondered,",
            "In a cozy burrow beneath the roots of an ancient oak tree,"
        ]
        
        print(f"\n🌗Running in batch mode with {len(prompts)} fairy tale prompts...🌓")
        print("=" * 60)
        
        texts = generate(cfg, prompts)
        
        for i, (prompt, text) in enumerate(zip(prompts, texts)):
            print(f"\n📖 Story {i+1}")
            print(f"prompt: {prompt}")
            print(f"generated: {text.strip()}")
            print("-" * 60)
    
    # Mode 2: 交互式创作
    else:
        print("\n" + "=" * 60)
        print("Welcome to Fairy Tale Generator!")
        print("输入童话开头（如：很久很久以前...），AI将帮你续写故事。")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue

                print("生成中...")
                responses = generate(cfg, [user_input])
                
                if responses and len(responses) > 0:
                    reply = responses[0].strip()
                    
                    print(f"AI: {reply}\n")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error occurred: {e}")
                import traceback
                traceback.print_exc()
            break


if __name__ == "__main__":
    chat()