import hydra
from omegaconf import DictConfig
import torch
from src import GPT

@hydra.main(config_path="configs", config_name="gpt", version_base=None)
def main(cfg: DictConfig):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    gpt = GPT(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads, 
        num_kv_heads=cfg.num_kv_heads,
        intermediate_dim=cfg.intermediate_dim, 
        num_layers=cfg.num_layers, 
        base=cfg.base, 
        dropout=cfg.dropout,
        device=device
    ).to(device)
    gpt.eval()
    
    # 输出模型的参数量
    total_params = sum(p.numel() for p in gpt.parameters())
    print(f"模型参数量为 {total_params / 1e6:.1f} M")
    
    x = torch.randint(0, 100000, (4,128)).to(device)
    y = gpt(x)
    print(x.shape) # torch.Size([4, 128])
    print(y.shape) # torch.Size([4, 128, 100277])
    
if __name__ == "__main__":
    main()