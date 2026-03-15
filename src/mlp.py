from torch import nn

# SwiGLU with llama3 style
class MLP(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int):
        """
        Llama 3 风格的 SwiGLU MLP。

        Args:
            d_model (int): 模型维度/模型宽度
            intermediate_dim (int): 中间隐藏层维度 
        Return:
            torch.Tensor: 经过 MLP 变换后的输出张量，形状为 (batch_size, seq_len, d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.intermediate_dim = intermediate_dim 
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        """
        MLP层的前向过程

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: MLP前向后的张量
        """
        h1 = self.act_fn(self.gate_proj(x))
        h2 = self.up_proj(x)
        h = self.down_proj(h1 * h2)
        return h
        