from torch import nn

# SwiGLU from llama3
class MLP(nn.Module):
    def __init__(self, d_model, intermediate_dim):
        """
        intermediate_dim ≈ 8/3 * d_model
        """
        super().__init__()
        self.d_model = d_model
        self.intermediate_dim = intermediate_dim 
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        h1 = self.act_fn(self.gate_proj(x))
        h2 = self.up_proj(x)
        h = self.down_proj(h1 * h2)
        return h
        