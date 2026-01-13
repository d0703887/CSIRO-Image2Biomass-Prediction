import torch.nn as nn

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            dropout: float = 0.4,
            mode: str = "biomass"
    ):
        super().__init__()
        self.mode = mode
        if mode == "biomass":
            self.final_act = nn.Softplus(beta=5)
        elif mode == "gate":
            self.final_act = nn.Sigmoid()
        elif mode == "height":
            self.final_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize the final linear layer to stabilize training
        if mode == "biomass":
            nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-5)
            nn.init.constant_(self.mlp[-1].bias, -0.6)

    def forward(self, x):
        logits = self.mlp(x)
        if self.mode == "gate":
            return self.final_act(logits)
        else:
            return self.final_act(logits)