import torch.nn as nn
import torch
from transformers import AutoModel


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            dropout: float = 0.1,
            classification: bool = False
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() if classification else nn.Softplus()
        )

    def forward(self, x):
        return self.mlp(x)

class DinoV3BackBone(nn.Module):
    def __init__(
            self,
            model_name: str,
            hidden_dim: int,
            predict_total: bool = True,
            predict_gdm: bool = False,
            predict_has_clover: bool = True,
            predict_height: bool = True,
            freeze_backbone: bool = True
    ):
        super().__init__()
        self.predict_total = predict_total
        self.predict_gdm = predict_gdm
        self.predict_has_clover = predict_has_clover
        self.predict_height = predict_height

        # DinoV3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.backbone_embed_dim = self.backbone.config.hidden_size
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Regression/Classification head
        make_head = lambda classification: MLP(self.backbone_embed_dim, hidden_dim, classification=classification)
        self.green_mlp = make_head(False)
        self.clover_mlp = make_head(False)
        self.dead_mlp = make_head(False)
        self.total_mlp = make_head(False) if predict_total else None
        self.gdm_mlp = make_head(False) if predict_gdm else None
        self.height_mlp = make_head(False) if predict_height else None
        self.has_clover_mlp = make_head(True) if predict_has_clover else None

    def sum_tile(self, tile_preds, b):
        pairs = tile_preds.view(b, 2)
        real_value = torch.exp(pairs) - 1
        total_real_value = real_value.sum(dim=1)
        return torch.log(1 + total_real_value)

    def forward(self, x):
        """

        :param x: shape (B * 2, C, H, W)
        :return: preds
        """
        b = x.shape[0] // 2
        img_feature = self.backbone(x).pooler_output # (B * 2, embed_dim)

        tile_green= self.green_mlp(img_feature)
        tile_clover = self.clover_mlp(img_feature)
        tile_dead = self.dead_mlp(img_feature)

        if self.predict_total:
            tile_total = self.total_mlp(img_feature)
        else:
            # sum log-transformed value
            real_green = torch.exp(tile_green) - 1
            real_clover = torch.exp(tile_clover) - 1
            real_dead = torch.exp(tile_dead) - 1
            tile_total = torch.log(1 + real_green + real_clover + real_dead)

        if self.predict_gdm:
            tile_gdm = self.gdm_mlp(img_feature)
        else:
            # sum log-transformed value
            real_green = torch.exp(tile_green) - 1
            real_clover = torch.exp(tile_clover) - 1
            tile_gdm = torch.log(1 + real_green + real_clover)

        tile_height = self.height_mlp(img_feature) if self.predict_height else None
        tile_has_clover = self.has_clover_mlp(img_feature) if self.predict_has_clover else None

        # Aggregation
        green_g = self.sum_tile(tile_green, b)
        clover_g = self.sum_tile(tile_clover, b)
        dead_g = self.sum_tile(tile_dead, b)
        total_g = self.sum_tile(tile_total, b)
        gdm_g = self.sum_tile(tile_gdm, b)

        if self.predict_height:
            real_height = torch.exp(tile_height)
            height = torch.log(real_height.view(b, 2).mean(dim=1))
        else:
            height = None

        has_clover = torch.max(tile_has_clover.view(b, 2), dim=1).values if self.predict_has_clover else None

        return {
            "green_g": green_g,
            "clover_g": clover_g,
            "dead_g": dead_g,
            "total_g": total_g,
            "gdm_g": gdm_g,
            "height": height,
            "has_clover": has_clover
        }


