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
            tile_total = tile_green + tile_clover + tile_dead

        if self.predict_gdm:
            tile_gdm = self.gdm_mlp(img_feature)
        else:
            tile_gdm = tile_green + tile_clover

        tile_height = self.height_mlp(img_feature) if self.predict_height else None
        tile_has_clover = self.has_clover_mlp(img_feature) if self.predict_has_clover else None

        # Aggregation
        green_g = tile_green.view(b, 2).sum(dim=1)
        clover_g = tile_clover.view(b, 2).sum(dim=1)
        dead_g = tile_dead.view(b, 2).sum(dim=1)
        total_g = tile_total.view(b, 2).sum(dim=1)
        gdm_g = tile_gdm.view(b, 2).sum(dim=1)
        height = tile_height.view(b, 2).mean(dim=1) if self.predict_height else None
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


