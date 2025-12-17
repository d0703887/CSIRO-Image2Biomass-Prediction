import torch.nn as nn
import torch
from transformers import AutoModel
# from transformers.models.dinov3_vit import DINOv3ViTModel, DINOv3ViTConfig
# import os, json

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            dropout: float = 0.2,
            mode: str = "biomass"
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus() if mode == "biomass" else nn.Sigmoid()
        )

        # Initialize the final linear layer to stabilize training
        if mode == "biomass":
            nn.init.normal_(self.mlp[-2].weight, mean=0.0, std=1e-5)
            nn.init.constant_(self.mlp[-2].bias, -5.0)

    def forward(self, x):
        return self.mlp(x)

class DinoV3BackBone(nn.Module):
    def __init__(
            self,
            model_name: str,
            hidden_dim: int,
            freeze_backbone: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # DinoV3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.backbone_embed_dim = self.backbone.config.hidden_size
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Regression/Classification head
        make_head = lambda mode: MLP(self.backbone_embed_dim, hidden_dim, mode=mode)
        self.green_mlp = make_head("biomass")
        self.clover_mlp = make_head("biomass")
        self.dead_mlp = make_head("biomass")

        # Gate MLP to prevent noise-cumulation
        self.green_gate = make_head("gate")
        self.clover_gate = make_head("gate")
        self.dead_gate = make_head("gate")



    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def aggregate_tile(self, tile_data):
        _, num_patch, _ = tile_data.shape
        tile_data = tile_data.view(-1, 2, num_patch)
        agg = tile_data.sum(dim=(1, 2))
        return agg

    def forward(self, x, return_patch_preds=False):
        """

        :param x: shape (B * 2, C, H, W)
        :return: preds
        """
        vit_feature = self.backbone(x) # (B * 2, embed_dim)
        patch_feature = vit_feature.last_hidden_state[:, 5:, :] # (B * 2, num_patch, embed_dim)

        # (B * 2, num_patch, 1)
        tile_green= self.green_mlp(patch_feature)
        tile_clover = self.clover_mlp(patch_feature)
        tile_dead = self.dead_mlp(patch_feature)

        # Gating
        tile_green_gate = self.green_gate(patch_feature)
        tile_clover_gate = self.clover_gate(patch_feature)
        tile_dead_gate = self.dead_gate(patch_feature)
        tile_green = tile_green * tile_green_gate
        tile_clover = tile_clover * tile_clover_gate
        tile_dead = tile_dead * tile_dead_gate

        tile_total = tile_green + tile_clover + tile_dead
        tile_gdm = tile_green + tile_clover

        # Aggregation
        green_g = self.aggregate_tile(tile_green)
        clover_g = self.aggregate_tile(tile_clover)
        dead_g = self.aggregate_tile(tile_dead)
        total_g = self.aggregate_tile(tile_total)
        gdm_g = self.aggregate_tile(tile_gdm)

        pred_dict = {
            "Dry_Green_g": green_g,
            "Dry_Clover_g": clover_g,
            "Dry_Dead_g": dead_g,
            "Dry_Total_g": total_g,
            "GDM_g": gdm_g,
        }

        if return_patch_preds:
            pred_dict.update({
                "Tile_Dry_Green_g": tile_green,
                "Tile_Dry_Clover_g": tile_clover,
                "Tile_Dry_Dead_g": tile_dead
            })

        return pred_dict

