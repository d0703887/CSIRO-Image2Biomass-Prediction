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
            freeze_backbone: bool = True,
            log_transform: bool = False
    ):
        super().__init__()
        self.log_transform = log_transform

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

    def aggregate_tile(self, tile_data):
        _, num_patch, _ = tile_data.shape
        real_data = torch.expm1(tile_data) if self.log_transform else tile_data
        real_data = real_data.view(-1, 2, num_patch)
        agg = real_data.sum(dim=(1, 2))
        if self.log_transform:
            agg = torch.log1p(agg)
        return agg

    def forward(self, x):
        """

        :param x: shape (B * 2, C, H, W)
        :return: preds
        """
        vit_feature = self.backbone(x) # (B * 2, embed_dim)
        # global_feature = vit_feature.pooler_output # (B * 2, embed_dim)
        patch_feature = vit_feature.last_hidden_state[:, 5:, :] # (B * 2, num_patch, embed_dim)

        # (B * 2, num_patch, 1)
        tile_green= self.green_mlp(patch_feature)
        tile_clover = self.clover_mlp(patch_feature)
        tile_dead = self.dead_mlp(patch_feature)

        if self.log_transform:
            # Tile total
            real_tile_green = torch.expm1(tile_green)
            real_tile_clover = torch.expm1(tile_clover)
            real_tile_dead = torch.expm1(tile_dead)
            real_tile_total = real_tile_green + real_tile_clover + real_tile_dead
            tile_total = torch.log1p(real_tile_total)

            # Tile GDM
            real_tile_gdm = real_tile_green + real_tile_clover
            tile_gdm = torch.log1p(real_tile_gdm)
        else:
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
        return pred_dict

