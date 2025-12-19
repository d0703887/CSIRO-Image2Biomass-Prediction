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
        if mode == "biomass":
            final_act = nn.Softplus()
        elif mode == "gate":
            final_act = nn.Sigmoid()
        elif mode == "height":
            final_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported MLP mode: {mode}")

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            final_act
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

        # Biomass head
        make_head = lambda mode: MLP(self.backbone_embed_dim, hidden_dim, mode=mode)
        self.green_mlp = make_head("biomass")
        self.clover_mlp = make_head("biomass")
        self.dead_mlp = make_head("biomass")

        # Gate MLP to prevent noise-cumulation
        self.green_gate = make_head("gate")
        self.clover_gate = make_head("gate")
        self.dead_gate = make_head("gate")

        # Auxiliary height prediction
        self.height_mlp = make_head("height")

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

    def forward(self, x, return_patch_preds=False, return_gates=False):
        """

        :param x: shape (B * 2, C, H, W)
        :return: preds
        """
        vit_feature = self.backbone(x) # (B * 2, embed_dim)
        patch_feature = vit_feature.last_hidden_state[:, 5:, :] # (B * 2, num_patch, embed_dim)

        # Biomass prediction
        tile_green= self.green_mlp(patch_feature)
        tile_clover = self.clover_mlp(patch_feature)
        tile_dead = self.dead_mlp(patch_feature)

        # Gates
        tile_green_gate = self.green_gate(patch_feature)
        tile_clover_gate = self.clover_gate(patch_feature)
        tile_dead_gate = self.dead_gate(patch_feature)
        tile_green = tile_green * tile_green_gate
        tile_clover = tile_clover * tile_clover_gate
        tile_dead = tile_dead * tile_dead_gate

        tile_total = tile_green + tile_clover + tile_dead
        tile_gdm = tile_green + tile_clover

        # Height
        b_times_2, num_patch, _ = patch_feature.shape
        height_weight = (tile_green_gate + tile_clover_gate + tile_dead_gate).clamp(max=1.0)
        patch_height = self.height_mlp(patch_feature)

        height_weight = height_weight.view(-1, 2 * num_patch).detach()
        patch_height = patch_height.view(-1, 2 * num_patch)
        weighted_height_sum = torch.sum(height_weight * patch_height, dim=1)
        weight_sum = torch.sum(height_weight, dim=1)
        avg_height = weighted_height_sum / (weight_sum + 1e-6)

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
            "Avg_Height": avg_height
        }

        if return_gates:
            pred_dict.update({
                "Tile_Gate_Dry_Green_g": tile_green_gate,
                "Tile_Gate_Dry_Clover_g": tile_clover_gate,
                "Tile_Gate_Dry_Dead_g": tile_dead_gate
            })

        if return_patch_preds:
            pred_dict.update({
                "Tile_Dry_Green_g": tile_green,
                "Tile_Dry_Clover_g": tile_clover,
                "Tile_Dry_Dead_g": tile_dead
            })

        return pred_dict

