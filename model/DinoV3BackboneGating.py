import torch.nn as nn
import torch
from transformers import AutoModel
from model.MLP import MLP


class DinoV3BackboneGating(nn.Module):
    def __init__(
            self,
            model_name: str,
            hidden_dim: int,
            freeze_backbone: bool = True,
            predict_height: bool = False,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.predict_height = predict_height

        # DinoV3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.embed_dim = self.backbone.config.hidden_size
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        make_head = lambda mode: MLP(self.embed_dim, hidden_dim, mode=mode)

        # Biomass head
        self.green_mlp = make_head("biomass")
        self.clover_mlp = make_head("biomass")
        self.dead_mlp = make_head("biomass")

        # Gate MLP to prevent noise-cumulation
        self.green_gate = make_head("gate")
        self.clover_gate = make_head("gate")
        self.dead_gate = make_head("gate")

        # Auxiliary height prediction
        if self.predict_height:
            self.height_mlp = make_head("height")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def aggregate_biomass(self, biomass, mode="tiled"):
        if mode == "tiled":
            _, num_patch, _ = biomass.shape
            biomass = biomass.view(-1, 2, num_patch)
        agg = biomass.sum(dim=(1, 2))
        return agg

    def aggregate_height(self, patch_height, weight, mode="tiled"):
        if mode == "tiled":
            _, num_patch, _ = patch_height.shape
            patch_height = patch_height.view(-1, 2, num_patch)
            weight = weight.view(-1, 2, num_patch)

        weighted_sum = torch.sum(patch_height * weight, dim=(1, 2))
        weight_sum = torch.sum(weight, dim=(1, 2))
        return weighted_sum / (weight_sum + 1e-6)

    def forward(self, x, mode="tiled", return_patch_preds=False, return_gates=False):
        """

        :param x: shape (B, C, H, W)
        :return: preds
        """
        vit_feature = self.backbone(x)
        patch_feature = vit_feature.last_hidden_state[:, 5:, :] # (B * 2, num_patch, embed_dim)

        # Biomass prediction
        raw_green= self.green_mlp(patch_feature)
        raw_clover = self.clover_mlp(patch_feature)
        raw_dead = self.dead_mlp(patch_feature)

        # Gates
        green_gate = self.green_gate(patch_feature)
        clover_gate = self.clover_gate(patch_feature)
        dead_gate = self.dead_gate(patch_feature)

        # Apply gates
        patch_green = raw_green * green_gate
        patch_clover = raw_clover * clover_gate
        patch_dead = raw_dead * dead_gate

        # Aggregation
        pred_green = self.aggregate_biomass(patch_green, mode)
        pred_clover = self.aggregate_biomass(patch_clover, mode)
        pred_dead = self.aggregate_biomass(patch_dead, mode)

        pred_dict = {
            "Dry_Green_g": pred_green,
            "Dry_Clover_g": pred_clover,
            "Dry_Dead_g": pred_dead,
            "Dry_Total_g": pred_green + pred_clover + pred_dead,
            "GDM_g": pred_green + pred_clover
        }

        if self.predict_height:
            weight = (green_gate + clover_gate + dead_gate).clamp(max=1.0)
            patch_height = self.height_mlp(patch_feature)
            pred_dict["Avg_Height"] = self.aggregate_height(patch_height, weight, mode)


        # TODO: do not return tiled value
        if return_gates:
            pred_dict.update({
                "Tile_Gate_Dry_Green_g": green_gate,
                "Tile_Gate_Dry_Clover_g": clover_gate,
                "Tile_Gate_Dry_Dead_g": dead_gate
            })

        if return_patch_preds:
            pred_dict.update({
                "Tile_Dry_Green_g": patch_green,
                "Tile_Dry_Clover_g": patch_clover,
                "Tile_Dry_Dead_g": patch_dead
            })

        return pred_dict


