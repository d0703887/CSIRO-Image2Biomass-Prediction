import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel
from model.MLP import MLP
import math
from peft import LoraConfig, get_peft_model

class DinoV3BackboneMultiScale(nn.Module):
    def __init__(
            self,
            model_name: str,
            hidden_dim: int,
            predict_height: bool = False,
            training_mode: str = "freeze_backbone"
    ):
        # low resolution: num_patch
        # high resolution: 4 * num_patch

        super().__init__()
        self.predict_height = predict_height
        self.training_mode = training_mode

        # DinoV3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.embed_dim = self.backbone.config.hidden_size

        if self.training_mode == "lora":
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        elif self.training_mode == "freeze_backbone":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        elif self.training_mode != "full_finetune":
            raise ValueError(f"Unsupported Training Mode: {self.training_mode}")

        # fine-grained + coarse-grained feature
        make_head = lambda mode: MLP(self.embed_dim * 2, hidden_dim, mode=mode)

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
        if self.training_mode == "freeze_backbone":
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

    def forward(self, high_res_x, low_res_x, mode="tiled", return_patch_preds=False, return_gates=False):
        high_res_out = self.backbone(high_res_x)
        low_res_out = self.backbone(low_res_x)
        hr_patch_feat = high_res_out.last_hidden_state[:, 5:, :]  # (B * 2, num_patch, embed_dim)
        lr_patch_feat = low_res_out.last_hidden_state[:, 5:, :]   # (B * 2, num_patch // 4, embed_dim)

        # Upsample low-res feature
        b_times_2, l_num_patch, d = lr_patch_feat.shape
        low_H = low_W = int(math.sqrt(l_num_patch))
        lr_grid_feat = lr_patch_feat.view(b_times_2, low_H, low_W, d).permute(0, 3, 1, 2)
        lr_grid_up = F.interpolate(
            lr_grid_feat,
            scale_factor=2,
            mode='nearest'
        )
        lr_patch_feat_up = lr_grid_up.permute(0, 2, 3, 1).flatten(1, 2)

        patch_feature = torch.cat([hr_patch_feat, lr_patch_feat_up], dim=2)

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


