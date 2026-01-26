import torch.nn as nn
import torch
from transformers import AutoModel
from model.MLP import MLP
import math
from peft import LoraConfig, get_peft_model


class DinoV3ViT(nn.Module):
    def __init__(
            self,
            model_name: str,
            hidden_dim: int,
            training_mode: str = "freeze_backbone",
            predict_height: bool = False,
            split_img: bool = True
    ):
        super().__init__()
        self.training_mode = training_mode
        self.predict_height = predict_height
        self.split_img = split_img

        # DinoV3
        self.backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.embed_dim = self.backbone.config.hidden_size

        if self.training_mode == "lora":
            peft_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.3,
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

        # Biomass head
        self.green_mlp = MLP(self.embed_dim * 2, hidden_dim, mode="biomass", dropout=0.6)
        self.clover_mlp = MLP(self.embed_dim * 2, hidden_dim, mode="biomass", dropout=0.6)
        self.dead_mlp = MLP(self.embed_dim * 2, hidden_dim, mode="biomass", dropout=0.6)

        # Occlusion head
        # self.green_occlusion_mlp = MLP(self.embed_dim if not self.predict_height else self.embed_dim + hidden_dim, hidden_dim, mode="biomass", dropout=0.6)
        # self.clover_occlusion_mlp = MLP(self.embed_dim if not self.predict_height else self.embed_dim + hidden_dim, hidden_dim, mode="biomass", dropout=0.6)
        # self.dead_occlusion_mlp = MLP(self.embed_dim if not self.predict_height else self.embed_dim + hidden_dim, hidden_dim, mode="biomass", dropout=0.6)

        # Gate
        # self.green_gate = MLP(self.embed_dim * 2, hidden_dim // 2, mode="gate", dropout=0.6)
        # self.clover_gate = MLP(self.embed_dim * 2, hidden_dim // 2, mode="gate", dropout=0.6)
        # self.dead_gate = MLP(self.embed_dim * 2, hidden_dim // 2, mode="gate", dropout=0.6)

        # Auxiliary height prediction
        if self.predict_height:
            self.height_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 1)
            )

    def train(self, mode=True):
        super().train(mode)
        if self.training_mode == "freeze_backbone":
            self.backbone.eval()
        return self

    def aggregate_biomass(self, biomass):
        _, num_patch, _ = biomass.shape
        if self.split_img:
            biomass = biomass.view(-1, 2, num_patch)  # (B, 2, num_patch)
        agg = biomass.sum(dim=(1, 2))
        return agg

    def forward(self, x, return_patch_preds=False, return_gates=False):
        """

        :param x: shape (B, C, H, W)
        :return: preds
        """
        vit_out = self.backbone(x)
        patch_feature = vit_out.last_hidden_state[:, 5:, :] # (B * 2, num_patch, embed_dim)
        global_feature = vit_out.pooler_output  # (B * 2, embed_dim)
        expanded_global_feature = global_feature.unsqueeze(1).expand(-1, patch_feature.shape[1], -1)
        patch_fused_feature = torch.cat([patch_feature, expanded_global_feature], dim=-1)
        #occlusion_fused_feature = global_feature

        # Biomass prediction
        raw_green= self.green_mlp(patch_fused_feature) * 1e-2
        raw_clover = self.clover_mlp(patch_fused_feature) * 1e-2
        raw_dead = self.dead_mlp(patch_fused_feature) * 1e-2
        # raw_green_occlusion = self.green_occlusion_mlp(occlusion_fused_feature).squeeze(-1)
        # raw_clover_occlusion = self.clover_occlusion_mlp(occlusion_fused_feature).squeeze(-1)
        # raw_dead_occlusion = self.dead_occlusion_mlp(occlusion_fused_feature).squeeze(-1)

        # Gate
        # green_gate = self.green_gate(patch_fused_feature)
        # clover_gate = self.clover_gate(patch_fused_feature)
        # dead_gate = self.dead_gate(patch_fused_feature)

        # Apply gating
        # raw_green *= green_gate
        # raw_clover *= clover_gate
        # raw_dead *= dead_gate

        if not self.training:
            raw_green[raw_green < 0] = 0
            raw_clover[raw_clover < 0] = 0
            raw_dead[raw_dead < 0] = 0

        # Aggregation
        pred_green = self.aggregate_biomass(raw_green)
        pred_clover = self.aggregate_biomass(raw_clover)
        pred_dead = self.aggregate_biomass(raw_dead)

        pred_dict = {
            "Dry_Green_g": pred_green,
            "Dry_Clover_g": pred_clover,
            "Dry_Dead_g": pred_dead,
            "Dry_Total_g": pred_green + pred_clover + pred_dead,
            "GDM_g": pred_green + pred_clover
        }

        if self.predict_height:
            height = self.height_mlp(global_feature).squeeze(-1)
            pred_dict["Height_Ave_cm"] = height

        if return_patch_preds:
            pred_dict.update({
                "Tile_Dry_Green_g": raw_green,
                "Tile_Dry_Clover_g": raw_clover,
                "Tile_Dry_Dead_g": raw_dead,
            })

        # if return_gates:
        #     pred_dict.update({
        #         "Tile_Gate_Dry_Green_g": green_gate,
        #         "Tile_Gate_Dry_Clover_g": clover_gate,
        #         "Tile_Gate_Dry_Dead_g": dead_gate,
        #     })

        return pred_dict


