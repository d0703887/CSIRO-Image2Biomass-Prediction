import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel
from model.MLP import MLP
import math
from peft import LoraConfig, get_peft_model

class DinoV3ConvNeXtGatingMultiScale(nn.Module):
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
        self.embed_dim = self.backbone.config.hidden_sizes[-1]

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

        # Feature map fusion
        self.proj_dim = 256
        backbone_hidden_sizes = self.backbone.config.hidden_sizes
        # self.proj_s1 = nn.Conv2d(backbone_hidden_sizes[0], self.proj_dim, kernel_size=1)
        self.proj_s2 = nn.Conv2d(backbone_hidden_sizes[1], self.proj_dim, kernel_size=1)
        self.proj_s3 = nn.Conv2d(backbone_hidden_sizes[2], self.proj_dim, kernel_size=1)
        self.proj_s4 = nn.Conv2d(backbone_hidden_sizes[3], self.proj_dim, kernel_size=1)

        # self.down_s1 = nn.Sequential(
        #     nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=3, stride=2, padding=1),  # 128x128
        #     nn.ReLU(),
        #     nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=3, stride=2, padding=1)  # 64x64
        # )
        self.down_s2 = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=3, stride=2, padding=1)
        self.fusion = nn.Conv2d(self.proj_dim * 3, self.proj_dim, kernel_size=3, padding=1)

        # Biomass head
        self.green_mlp = MLP(self.proj_dim, hidden_dim, mode="biomass")
        self.clover_mlp = MLP(self.proj_dim, hidden_dim, mode="biomass")
        self.dead_mlp = MLP(self.proj_dim, hidden_dim, mode="biomass")

        # Gate MLP to prevent noise-cumulation
        # self.green_gate = MLP(self.embed_dim, hidden_dim, mode="gate")
        # self.clover_gate = MLP(self.embed_dim, hidden_dim, mode="gate")
        # self.dead_gate = MLP(self.embed_dim, hidden_dim, mode="gate")
        #
        # # Auxiliary height prediction
        # if self.predict_height:
        #     self.height_mlp = make_head("height")

    def train(self, mode=True):
        super().train(mode)
        if self.training_mode == "freeze_backbone":
            self.backbone.eval()
        return self

    def aggregate_biomass(self, biomass):
        _, num_patch, _ = biomass.shape
        biomass = biomass.view(-1, 2, num_patch)
        agg = biomass.sum(dim=(1, 2))
        return agg

    def fuse_feat_maps(self, feat_maps):
        s1, s2, s3, s4 = feat_maps
        # p1 = self.proj_s1(s1)
        p2 = self.proj_s2(s2)
        p3 = self.proj_s3(s3)
        p4 = self.proj_s4(s4)

        # out1 = self.down_s1(p1)
        out2 = self.down_s2(p2)
        out3 = p3
        out4 = F.interpolate(p4, size=(64, 64), mode="bilinear", align_corners=False)
        concat = torch.cat([out2, out3, out4], dim=1)
        final_map = self.fusion(concat) # (B * 2, 128, 64, 64)
        final_map = final_map.flatten(2).transpose(1, 2) # (B * 2, 64 * 64, 128)
        return final_map

    # def aggregate_height(self, patch_height, weight):
    #     _, num_patch, _ = patch_height.shape
    #     patch_height = patch_height.view(-1, 2, num_patch)
    #     weight = weight.view(-1, 2, num_patch)
    #     weighted_sum = torch.sum(patch_height * weight, dim=(1, 2))
    #     weight_sum = torch.sum(weight, dim=(1, 2))
    #     return weighted_sum / (weight_sum + 1e-6)

    def forward(self, x, return_patch_preds=False, return_gates=False):
        conv_out = self.backbone(x, output_hidden_states=True)
        feat_maps = conv_out.hidden_states[1:]
        final_map = self.fuse_feat_maps(feat_maps) # (B * 2, 64 * 64, 128)

        # Biomass prediction (B * 2, height * width, 1)
        raw_green= self.green_mlp(final_map)
        raw_clover = self.clover_mlp(final_map)
        raw_dead = self.dead_mlp(final_map)

        # # Gates
        # green_gate = self.green_gate(feats)
        # clover_gate = self.clover_gate(feats)
        # dead_gate = self.dead_gate(feats)

        # Apply gates
        # patch_green = raw_green * green_gate
        # patch_clover = raw_clover * clover_gate
        # patch_dead = raw_dead * dead_gate

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

        # if self.predict_height:
        #     weight = (green_gate + clover_gate + dead_gate).clamp(max=1.0)
        #     patch_height = self.height_mlp(patch_feature)
        #     pred_dict["Avg_Height"] = self.aggregate_height(patch_height, weight)


        # TODO: do not return tiled value
        # if return_gates:
        #     pred_dict.update({
        #         "Tile_Gate_Dry_Green_g": green_gate,
        #         "Tile_Gate_Dry_Clover_g": clover_gate,
        #         "Tile_Gate_Dry_Dead_g": dead_gate
        #     })

        if return_patch_preds:
            pred_dict.update({
                "Tile_Dry_Green_g": raw_green,
                "Tile_Dry_Clover_g": raw_clover,
                "Tile_Dry_Dead_g": raw_dead
            })

        return pred_dict


if __name__ == '__main__':
    model = DinoV3ConvNeXtGatingMultiScale(
        model_name="facebook/dinov3-convnext-base-pretrain-lvd1689m",
        hidden_dim=128,
    )
    device = "cuda"
    model.to(device)
    x = torch.rand((2, 3, 1024, 1024), dtype=torch.float32, device=device)
    model(x)


