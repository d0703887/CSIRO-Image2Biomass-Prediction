import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split

import pandas as pd
import os
from datetime import datetime
import numpy as np
import argparse

from model import DinoV3BackBone
from utils import load_data


LOSS_KEYS = [
    "green_g", "clover_g", "dead_g", "gdm_g",
    "total_g", "height", "has_clover"
]

class CSIRODataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            df: pd.DataFrame,
            transform = None,
            overlap_ratio = 0.1
    ):
        self.data_folder = data_folder
        self.df = df
        self.transform = transform
        self.overlap_ratio = overlap_ratio

        self.img_paths = df["image_path"].tolist()
        self.species = df["Species"].tolist()
        self.data_values = df[[
            "Pre_GSHH_NDVI", "Height_Ave_cm", "Dry_Green_g",
            "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"
        ]].to_dict('records')

    def __len__(self):
        return len(self.df)

    def split_img(self, image):
        _, _, w = image.shape
        center = w // 2
        offset = int(w * self.overlap_ratio / 2)
        left_img = image[:, :, :center + offset]
        right_img = image[:, :, center - offset:]
        return left_img, right_img

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.img_paths[idx])
        img = read_image(img_path)
        left_img, right_img = self.split_img(img)
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        input_img = torch.cat([left_img, right_img]) # (2, C, H, W)
        row = self.data_values[idx]
        return {
            "input_img": input_img,
            "ndvi": torch.tensor(row["Pre_GSHH_NDVI"], dtype=torch.float32),
            "height": torch.tensor(np.log(row["Height_Ave_cm"]), dtype=torch.float32),
            "green_g": torch.tensor(np.log(1 + row["Dry_Green_g"]), dtype=torch.float32),
            "clover_g": torch.tensor(np.log(1 + row["Dry_Clover_g"]), dtype=torch.float32),
            "dead_g": torch.tensor(np.log(1 + row["Dry_Dead_g"]), dtype=torch.float32),
            "total_g": torch.tensor(np.log(1 + row["Dry_Total_g"]), dtype=torch.float32),
            "gdm_g": torch.tensor(np.log(1 + row["GDM_g"]), dtype=torch.float32),
            "has_clover": torch.tensor(1.0 if "clover" in self.species[idx].lower() else 0.0, dtype=torch.float32)
        }

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: CSIRODataset,
            val_dataset: CSIRODataset,
            config: dict,
            wandb_run: wandb.sdk.wandb_run.Run
    ):
        self.model = model
        self.epochs = config["epochs"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.input_H = config["input_H"]
        self.input_W = config["input_W"]
        self.predict_height = config["predict_height"]
        self.predict_has_clover = config["predict_has_clover"]
        self.loss_coefficient = config["loss_coefficient"]
        self.freeze_backbone = config["freeze_backbone"]

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.wandb_run = wandb_run

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.regression_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.BCELoss()
        self.r2_coeff = {
            "green_g": 0.1,
            "clover_g": 0.1,
            "dead_g": 0.1,
            "total_g": 0.5,
            "gdm_g": 0.2
        }

        self.global_weighted_mean = self._compute_global_mean(val_dataset)

    def _prefix_metrics(self, metrics: dict, prefix: str):
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def _compute_global_mean(self, dataset):
        numerator = 0
        denominator = 0
        for row in dataset.data_values:
            for target_name, coeff in self.r2_coeff.items():
                numerator += coeff * row[self._map_key_to_csv(target_name)]
                denominator += coeff
        return numerator / denominator

    def _map_key_to_csv(self, key):
        # Helper to map internal keys back to CSV headers if needed for init calc
        mapping = {
            "green_g": "Dry_Green_g", "clover_g": "Dry_Clover_g",
            "dead_g": "Dry_Dead_g", "total_g": "Dry_Total_g", "gdm_g": "GDM_g"
        }
        return mapping.get(key, key)


    def process_batch(self, data_dict, is_train=True):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)

        if is_train:
            self.optimizer.zero_grad()

        input_imgs = data_dict["input_img"].view(data_dict["input_img"].shape[0] * 2, 3, self.input_H, self.input_W)

        pred_dict = self.model(input_imgs)

        loss_dict = {}
        total_loss = 0
        for key, coeff in self.loss_coefficient.items():
            if key == "has_clover":
                loss = self.classification_loss_fn(pred_dict[key], data_dict[key])
            else:
                loss = self.regression_loss_fn(pred_dict[key], data_dict[key])

            loss_dict[key] = loss
            total_loss += loss * coeff

        loss_dict["main_loss"] = total_loss

        if is_train:
            total_loss.backward()
            self.optimizer.step()

        detached_preds = {k: v.detach() for k, v in pred_dict.items()}
        return loss_dict, detached_preds

    def train_one_epoch(self, epoch: int):
        self.model.train()
        data_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        for data_dict in data_pbar:
            loss_dict, _ = self.process_batch(data_dict)
            data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

            for k, v in loss_dict.items():
                loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

        # Average losses
        avg_losses = {k: v / len(self.train_dataloader) for k, v in loss_accumulator.items()}
        return avg_losses


    def validation(self, epoch: int):
        self.model.eval()
        data_pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        with torch.no_grad():
            for data_dict in data_pbar:
                loss_dict, preds = self.process_batch(data_dict, False)
                data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

                for k, v in loss_dict.items():
                    loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

                for k in self.r2_coeff.keys():
                    all_preds[k].append(preds[k].cpu())
                    all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(self.val_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_score = self.compute_r2(all_preds, all_targets)
        avg_losses["r2"] = r2_score

        original_mae = self.compute_orig_scale_metrics(all_preds, all_targets)
        return avg_losses, original_mae

    def compute_r2(self, pred_dict, target_dict):
        numerator = 0
        denominator = 0
        # Transform log scale to normal range
        for target_name, coeff in self.r2_coeff.items():
            flat_preds = torch.cat(pred_dict[target_name])
            flat_targets = torch.cat(target_dict[target_name])

            mse = torch.sum(((torch.exp(flat_targets) - 1) - (torch.exp(flat_preds) - 1)) ** 2)
            var = torch.sum(((torch.exp(flat_targets) - 1) - self.global_weighted_mean) ** 2)

            numerator += coeff * mse
            denominator += coeff * var

        return 1 - (numerator / denominator).item()

    def compute_orig_scale_metrics(self, pred_dict, target_dict):
        """
        Computes MAE in the original units (grams, cm).
        """
        metrics = {}
        for key, preds in pred_dict.items():
            flat_preds = torch.cat(preds)
            flat_targets = torch.cat(target_dict[key])

            # Inverse Transform Logic
            if key == "height":
                real_pred = torch.exp(flat_preds)
                real_target = torch.exp(flat_targets)
            elif key in ["green_g", "clover_g", "dead_g", "total_g", "gdm_g"]:
                # Dataset used: np.log(1 + x) -> Inverse: exp(x) - 1
                real_pred = torch.exp(flat_preds) - 1
                real_target = torch.exp(flat_targets) - 1

                # Clamp negative predictions to 0 (since mass can't be negative)
                real_pred = torch.relu(real_pred)
            else:
                continue

            # Compute MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(real_pred - real_target))
            metrics[key] = mae.item()

        return metrics

    def train(self):
        for epoch in range(1, self.epochs + 1):
            # two-stage training to stabilize training
            if not self.freeze_backbone:
                if epoch <= 5:
                    for param in self.model.backbone.parameters():
                        param.requires_grad = False
                elif epoch == 6:
                    for param in self.model.backbone.parameters():
                        param.requires_grad = True

            train_metrics = self.train_one_epoch(epoch)
            val_metrics, original_mae = self.validation(epoch)

            log = {}
            log.update(self._prefix_metrics(train_metrics, "train"))
            log.update(self._prefix_metrics(val_metrics, "val"))
            log.update(self._prefix_metrics(original_mae, "orig_MAE"))

            self.wandb_run.log(log)

            print(f"Epoch {epoch + 1}/{self.epochs}: Train Loss={train_metrics['main_loss']:.4f}, Val R2={val_metrics['r2']:.4f}")


def main(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    wandb_run = wandb.init(
        entity="d0703887",
        project="CSIRO",
        name=f"{timestamp}_DecoderOnly",
        config=config,
        mode=config["wandb_mode"]
    )
    df = load_data(config["data_folder"])
    train_transform = v2.Compose([
        v2.ToImage(),

        # Geometric
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice([
            v2.Identity(),
            v2.RandomRotation(degrees=(90, 90), expand=False),
            v2.RandomRotation(degrees=(180, 180), expand=False),
            v2.RandomRotation(degrees=(270, 270), expand=False)
        ]),

        v2.Resize((config["input_H"], config["input_W"]), antialias=True),

        # Color
        v2.RandomAutocontrast(p=0.2),
        v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),

        # Blur
        v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), )], p=0.1),

        # Normalization
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((config["input_H"], config["input_W"]), antialias=True),
        # Normalization
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dataset = CSIRODataset(config["data_folder"], train_df, train_transform, overlap_ratio=0)
    val_dataset = CSIRODataset(config["data_folder"], val_df, val_transform, overlap_ratio=0)

    model = DinoV3BackBone(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        predict_total=config["predict_total"],
        predict_gdm=config["predict_gdm"],
        predict_has_clover=config["predict_has_clover"],
        predict_height=config["predict_height"],
        freeze_backbone=config["freeze_backbone"],
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        wandb_run=wandb_run
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_H", type=int, default=768)
    parser.add_argument("--input_W", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_coefficient", type=float, nargs="+")
    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--predict_total", action="store_true")
    parser.add_argument("--predict_gdm", action="store_true")
    parser.add_argument("--predict_height", action="store_true")
    parser.add_argument("--predict_has_clover", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=1024)

    parser.add_argument("--data_folder", type=str, default="data")
    args = parser.parse_args()

    # Validating loss coefficient
    coeffs = args.loss_coefficient
    if len(coeffs) != len(LOSS_KEYS):
        raise ValueError(
            f"Expected {len(LOSS_KEYS)} loss coefficients, but got {len(coeffs)}.\n"
            f"Required keys: {LOSS_KEYS}"
        )
    if not torch.isclose(torch.tensor(sum(coeffs)), torch.tensor(1.0)):
        raise ValueError(f"Loss coefficients must sum to 1. Current sum: {sum(coeffs)}")

    loss_coefficient = dict(zip(LOSS_KEYS, coeffs))

    config = {
        # Training config
        "epochs": args.epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "input_H": args.input_H,
        "input_W": args.input_W,
        "loss_coefficient": loss_coefficient,
        "lr": args.lr,
        "weight_decay": args.weight_decay,

        # Model config
        "model_name": args.model_name,
        "predict_total": args.predict_total,
        "predict_gdm": args.predict_gdm,
        "predict_height": args.predict_height,
        "predict_has_clover": args.predict_has_clover,
        "freeze_backbone": args.freeze_backbone,
        "hidden_dim": args.hidden_dim,

        # Other
        "data_folder": args.data_folder,
        "wandb_mode": args.wandb_mode
    }
    main(config)

