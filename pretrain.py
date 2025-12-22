import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import wandb
from rich.console import Console
from rich.table import Table

import pandas as pd
import os
from datetime import datetime
import argparse

from model.DinoV3Backbone import DinoV3Backbone
from utils.utils import merge_Irish_Grass, load_CSIRO
from dataset import CombinedExternalDataset, CSIRODataset


class Trainer:
    def __init__(
            self,
            train_dataloader,
            val_dataloader,
            config: dict,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.project_dir = f"wandb/{self.timestamp}_CSIRO_pretrain"
        os.makedirs(self.project_dir, exist_ok=True)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # Training config
        self.epochs = config["epochs"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.loss_coefficient = {
            "Dry_Green_g": 0.5,
            "Dry_Clover_g": 0.5
        }
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.stage2_start_epoch = config["stage2_start_epoch"]

        # Model config
        self.model_name = config["model_name"]
        self.hidden_dim = config["hidden_dim"]
        self.predict_height = config["predict_height"]

        # Other
        self.wandb_mode = config["wandb_mode"]

        self.regression_loss_fn = nn.HuberLoss()
        self.classification_loss_fn = nn.BCELoss()
        self.r2_coeff = {
            "Dry_Green_g": 0.5,
            "Dry_Clover_g": 0.5,
        }

        # Initialize
        self.wandb_run = wandb.init(
            entity="d0703887",
            project="CSIRO",
            name=f"{self.timestamp}_CSIRO_pretrain",
            config=self.config,
            mode=self.wandb_mode,
            dir=self.project_dir
        )

        self.model = DinoV3Backbone(
            model_name=self.model_name,
            hidden_dim=self.hidden_dim,
            freeze_backbone=False,
            predict_height=self.predict_height,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train_global_mean = self._compute_global_mean(train_dataloader.dataset)
        self.val_global_mean = self._compute_global_mean(val_dataloader.dataset)


    def _prefix_metrics(self, metrics: dict, prefix: str):
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def _compute_global_mean(self, dataset):
        global_means = {k: 0 for k in self.r2_coeff.keys()}
        for row in dataset.data_values:
            for target_name, _ in self.r2_coeff.items():
                global_means[target_name] += row[target_name]
        for k in global_means.keys():
            global_means[k] /= len(dataset.data_values)

        return global_means

    def process_batch(self, data_dict, is_train=True, mode="tiled"):
        for k, v in data_dict.items():
            data_dict[k] = v.to(self.device)

        if is_train:
            self.optimizer.zero_grad()

        if mode == "tiled":
            input_imgs = data_dict["Input_Img"].view(data_dict["Input_Img"].shape[0] * 2, 3, 768, 768)
        else:
            input_imgs = data_dict["Input_Img"]
        pred_dict = self.model(input_imgs, mode)

        loss_dict = {}
        total_loss = 0

        for k in pred_dict.keys():
            if k in self.loss_coefficient.keys():
                loss = self.regression_loss_fn(pred_dict[k], data_dict[k])
                loss_dict[k] = loss
                total_loss += loss * self.loss_coefficient[k]
        loss_dict["main_loss"] = total_loss

        if is_train:
            total_loss.backward()
            self.optimizer.step()

        detached_preds = {}
        for k, v in pred_dict.items():
            detached_preds[k] = v.detach()

        return loss_dict, detached_preds

    def train_one_epoch(self, epoch):
        self.model.train()
        data_pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        for data_dict in data_pbar:
            loss_dict, preds = self.process_batch(data_dict, mode=None)
            data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

            for k, v in loss_dict.items():
                loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

            for k in self.r2_coeff.keys():
                all_preds[k].append(preds[k].cpu())
                all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(self.train_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_scores = self.compute_r2(all_preds, all_targets, self.train_global_mean)
        avg_losses.update(r2_scores)

        # Build prediction table
        # TODO: only build prediction table if r2 score improve
        prediction_tables = self.build_prediction_table(all_preds, all_targets)

        return avg_losses, prediction_tables


    def validation(self, epoch):
        self.model.eval()
        data_pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        with torch.no_grad():
            for data_dict in data_pbar:
                loss_dict, preds = self.process_batch(data_dict, False, mode="tiled")
                data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

                for k, v in loss_dict.items():
                    loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

                for k in self.r2_coeff.keys():
                    all_preds[k].append(preds[k].cpu())
                    all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(self.val_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_scores = self.compute_r2(all_preds, all_targets, self.val_global_mean)
        avg_losses.update(r2_scores)

        # Build prediction table
        prediction_tables = self.build_prediction_table(all_preds, all_targets)
        return avg_losses, prediction_tables

    # Use standard r2 score instead of competition's r2 score
    def compute_r2(self, pred_dict, target_dict, weighted_mean):
        r2_scores =  {"r2": 0}
        for target_name, coeff in self.r2_coeff.items():
            flat_preds = torch.cat(pred_dict[target_name])
            flat_targets = torch.cat(target_dict[target_name])

            mse = torch.sum((flat_targets - flat_preds) ** 2)
            var = torch.sum((flat_targets - weighted_mean[target_name]) ** 2)

            r2 = 1 - mse / var
            r2_scores[f"{target_name}_r2"] = r2
            r2_scores["r2"] += coeff * r2

        return r2_scores

    def build_prediction_table(self, pred_dict, target_dict):
        log_tables = {}
        for target_name, coeff in self.r2_coeff.items():
            flat_preds = torch.cat(pred_dict[target_name])
            flat_targets = torch.cat(target_dict[target_name])
            df_view = pd.DataFrame({
                "Target": flat_targets.cpu().numpy(),  # Ensure on CPU
                "Pred": flat_preds.cpu().numpy(),  # Ensure on CPU
            })
            table = wandb.Table(dataframe=df_view)
            log_tables[target_name] = table

        return log_tables


    def train(self):
        best_val_r2 = -float("inf")
        console = Console()

        for epoch in range(1, self.epochs + 1):
            # Two-Stage Training
            # Stage 1: freeze backbone
            # Stage 2: unfreeze backbone and lower lr
            if epoch == 1:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            if epoch == self.stage2_start_epoch + 1:
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.1

            train_metrics, train_pred_tables = self.train_one_epoch(epoch)
            val_metrics, val_pred_tables = self.validation(epoch)

            # Rich ogging
            table = Table(title=f"Epoch {epoch:02d}/{self.epochs:02d} Summary", show_header=True,
                          header_style="bold magenta")

            # Add Columns
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Train", justify="right", style="red")
            table.add_column("Val", justify="right", style="red")

            # Add Main Rows
            table.add_row("Main Loss", f"{train_metrics['main_loss']:.4f}", f"{val_metrics['main_loss']:.4f}")
            table.add_row("R2 Score (Weighted)", f"{train_metrics['r2']:.4f}", f"{val_metrics['r2']:.4f}")

            # Add Separator
            table.add_section()

            # Add Per-Target R2 Rows
            # We iterate over your R2 keys to get individual scores
            for target in self.r2_coeff.keys():
                key = f"{target}_r2"
                train_score = train_metrics.get(key, 0.0)
                val_score = val_metrics.get(key, 0.0)

                # Color code negative R2s to make them stand out
                t_str = f"[bold red]{train_score:.4f}[/]" if train_score < 0 else f"{train_score:.4f}"
                v_str = f"[bold red]{val_score:.4f}[/]" if val_score < 0 else f"{val_score:.4f}"

                table.add_row(f"{target} R2", t_str, v_str)

            console.print(table)

            log = {}
            log.update(self._prefix_metrics(train_metrics, "train"))
            log.update(self._prefix_metrics(val_metrics, "val"))

            # Save model
            cur_r2 = val_metrics["r2"]
            if cur_r2 > best_val_r2:
                log.update(self._prefix_metrics(train_pred_tables, "Train_Pred"))
                log.update(self._prefix_metrics(val_pred_tables, "val_Pred"))

                best_val_r2 = cur_r2
                torch.save(self.model.state_dict(), os.path.join(self.wandb_run.dir, f"best_model_{best_val_r2:.3f}.pth"))
                print(f"New best model saved! (R2: {best_val_r2:.4f})")

            # Wandb logging
            self.wandb_run.log(log, step=epoch)

        self.wandb_run.finish()
        return best_val_r2


def main(config):
    train_df, _ = merge_Irish_Grass(config["grass_data_folder"], config["irish_data_folder"])
    train_transforms = v2.Compose([
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
        # Preserve pixel per cm
        v2.Resize((1280, 1104), antialias=True),

        # Color
        v2.RandomApply([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
        ], p=0.5),  # High probability!
        v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),

        # Blur
        v2.RandomApply([v2.GaussianBlur(kernel_size=(11, 11), )], p=0.3),

        # Normalization
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((768, 768), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
    train_transform_dict = {"grass": train_transforms, "irish": train_transforms}

    train_dataloader = DataLoader(
        CombinedExternalDataset(train_df, train_transform_dict),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    csiro_df = load_CSIRO(config["csiro_data_folder"])
    csiro_dataloader = DataLoader(
        CSIRODataset(config["csiro_data_folder"], csiro_df, val_transforms),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    trainer = Trainer(
        train_dataloader=train_dataloader,
        val_dataloader=csiro_dataloader,
        config=config
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage2_start_epoch", type=int, default=10)

    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--predict_height", action="store_true")

    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--irish_data_folder", type=str, default="data/IrishGrassClover")
    parser.add_argument("--grass_data_folder", type=str, default="data/GrassClover")
    parser.add_argument("--csiro_data_folder", type=str, default="data/CSIRO")
    args = parser.parse_args()

    config = {
        # Training config
        "epochs": args.epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stage2_start_epoch": args.stage2_start_epoch,

        # Model config
        "model_name": args.model_name,
        "hidden_dim": args.hidden_dim,
        "predict_height": args.predict_height,

        # Other
        "irish_data_folder": args.irish_data_folder,
        "grass_data_folder": args.grass_data_folder,
        "csiro_data_folder": args.csiro_data_folder,
        "wandb_mode": args.wandb_mode
    }
    main(config)

