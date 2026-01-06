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

from model.DinoV3MultiScale import DinoV3MultiScale
from model.DinoV3GatingMultiScale import DinoV3GatingMultiScale
from model.DinoV3ViT import DinoV3ViT
from utils.utils import load_CSIRO, CSIRO_group_k_fold, CSIRO_stratified_group_k_fold
from dataset import CSIROMultiScaleDataset, CSIRODataset


LOSS_KEYS = [
    "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Avg_Height"
]

class Trainer:
    def __init__(
            self,
            df: pd.DataFrame,
            train_idxs,
            val_idxs,
            train_transforms,
            val_transforms,
            config: dict,
    ):

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.project_dir = f"wandb/{self.timestamp}_CSIRO"
        os.makedirs(self.project_dir, exist_ok=True)
        self.df = df
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.config = config

        # Training config
        self.epochs = config["epochs"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.input_H = config["resolution"]
        self.input_W = config["resolution"]
        self.loss_coefficient = config["loss_coefficient"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.stage2_start_epoch = config["stage2_start_epoch"]
        self.accumulation_steps = config["accumulation_steps"]

        # Model config
        self.gating = config["gating"]
        self.multi_scale = config["multi_scale"]
        self.model_name = config["model_name"]
        self.training_mode = config["training_mode"]
        self.hidden_dim = config["hidden_dim"]
        self.predict_height = config["predict_height"]

        # Other
        self.data_folder = config["data_folder"]
        self.wandb_mode = config["wandb_mode"]

        self.regression_loss_fn = nn.HuberLoss()
        self.r2_coeff = {
            "Dry_Green_g": 0.1,
            "Dry_Clover_g": 0.1,
            "Dry_Dead_g": 0.1,
            "Dry_Total_g": 0.5,
            "GDM_g": 0.2
        }

        #self.global_weighted_mean = self._compute_global_mean(val_dataset)

    def _initialize_wandb(self, fold_idx: int):
        wandb_run = wandb.init(
            entity="d0703887",
            project="CSIRO",
            name=f"{self.timestamp}_ViT_{fold_idx}",
            group=f"{self.timestamp}_ViT",
            config=self.config,
            mode=self.wandb_mode,
            dir=self.project_dir
        )
        return wandb_run

    def _initialize_model(self):
        if self.multi_scale:
            if self.gating:
                model = DinoV3GatingMultiScale(
                    model_name=self.model_name,
                    hidden_dim=self.hidden_dim,
                    training_mode=self.training_mode,
                    predict_height=self.predict_height
                )
            else:
                model = DinoV3MultiScale(
                    model_name=self.model_name,
                    hidden_dim=self.hidden_dim,
                    training_mode=self.training_mode,
                    predict_height=self.predict_height
                )

        else:
            model = DinoV3ViT(
                model_name=self.model_name,
                hidden_dim=self.hidden_dim,
                training_mode=self.training_mode,
                predict_height=self.predict_height
            )
        return model

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

    def process_batch(self, model, data_dict):
        for k, v in data_dict.items():
            data_dict[k] = v.to(self.device)

        if self.multi_scale:
            b_tmp = data_dict["HR_Input_Img"].shape[0]
            hr_input_imgs = data_dict["HR_Input_Img"].view(b_tmp * 2, 3, self.input_H, self.input_W)
            lr_input_imgs = data_dict["LR_Input_Img"].view(b_tmp * 2, 3, self.input_H // 2, self.input_W // 2)
            pred_dict = model(hr_input_imgs, lr_input_imgs)
        else:
            input_img = data_dict["Input_Img"]
            b = input_img.shape[0]
            input_img = input_img.view(b * 2, 3, self.input_H, self.input_W)
            pred_dict = model(input_img)

        loss_dict = {}
        total_loss = 0

        for k in pred_dict.keys():
            if k in self.loss_coefficient.keys():
                loss = self.regression_loss_fn(pred_dict[k], data_dict[k])
                loss_dict[k] = loss
                total_loss += loss * self.loss_coefficient[k]
        loss_dict["main_loss"] = total_loss

        detached_preds = {}
        for k, v in pred_dict.items():
            detached_preds[k] = v.detach()

        return loss_dict, detached_preds

    def train_one_epoch(self, model, optimizer, train_dataloader, epoch, global_mean):
        model.train()
        data_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        optimizer.zero_grad()

        for i, data_dict in enumerate(data_pbar):
            loss_dict, preds = self.process_batch(model, data_dict)

            loss = loss_dict["main_loss"] / self.accumulation_steps
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

            data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

            for k, v in loss_dict.items():
                loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

            for k in self.r2_coeff.keys():
                all_preds[k].append(preds[k].cpu())
                all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(train_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_scores = self.compute_r2(all_preds, all_targets, global_mean)
        avg_losses.update(r2_scores)

        # Build prediction table
        prediction_tables = self.build_prediction_table(all_preds, all_targets)

        return avg_losses, prediction_tables


    def validation(self, model, optimizer, val_dataloader, epoch, global_mean):
        model.eval()
        data_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        with torch.no_grad():
            for data_dict in data_pbar:
                loss_dict, preds = self.process_batch(model, data_dict)
                data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

                for k, v in loss_dict.items():
                    loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

                for k in self.r2_coeff.keys():
                    all_preds[k].append(preds[k].cpu())
                    all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(val_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_scores = self.compute_r2(all_preds, all_targets, global_mean)
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


    def _initialize_data(self, fold_idx):
        train_df = self.df.iloc[self.train_idxs[fold_idx]]
        val_df = self.df.iloc[self.val_idxs[fold_idx]]
        if self.multi_scale:
            train_dataloader = DataLoader(
                CSIROMultiScaleDataset(self.data_folder, train_df, self.input_H, self.input_W, self.train_transforms, is_train=True),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            val_dataloader = DataLoader(
                CSIROMultiScaleDataset(self.data_folder, val_df, self.input_H, self.input_W, self.val_transforms, is_train=False),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            train_dataloader = DataLoader(
                CSIRODataset(self.data_folder, train_df, self.train_transforms),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            val_dataloader = DataLoader(
                CSIRODataset(self.data_folder, val_df, self.val_transforms),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

        print(f"Training Data Length: {len(train_df)}")
        print(f"Validation Data Length: {len(val_df)}")

        return train_dataloader, val_dataloader

    def train_one_fold(self, fold_idx: int):
        wandb_run = self._initialize_wandb(fold_idx)
        train_dataloader, val_dataloader = self._initialize_data(fold_idx)
        val_global_mean = self._compute_global_mean(val_dataloader.dataset)
        train_global_mean = self._compute_global_mean(train_dataloader.dataset)

        model = self._initialize_model()
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.stage2_start_epoch if self.training_mode == "full_finetune" else self.epochs,
            eta_min=1e-5
        )

        best_val_r2 = -float("inf")
        console = Console()

        for epoch in range(1, self.epochs + 1):
            # Two-Stage Training
            if self.training_mode == "full_finetune" and epoch == 1:
                print('Stage 1: Freezing Backbone')
                for param in model.backbone.parameters():
                    param.requires_grad = False

            if self.training_mode == "full_finetune" and epoch == self.stage2_start_epoch + 1:
                print("Stage 2: Full Fine-Tune")
                for param in model.backbone.parameters():
                    param.requires_grad = True

                backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
                head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

                stage2_head_lr = self.lr * 0.1  # e.g., 1e-5
                stage2_backbone_lr = stage2_head_lr * 0.1  # e.g., 1e-6 (Critical!)

                optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': stage2_backbone_lr},
                    {'params': head_params, 'lr': stage2_head_lr}
                ], weight_decay=self.weight_decay)

                # 4. New Scheduler for remaining epochs
                remaining_epochs = self.epochs - epoch + 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs,
                    eta_min=1e-7
                )

            train_metrics, train_pred_tables = self.train_one_epoch(model, optimizer, train_dataloader, epoch, train_global_mean)
            val_metrics, val_pred_tables = self.validation(model, optimizer, val_dataloader, epoch, val_global_mean)
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            # Rich logging
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

            log = {"lr": current_lr}
            log.update(self._prefix_metrics(train_metrics, "train"))
            log.update(self._prefix_metrics(val_metrics, "val"))

            # Save model
            cur_r2 = val_metrics["r2"]
            if cur_r2 > best_val_r2:
                log.update(self._prefix_metrics(train_pred_tables, "Train_Pred"))
                log.update(self._prefix_metrics(val_pred_tables, "val_Pred"))

                best_val_r2 = cur_r2
                torch.save(model.state_dict(), os.path.join(wandb_run.dir, f"{fold_idx}_best_model_{best_val_r2:.3f}.pth"))
                print(f"New best model saved! (R2: {best_val_r2:.4f})")

            # Wandb logging
            wandb_run.log(log, step=epoch)

        wandb_run.finish()
        return best_val_r2

    def cross_validation(self):
        val_r2s = []
        for fold_idx in range(len(self.train_idxs)):
            cur_val_r2 = self.train_one_fold(fold_idx)
            val_r2s.append(cur_val_r2)

        for fold_idx, r2 in enumerate(val_r2s):
            print(f"Fold {fold_idx}: R2 = {r2:.4f}")
        print(f"Avg R2 = {sum(val_r2s) / len(val_r2s)}")


def main(config, mode: str):
    df = load_CSIRO(config["data_folder"])
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

        v2.Resize((config["resolution"], config["resolution"]), antialias=True),

        # Color
        v2.RandomApply([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
        ], p=0.5),  # High probability!
        # v2.RandomAutocontrast(p=0.3),
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
        v2.Resize((config["resolution"], config["resolution"]), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    if config["multi_scale"]:
        train_transforms = v2.Compose([
            # v2.ToImage(),
            #
            # # Geometric
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomVerticalFlip(p=0.5),
            # v2.RandomChoice([
            #     v2.Identity(),
            #     v2.RandomRotation(degrees=(90, 90), expand=False),
            #     v2.RandomRotation(degrees=(180, 180), expand=False),
            #     v2.RandomRotation(degrees=(270, 270), expand=False)
            # ]),
            #
            # v2.Resize((config["resolution"], config["resolution"]), antialias=True),
            #
            # # Color
            # v2.RandomApply([
            #     v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
            # ], p=0.5),  # High probability!
            # # v2.RandomAutocontrast(p=0.3),
            # v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            #
            # # Blur
            # v2.RandomApply([v2.GaussianBlur(kernel_size=(11, 11), )], p=0.3),

            # Normalization
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
        val_transforms = v2.Compose([
            # v2.ToImage(),
            # v2.Resize((config["resolution"], config["resolution"]), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])


    train_idxs, val_idxs = CSIRO_stratified_group_k_fold(df)

    trainer = Trainer(
        df,
        train_idxs,
        val_idxs,
        train_transforms,
        val_transforms,
        config
    )
    if mode == "cross-validation":
        trainer.cross_validation()
    elif mode == "single-fold":
        trainer.train_one_fold(0)
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_coefficient", type=float, nargs="+")
    parser.add_argument("--stage2_start_epoch", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    parser.add_argument("--gating", action="store_true")
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--training_mode", type=str, default="freeze_backbone", choices=["full_finetune", "lora", "freeze_backbone"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--predict_height", action="store_true")

    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--mode", type=str, default="single-fold")
    args = parser.parse_args()

    # Validating loss coefficient
    coeffs = args.loss_coefficient
    if len(coeffs) != len(LOSS_KEYS):
        raise ValueError(
            f"Expected {len(LOSS_KEYS)} loss coefficients, but got {len(coeffs)}.\n"
            f"Required keys: {LOSS_KEYS}"
        )

    loss_coefficient = dict(zip(LOSS_KEYS, coeffs))

    config = {
        # Training config
        "epochs": args.epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "loss_coefficient": loss_coefficient,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stage2_start_epoch": args.stage2_start_epoch,
        "accumulation_steps": args.accumulation_steps,

        # Model config
        "gating": args.gating,
        "multi_scale": args.multi_scale,
        "model_name": args.model_name,
        "training_mode": args.training_mode,
        "hidden_dim": args.hidden_dim,
        "predict_height": args.predict_height,

        # Other
        "resolution": args.resolution,
        "data_folder": args.data_folder,
        "wandb_mode": args.wandb_mode
    }
    main(config, args.mode)

