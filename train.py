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
from utils import load_data, group_k_fold


LOSS_KEYS = [
    "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g",
    "Dry_Total_g", "Height_Ave_cm", "Has_Clover"
]

class CSIRODataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            df: pd.DataFrame,
            transform = None,
    ):
        self.data_folder = data_folder
        self.df = df
        self.transform = transform

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
        left_img = image[:, :, :center]
        right_img = image[:, :, center:]
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
            "Input_Img": input_img,
            "Pre_GSHH_NDVI": torch.tensor(row["Pre_GSHH_NDVI"], dtype=torch.float32),
            "Height_Ave_cm": torch.tensor(np.log(row["Height_Ave_cm"]), dtype=torch.float32),
            "Dry_Green_g": torch.tensor(np.log(1 + row["Dry_Green_g"]), dtype=torch.float32),
            "Dry_Clover_g": torch.tensor(np.log(1 + row["Dry_Clover_g"]), dtype=torch.float32),
            "Dry_Dead_g": torch.tensor(np.log(1 + row["Dry_Dead_g"]), dtype=torch.float32),
            "Dry_Total_g": torch.tensor(np.log(1 + row["Dry_Total_g"]), dtype=torch.float32),
            "GDM_g": torch.tensor(np.log(1 + row["GDM_g"]), dtype=torch.float32),
            "Has_Clover": torch.tensor(1.0 if "clover" in self.species[idx].lower() else 0.0, dtype=torch.float32)
        }

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
        self.input_H = 768
        self.input_W = 768
        self.loss_coefficient = config["loss_coefficient"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.stage2_start_epoch = config["stage2_start_epoch"]

        # Model config
        self.model_name = config["model_name"]
        self.predict_total = config["predict_total"]
        self.predict_gdm = config["predict_gdm"]
        self.predict_height = config["predict_height"]
        self.predict_has_clover = config["predict_has_clover"]
        self.freeze_backbone = config["freeze_backbone"]
        self.hidden_dim = config["hidden_dim"]

        # Other
        self.data_folder = config["data_folder"]
        self.wandb_mode = config["wandb_mode"]


        # self.wandb_run = wandb_run

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.regression_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.BCELoss()
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
            name=f"{self.timestamp}_CSIRO_{fold_idx}",
            group=f"{self.timestamp}_CSIRO",
            config=self.config,
            mode=self.wandb_mode,
            dir=self.project_dir
        )
        return wandb_run

    def _initialize_model(self):
        model = DinoV3BackBone(
            model_name=self.model_name,
            hidden_dim=self.hidden_dim,
            predict_total=self.predict_total,
            predict_gdm=self.predict_gdm,
            predict_has_clover=self.predict_has_clover,
            predict_height=self.predict_height,
            freeze_backbone=self.freeze_backbone,
        )
        return model

    def _prefix_metrics(self, metrics: dict, prefix: str):
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def _compute_global_mean(self, dataset):
        numerator = 0
        denominator = 0
        for row in dataset.data_values:
            for target_name, coeff in self.r2_coeff.items():
                numerator += coeff * row[target_name]
                denominator += coeff
        return numerator / denominator

    def process_batch(self, model, optimizer, data_dict, is_train=True):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)

        if is_train:
            optimizer.zero_grad()

        input_imgs = data_dict["Input_Img"].view(data_dict["Input_Img"].shape[0] * 2, 3, self.input_H, self.input_W)
        pred_dict = model(input_imgs)

        loss_dict = {}
        total_loss = 0

        for k in pred_dict.keys():
            if k == "Has_Clover":
                loss = self.classification_loss_fn(pred_dict[k], data_dict[k])
            else:
                loss = self.regression_loss_fn(pred_dict[k], data_dict[k])
            loss_dict[k] = loss
            total_loss += loss * self.loss_coefficient[k]
        loss_dict["main_loss"] = total_loss

        if is_train:
            total_loss.backward()
            optimizer.step()

        detached_preds = {}
        for k, v in pred_dict.items():
            detached_preds[k] = v.detach()

        return loss_dict, detached_preds

    def train_one_epoch(self, model, optimizer, train_dataloader, epoch: int):
        model.train()
        data_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        for data_dict in data_pbar:
            loss_dict, _ = self.process_batch(model, optimizer, data_dict)
            data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

            for k, v in loss_dict.items():
                loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

        # Average losses
        avg_losses = {k: v / len(train_dataloader) for k, v in loss_accumulator.items()}
        return avg_losses


    def validation(self, model, optimizer, val_dataloader, epoch, global_weighted_mean):
        model.eval()
        data_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}/{self.epochs}: ", position=0)
        loss_accumulator = {}
        all_preds = {k: [] for k in self.r2_coeff.keys()}
        all_targets = {k: [] for k in self.r2_coeff.keys()}

        with torch.no_grad():
            for data_dict in data_pbar:
                loss_dict, preds = self.process_batch(model, optimizer, data_dict, False)
                data_pbar.set_postfix({"loss": loss_dict["main_loss"].item()})

                for k, v in loss_dict.items():
                    loss_accumulator[k] = loss_accumulator.get(k, 0) + v.item()

                for k in self.r2_coeff.keys():
                    all_preds[k].append(preds[k].cpu())
                    all_targets[k].append(data_dict[k].cpu())

        # Average losses
        avg_losses = {k: v / len(val_dataloader) for k, v in loss_accumulator.items()}

        # Compute R2 on full dataset tensors
        r2_score = self.compute_r2(all_preds, all_targets, global_weighted_mean)
        avg_losses["r2"] = r2_score

        original_mae = self.compute_orig_scale_metrics(all_preds, all_targets)
        return avg_losses, original_mae

    def compute_r2(self, pred_dict, target_dict, global_weighted_mean):
        numerator = 0
        denominator = 0
        # Transform log scale to normal range
        for target_name, coeff in self.r2_coeff.items():
            flat_preds = torch.cat(pred_dict[target_name])
            flat_targets = torch.cat(target_dict[target_name])

            mse = torch.sum(((torch.exp(flat_targets) - 1) - (torch.exp(flat_preds) - 1)) ** 2)
            var = torch.sum(((torch.exp(flat_targets) - 1) - global_weighted_mean) ** 2)

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
            if key in ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"]:
                # Dataset used: np.log(1 + x) -> Inverse: exp(x) - 1
                real_pred = torch.exp(flat_preds) - 1
                real_target = torch.exp(flat_targets) - 1

            else:
                continue

            # Compute MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(real_pred - real_target))
            metrics[key] = mae.item()

        return metrics

    def _initialize_data(self, fold_idx):
        train_df = self.df.iloc[self.train_idxs[fold_idx]]
        val_df = self.df.iloc[self.val_idxs[fold_idx]]
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
        return train_dataloader, val_dataloader

    def train_one_fold(self, fold_idx: int):
        wandb_run = self._initialize_wandb(fold_idx)
        train_dataloader, val_dataloader = self._initialize_data(fold_idx)
        global_weighted_mean = self._compute_global_mean(val_dataloader.dataset)
        model = self._initialize_model()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_r2 = -float("inf")

        for epoch in range(1, self.epochs + 1):
            # two-stage training to stabilize training
            if not self.freeze_backbone:
                if epoch <= self.stage2_start_epoch:
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                # Stage 2: unfreeze backbone and lower lr
                elif epoch == self.stage2_start_epoch:
                    for param in model.backbone.parameters():
                        param.requires_grad = True
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.1

            train_metrics = self.train_one_epoch(model, optimizer, train_dataloader, epoch)
            val_metrics, original_mae = self.validation(model, optimizer, val_dataloader, epoch, global_weighted_mean)

            log = {}
            log.update(self._prefix_metrics(train_metrics, "train"))
            log.update(self._prefix_metrics(val_metrics, "val"))
            log.update(self._prefix_metrics(original_mae, "orig_MAE"))

            wandb_run.log(log, step=epoch)

            print(f"Epoch {epoch}/{self.epochs}: Train Loss={train_metrics['main_loss']:.4f}, Val R2={val_metrics['r2']:.4f}")

            cur_r2 = val_metrics["r2"]
            if cur_r2 > best_val_r2:
                best_val_r2 = cur_r2
                save_dir = os.path.join(wandb_run.dir, str(fold_idx))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{best_val_r2:.3f}.pth"))
                print(f"New best model saved! (R2: {best_val_r2:.4f})")
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
    df = load_data(config["data_folder"])
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

        v2.Resize((768, 768), antialias=True),

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
    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((768, 768), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    train_idxs, val_idxs = group_k_fold(df)

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

    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--predict_total", action="store_true")
    parser.add_argument("--predict_gdm", action="store_true")
    parser.add_argument("--predict_height", action="store_true")
    parser.add_argument("--predict_has_clover", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=1024)

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
    if not torch.isclose(torch.tensor(sum(coeffs)), torch.tensor(1.0)):
        raise ValueError(f"Loss coefficients must sum to 1. Current sum: {sum(coeffs)}")

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
    main(config, args.mode)

