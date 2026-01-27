import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys

# --- Assumptions ---
# This code assumes the classes (DinoV3ViT, CSIRODataset, etc.)
# and imports (StratifiedGroupKFold, etc.) are already defined or imported in your environment.
from utils.utils import CSIRO_stratified_group_k_fold, load_CSIRO
from model.DinoV3ViT import DinoV3ViT
from dataset import CSIRODataset


def get_valid_global_means(df, bad_images_dict):
    """
    Computes the global mean for each target over the entire dataset,
    excluding the specific bad images defined in the Dataset class.
    """
    targets = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"]
    global_means = {}

    for t in targets:
        values = df[t].values
        image_names = df["image_path"].apply(os.path.basename).values

        if t in bad_images_dict:
            bad_set = set(bad_images_dict[t])
            is_valid = [name not in bad_set for name in image_names]
            valid_values = values[is_valid]
        else:
            valid_values = values

        global_means[t] = torch.tensor(valid_values.mean(), dtype=torch.float32, device='cuda')

    return global_means


def run_cross_validation_inference(
        df: pd.DataFrame,
        data_folder: str,
        model_name: str,
        weight_paths_config: dict,
        predict_height: bool,
        input_h: int,
        input_w: int,
        hidden_dim: int,
        batch_size: int = 4,
        num_workers: int = 4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Data and Stratification
    train_idxs_list, val_idxs_list = CSIRO_stratified_group_k_fold(df)

    # Define bad images
    bad_images = {
        "Dry_Green_g": ["ID1139918758.jpg", "ID1337107565.jpg", "ID1403107574.jpg", "ID1761544403.jpg",
                        "ID40849327.jpg", "ID473494649.jpg", "ID681680726.jpg", "ID697718693.jpg"],
        "Dry_Clover_g": ["ID1403107574.jpg"],
        "Dry_Dead_g": ["ID473494649.jpg", "ID661372352.jpg", "ID1403107574.jpg", "ID1444674500.jpg", "ID1761544403.jpg",
                       "ID230058600.jpg"]
    }

    # 2. Compute Global Mean
    print("Computing global means...")
    global_means = get_valid_global_means(df, bad_images)
    print(f"Global Means: {global_means}")

    agg_preds = {k: [] for k in global_means.keys()}
    agg_targets = {k: [] for k in global_means.keys()}

    r2_coeff = {
        "Dry_Green_g": 0.1, "Dry_Clover_g": 0.1, "Dry_Dead_g": 0.1,
        "Dry_Total_g": 0.5, "GDM_g": 0.2
    }

    # 3. Loop over Folds
    for fold_idx, val_idxs in enumerate(val_idxs_list):
        fold_str = str(fold_idx)
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Check if weights exist for this fold
        if fold_str not in weight_paths_config:
            print(f"Warning: No weights found for Fold {fold_idx} in config. Skipping...")
            continue

        fold_preds = {k: [] for k in global_means.keys()}
        fold_targets = {k: [] for k in global_means.keys()}

        # A. Create Validation Loader
        val_df = df.iloc[val_idxs].reset_index(drop=True)
        val_dataset = CSIRODataset(
            data_folder=data_folder,
            df=val_df,
            input_h=input_h,
            input_w=input_w,
            split_img=False,
            is_train=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # B. Load Model
        model = DinoV3ViT(
            model_name=model_name,
            hidden_dim=hidden_dim,
            training_mode="freeze_backbone",
            predict_height=predict_height,
            split_img=False
        )

        weight_path = weight_paths_config[fold_str]
        print(f"Loading weights from: {weight_path}")
        weights = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Inference"):
                imgs = batch["Input_Img"].to(device)

                pred_dict = model(imgs)
                batch_preds_map = {
                    "Dry_Green_g": pred_dict["Dry_Green_g"],
                    "Dry_Clover_g": pred_dict["Dry_Clover_g"],
                    "Dry_Dead_g": pred_dict["Dry_Dead_g"],
                    "Dry_Total_g": pred_dict["Dry_Total_g"],
                    "GDM_g": pred_dict["GDM_g"]
                }

                for t_name in global_means.keys():
                    raw_target = batch[t_name].to(device)
                    raw_pred = batch_preds_map[t_name]

                    valid_mask = raw_target != -1.0

                    if valid_mask.sum() > 0:
                        masked_preds = raw_pred[valid_mask]
                        masked_targets = raw_target[valid_mask]

                        agg_preds[t_name].append(masked_preds)
                        agg_targets[t_name].append(masked_targets)
                        fold_preds[t_name].append(masked_preds)
                        fold_targets[t_name].append(masked_targets)

        print(f"\n>>> Fold {fold_idx} Results:")
        fold_weighted_r2 = 0
        for t_name in global_means.keys():
            if len(fold_preds[t_name]) > 0:
                f_preds = torch.cat(fold_preds[t_name])
                f_targets = torch.cat(fold_targets[t_name])

                mse = torch.sum((f_targets - f_preds) ** 2)
                var = torch.sum((f_targets - global_means[t_name]) ** 2)

                r2_val = (1 - (mse / var)).item()
                fold_weighted_r2 += r2_coeff[t_name] * r2_val

                print(f"   {t_name:<15} | R2: {r2_val:.5f} | MSE: {mse / len(f_preds):.2f}")
            else:
                print(f"   {t_name:<15} | No valid samples in this fold.")

        print(f"   >>> Fold {fold_idx} Weighted R2: {fold_weighted_r2:.5f}")

        del model
        torch.cuda.empty_cache()

    # 4. Compute Final CV R2 Score
    print("\n\n====== Final Cross-Validation Results ======")
    final_scores = {"r2": 0}

    for t_name in global_means.keys():
        if len(agg_preds[t_name]) > 0:
            flat_preds = torch.cat(agg_preds[t_name])
            flat_targets = torch.cat(agg_targets[t_name])

            mse = torch.sum((flat_targets - flat_preds) ** 2)
            var = torch.sum((flat_targets - global_means[t_name]) ** 2)
            r2 = 1 - (mse / var)
            r2_val = r2.item()

            final_scores[f"{t_name}_r2"] = r2_val
            final_scores["r2"] += r2_coeff[t_name] * r2_val

            print(f"{t_name:<15} | R2: {r2_val:.5f} | MSE: {mse / len(flat_preds):.2f} | Count: {len(flat_preds)}")
        else:
            print(f"{t_name:<15} | No valid predictions found across all folds.")

    print("-" * 50)
    print(f"TOTAL WEIGHTED R2: {final_scores['r2']:.5f}")

    return final_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Validation Inference for DinoV3ViT")

    # --- Required Arguments (or strongly recommended) ---
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the CSIRO data directory"
    )
    parser.add_argument(
        "--weights_paths",
        type=str,
        nargs='+',
        required=True,
        help="List of 5 model weight paths for folds 0, 1, 2, 3, 4 (order matters)"
    )

    # --- Optional Arguments with Defaults ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size for the regression head"
    )
    parser.add_argument(
        "--input_h",
        type=int,
        default=768,
        help="Input image height"
    )
    parser.add_argument(
        "--input_w",
        type=int,
        default=1536,
        help="Input image width"
    )

    # Internal defaults (not requested to be exposed, but passed to function)
    # If you want to expose predict_height later, add: parser.add_argument("--predict_height", action="store_true")

    args = parser.parse_args()

    # Basic Validation
    if len(args.weights_paths) != 5:
        print(f"Error: Expected exactly 5 weight paths (one for each fold), but got {len(args.weights_paths)}.")
        sys.exit(1)

    # Convert list of weights to dictionary {"0": path0, "1": path1, ...}
    weight_paths_cfg = {str(i): path for i, path in enumerate(args.weights_paths)}

    print(f"Loading data from: {args.data_folder}")
    df = load_CSIRO(args.data_folder)

    scores = run_cross_validation_inference(
        df=df,
        data_folder=args.data_folder,
        model_name=args.model_name,
        weight_paths_config=weight_paths_cfg,
        predict_height=False,  # Hardcoded as not requested in arguments
        input_h=args.input_h,
        input_w=args.input_w,
        hidden_dim=args.hidden_dim
    )