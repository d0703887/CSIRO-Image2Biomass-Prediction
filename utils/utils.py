import torchvision
from torchvision.transforms import v2
import torch
import pandas as pd
import os
from sklearn.model_selection import GroupKFold, train_test_split, StratifiedGroupKFold

def load_CSIRO(data_folder: str):
    # TODO: change data path
    df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    pivoted_df = df.pivot_table(
        index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
        columns="target_name",
        values="target").reset_index()
    #bad_images = ["ID230058600.jpg", "ID1403107574.jpg", "ID1761544403.jpg", "ID681680726.jpg"]
    bad_images = ["train/ID230058600.jpg", "train/ID681680726.jpg", "train/ID1761544403.jpg", "train/ID1403107574.jpg", "train/ID443091455.jpg", "train/ID1717006117.jpg", "train/ID572336285.jpg", "train/"]
    pivoted_df = pivoted_df[~pivoted_df['image_path'].isin(bad_images)].reset_index(drop=True)
    return pivoted_df

def load_Grass(data_folder, df_name):
    df = pd.read_csv(os.path.join(data_folder, df_name), sep=";")
    df["Dry_Green_g"] = df["dry_grass"] + df["dry_weeds"]
    df.rename(columns={"image_file_name": "image_path", "dry_clover": "Dry_Clover_g"}, inplace=True)
    df = df[["image_path", "Dry_Green_g", "Dry_Clover_g"]]
    df["image_path"] = df["image_path"].map(
        lambda x: os.path.join(
            data_folder,
            "images",
            x
        )
    )
    df["source"] = "grass"

    train_df, val_df = train_test_split(df, test_size=0.2)
    return train_df, val_df

def load_Irish(data_folder, sub_folder, df_name):
    df = pd.read_csv(os.path.join(data_folder, sub_folder, df_name))
    df["Dry_Green_g"] = (df["Grass Dried"] * df["Herbage Mass (kg DM/ha)"] / 100 + df["Weeds Dried"] * df["Herbage Mass (kg DM/ha)"]  / 100) * 1000 * (0.25 / 10000)
    df["Dry_Clover_g"] = (df["Clover Dried"] * df["Herbage Mass (kg DM/ha)"] / 100) * 1000 * (0.25 / 10000)
    df.rename(columns={"Image Name": "image_path"}, inplace=True)
    df = df[["image_path", "Dry_Green_g", "Dry_Clover_g"]]
    df["image_path"] = df["image_path"].map(
        lambda x: os.path.join(
            data_folder,
            sub_folder,
            "images",
            x
        )
    )
    df["source"] = "irish"
    return df

def merge_Irish_Grass(grass_folder, irish_folder):
    # Grass
    grass_train_df, grass_val_df = load_Grass(os.path.join(grass_folder, "rectified_train"), "biomass_train_data.csv")

    # Irish
    # Camera
    irish_camera_train_df = load_Irish(irish_folder, "camera", "train.csv")
    irish_camera_val_df = load_Irish(irish_folder, "camera", "val.csv")
    irish_phone_train_df = load_Irish(irish_folder, "phone", "train.csv")
    irish_phone_val_df = load_Irish(irish_folder, "phone", "val.csv")

    # train_df = pd.concat([grass_train_df, irish_camera_train_df, irish_phone_train_df])
    # val_df = pd.concat([grass_val_df, irish_camera_val_df, irish_phone_val_df])
    train_df = pd.concat([grass_train_df,])
    val_df = pd.concat([grass_val_df,])

    # Logging
    print(f"[Grass] Train: {len(grass_train_df):,} | Val: {len(grass_val_df):,}")
    print(f"[Irish-Camera] Train: {len(irish_camera_train_df):,} | Val: {len(irish_camera_val_df):,}")
    print(f"[Irish-Phone]  Train: {len(irish_phone_train_df):,} | Val: {len(irish_phone_val_df):,}")
    print(f"{'=' * 40}")
    print(f"[Total Merged] Train: {len(train_df):,} | Val: {len(val_df):,}")
    print(f"{'=' * 40}")

    # SAFETY FIX: Reset index to ensure unique indices for data loaders
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def CSIRO_group_k_fold(df: pd.DataFrame):
    #groups = df["State"].astype(str) + "_" + df["Sampling_Date"].astype(str)
    groups = df["Sampling_Date"]
    gkf = GroupKFold(n_splits=6)
    train_idxs = []
    val_idxs = []
    for train_idx, val_idx in gkf.split(df, groups, groups):
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)

    return train_idxs, val_idxs


def CSIRO_stratified_group_k_fold(df: pd.DataFrame, n_splits: int = 5):
    # 1. Create bins for stratification
    df["stratify_bins"] = pd.qcut(df["Dry_Green_g"], q=10, labels=False, duplicates="drop")

    # 2. Define Groups (Sampling_Date)
    groups = df["Sampling_Date"]

    # 3. Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    train_idxs = []
    val_idxs = []

    # Note: We pass groups explicitly here
    for train_idx, val_idx in sgkf.split(X=df, y=df["stratify_bins"], groups=groups):
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)

    df.drop(columns=["stratify_bins"], inplace=True)
    return train_idxs, val_idxs

if __name__ == '__main__':
    df = pd.read_csv(os.path.join("../data/CSIRO", "train.csv"))
    pivoted_df = df.pivot_table(
        index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
        columns="target_name",
        values="target").reset_index()
    pivoted_df.to_csv("train_table.csv")