import torchvision
from torchvision.transforms import v2
import torch
import pandas as pd
import os
from sklearn.model_selection import GroupKFold


def make_transform(
        resize_size: int = 768,
):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    # Basic Transform
    transform = [to_tensor, resize, to_float, normalize]

    # Data Augmentation
    # transform.extend([
    #
    # ])
    return v2.Compose(transform)


def load_data(data_folder: str):
    # TODO: change data path
    df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    pivoted_df = df.pivot_table(
        index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
        columns="target_name",
        values="target").reset_index()
    return pivoted_df


def group_k_fold(df: pd.DataFrame):
    #groups = df["State"].astype(str) + "_" + df["Sampling_Date"].astype(str)
    groups = df["Sampling_Date"]
    gkf = GroupKFold(n_splits=6)
    train_idxs = []
    val_idxs = []
    for train_idx, val_idx in gkf.split(df, groups, groups):
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)

    return train_idxs, val_idxs