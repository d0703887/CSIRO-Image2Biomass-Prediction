import pandas as pd
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

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
        self.data_values = df[[
            "Height_Ave_cm", "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"
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
            "Avg_Height": torch.tensor(row["Height_Ave_cm"], dtype=torch.float32),
            "Dry_Green_g": torch.tensor(row["Dry_Green_g"], dtype=torch.float32),
            "Dry_Clover_g": torch.tensor(row["Dry_Clover_g"], dtype=torch.float32),
            "Dry_Dead_g": torch.tensor(row["Dry_Dead_g"], dtype=torch.float32),
            "Dry_Total_g": torch.tensor(row["Dry_Total_g"], dtype=torch.float32),
            "GDM_g": torch.tensor(row["GDM_g"], dtype=torch.float32),
        }

class CombinedExternalDataset(Dataset):
    def __init__(
            self,
            df,
            transform_dict: dict,
    ):
        self.irish_transform = transform_dict["irish"]
        self.grass_transform = transform_dict["grass"]

        self.df = df

        self.img_paths = self.df["image_path"].tolist()
        self.sources = self.df["source"].tolist()
        self.data_values = self.df[["Dry_Green_g", "Dry_Clover_g"]].to_dict('records')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx])
        if self.sources[idx] == "irish":
            img = self.irish_transform(img)
        else:
            img = self.grass_transform(img)
        row = self.data_values[idx]
        return {
            "Input_Img": img,
            "Dry_Green_g": torch.tensor(row["Dry_Green_g"], dtype=torch.float32),
            "Dry_Clover_g": torch.tensor(row["Dry_Clover_g"], dtype=torch.float32),
        }

