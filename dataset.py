import pandas as pd
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as v2

class CSIRODataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            df: pd.DataFrame,
            input_h,
            input_w,
            split_img = True,
            is_train = True,
    ):
        self.data_folder = data_folder
        self.df = df
        self.split_img = split_img
        self.is_train = is_train
        self.input_h = input_h
        self.input_w = input_w


        self.geometric_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.Identity(),
                v2.RandomRotation(degrees=(90, 90), expand=False),
                v2.RandomRotation(degrees=(180, 180), expand=False),
                v2.RandomRotation(degrees=(270, 270), expand=False)
            ]),
        ])

        self.resize = v2.Resize((input_h, input_w), antialias=True)

        # self.color_transform = v2.Compose([
        #     v2.RandomApply([
        #         v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
        #     ], p=0.5),  # High probability!
        #     v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
        # ])

        self.normalize_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

        self.img_paths = df["image_path"].tolist()
        self.species = df["Species"].tolist()
        self.data_values = df[[
            "Height_Ave_cm", "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"
        ]].to_dict('records')

        self.bad_images = {
            "Dry_Green_g": [
                "ID1139918758.jpg",
                "ID1337107565.jpg",
                "ID1403107574.jpg",
                "ID1761544403.jpg",
                "ID40849327.jpg",
                "ID473494649.jpg",
                "ID681680726.jpg",
                "ID697718693.jpg",
            ],
            "Dry_Clover_g": [
                "ID1403107574.jpg"
            ],
            "Dry_Dead_g": [
                "ID473494649.jpg",
                "ID661372352.jpg",
                "ID1403107574.jpg",
                "ID1444674500.jpg",
                "ID1761544403.jpg",
                "ID230058600.jpg"
            ]
        }

    def __len__(self):
        return len(self.df)

    def split_left_and_right(self, image):
        _, _, w = image.shape
        center = w // 2
        left_img = image[:, :, :center]
        right_img = image[:, :, center:]
        return left_img, right_img

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.img_paths[idx])
        img = read_image(img_path)
        img_basename = os.path.basename(img_path)

        gate_names = ["green", "clover", "dead"]
        masks = []

        for name in gate_names:
            path = os.path.join(self.data_folder, f"pseudo_gates/{name}", img_basename)
            if os.path.exists(path):
                # Read, convert to binary 0/1 float, ensure 1 channel
                m = (read_image(path) > 0).float()
            else:
                # Create empty mask with same H,W as image
                m = -torch.ones((1, 48, 96), dtype=torch.float32)
            masks.append(m)

        mask_tensor = torch.cat(masks, dim=0)

        if self.split_img:
            left_img, right_img = self.split_left_and_right(img)
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            input_img = torch.cat([left_img, right_img]) # (2, C, H, W)
        else:
            img = v2.ToImage()(img)
            if self.is_train:
                img, mask_tensor = self.geometric_transform(img, mask_tensor)
                img = self.resize(img)
                #img = self.color_transform(img)
            else:
                img = self.resize(img)

            img = self.normalize_transform(img)
            input_img = img

        row = self.data_values[idx]
        data_dict =  {
            "Input_Img": input_img,
            "Height_Ave_cm": torch.tensor(row["Height_Ave_cm"], dtype=torch.float32),
            "Dry_Green_g": torch.tensor(row["Dry_Green_g"], dtype=torch.float32) if img_basename not in self.bad_images["Dry_Green_g"] else torch.tensor(-1, dtype=torch.float32),
            "Dry_Clover_g": torch.tensor(row["Dry_Clover_g"], dtype=torch.float32) if img_basename not in self.bad_images["Dry_Clover_g"] else torch.tensor(-1, dtype=torch.float32),
            "Dry_Dead_g": torch.tensor(row["Dry_Dead_g"], dtype=torch.float32) if img_basename not in self.bad_images["Dry_Dead_g"] else torch.tensor(-1, dtype=torch.float32),
            "Dry_Total_g": torch.tensor(row["Dry_Total_g"], dtype=torch.float32),
            "Dry_Green_g_Gate": mask_tensor[0].view(96 * 48, 1),
            "Dry_Clover_g_Gate": mask_tensor[1].view(96 * 48, 1),
            "Dry_Dead_g_Gate": mask_tensor[2].view(96 * 48, 1),
            "GDM_g": torch.tensor(row["GDM_g"], dtype=torch.float32),
            "image_path": img_path,
            "species": self.species[idx]
        }

        return data_dict

class CSIROMultiScaleDataset(Dataset):
    def __init__(
            self,
            data_folder: str,
            df: pd.DataFrame,
            h: int,
            w: int,
            transform = None,
            is_train: bool = True
    ):
        self.data_folder = data_folder
        self.df = df
        self.h = h
        self.w = w
        self.transform = transform
        self.is_train = is_train

        self.img_paths = df["image_path"].tolist()
        self.data_values = df[[
            "Height_Ave_cm", "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Total_g", "GDM_g"
        ]].to_dict('records')

        basic_transform = [v2.ToImage()]
        if self.is_train:
            basic_transform.extend([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice([
                    v2.Identity(),
                    v2.RandomRotation(degrees=(90, 90), expand=False),
                    v2.RandomRotation(degrees=(180, 180), expand=False),
                    v2.RandomRotation(degrees=(270, 270), expand=False)
                ]),
                # Color
                v2.RandomApply([
                    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
                ], p=0.5),  # High probability!
                v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                # Blur
                v2.RandomApply([v2.GaussianBlur(kernel_size=(11, 11), )], p=0.3),
            ])

        self.basic_transform = v2.Compose(basic_transform)

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
        left_img = self.basic_transform(left_img)
        right_img = self.basic_transform(right_img)
        high_res_left = v2.functional.resize(left_img, size=[self.h, self.w], antialias=True)
        low_res_left = v2.functional.resize(left_img, size=[self.h // 2, self.w // 2], antialias=True)
        high_res_right = v2.functional.resize(right_img, size=[self.h, self.w], antialias=True)
        low_res_right = v2.functional.resize(right_img, size=[self.h // 2, self.w // 2], antialias=True)

        if self.transform:
            high_res_left = self.transform(high_res_left)
            low_res_left = self.transform(low_res_left)
            high_res_right = self.transform(high_res_right)
            low_res_right = self.transform(low_res_right)

        high_res_input = torch.cat([high_res_left, high_res_right])
        low_res_input = torch.cat([low_res_left, low_res_right])
        row = self.data_values[idx]
        return {
            "HR_Input_Img": high_res_input,
            "LR_Input_Img": low_res_input,
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

