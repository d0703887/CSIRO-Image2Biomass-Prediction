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

    bad_images = [
        # Clover
        # "train/ID2002797732.jpg", "train/ID230058600.jpg", "train/ID681680726.jpg", "train/ID1761544403.jpg", "train/ID1403107574.jpg", "train/ID443091455.jpg", "train/ID1717006117.jpg", "train/ID572336285.jpg",
        # Green
        # "train/ID1337107565.jpg", "train/ID1139918758.jpg", "train/ID40849327.jpg",
    ]
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
    import pandas as pd


    def inspect_fold_months(df, train_idxs, val_idxs):
        """
        Prints which months are present in Train vs Val for each fold
        and checks for month leakage.
        """
        # 1. Create a Month-Year column (e.g., "2015-10")
        # We use temporary columns so we don't affect the main df permanently
        temp_df = df.copy()
        temp_df['dt'] = pd.to_datetime(temp_df['Sampling_Date'], format="%Y/%m/%d")
        temp_df['month_str'] = temp_df['dt'].dt.strftime('%m')

        print(f"{'Fold':<5} | {'Val Months (Count)':<30} | {'Unique to Val (Not in Train)'}")
        print("-" * 80)

        for fold, (train_idx, val_idx) in enumerate(zip(train_idxs, val_idxs)):
            # Get the list of months for this fold's train/val sets
            train_months = set(temp_df.iloc[train_idx]['month_str'])
            val_months = set(temp_df.iloc[val_idx]['month_str'])

            # Calculate months that are in Val but NOT in Train
            unique_to_val = val_months - train_months

            # Formatting for display
            val_disp = f"{len(val_months)} months"
            unique_disp = str(sorted(list(unique_to_val))) if unique_to_val else "None (Month exists in Train)"

            print(f"{fold + 1:<5} | {val_disp:<30} | {unique_disp}")

            # Optional: Detailed debug for the first fold to see exactly what's happening
            if fold == 0 and not unique_to_val:
                print(f"\n[Info] In Fold 1, dates are split by group, but months overlap.")
                print(f"       Example: '2015-10' contains {len(temp_df[temp_df['month_str'] == '2015-10'])} images.")
                print("-" * 80)

    # --- Usage ---
    # Assuming you have loaded your df and run your split function:
    df = load_CSIRO("../data/CSIRO")
    train_idxs, val_idxs = CSIRO_stratified_group_k_fold(df)
    inspect_fold_months(df, train_idxs, val_idxs)