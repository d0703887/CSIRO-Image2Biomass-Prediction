import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pprint

def main():
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.pivot_table(
        index=["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"],
        columns="target_name", values="target").reset_index()

    species = sorted(train_df["Species"].map(lambda x: x.lower()).unique().tolist())

    for specie in species:
        tmp_df = train_df[train_df["Species"].str.contains(specie, case=False)]
        tmp_df = tmp_df.sort_values(by=["Height_Ave_cm"])
        for idx, row in tmp_df.iterrows():
            image_path = row["image_path"]
            pprint.pp(row)
            plt.imshow(Image.open(os.path.join("data", image_path)))
            plt.axis("off")
            plt.waitforbuttonpress()

if __name__ == '__main__':
    main()