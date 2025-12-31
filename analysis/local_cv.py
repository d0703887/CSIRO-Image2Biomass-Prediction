from model.DinoV3GatingMultiScale import DinoV3GatingMultiScale
from dataset import CSIROMultiScaleDataset
from utils.utils import load_CSIRO, CSIRO_group_k_fold
import torchvision.transforms.v2 as v2
import torch

if __name__ == '__main__':
    config = {
        # Model config
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "hidden_dim": 128,
        "predict_height": False
    }

    model = DinoV3GatingMultiScale(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        predict_height=config["predict_height"]
    )
    model.to("cuda")
    model.eval()

    df = load_CSIRO("../data/CSIRO")
    train_transforms = v2.Compose([
        # Normalization
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])
    val_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    train_idxs, val_idxs = CSIRO_group_k_fold(df)

    cnt = 0
    for i in range(6):
        cnt += len(val_idxs[i])

