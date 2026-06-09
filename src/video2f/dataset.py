import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class FramePairDataset(Dataset):
    def __init__(self, metadata_csv: str, resolution: int = 64):
        self.df = pd.read_csv(metadata_csv)

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1 = Image.open(row["frame1_path"]).convert("RGB")
        img2 = Image.open(row["frame2_path"]).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        x = torch.cat([img1, img2], dim=0)  # (6, H, W)

        return {
            "pair": x,
            "label_id": int(row["label_id"]),
            "label_name": row["label_name"],
        }