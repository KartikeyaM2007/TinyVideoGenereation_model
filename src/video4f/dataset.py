import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class FrameQuadDataset(Dataset):
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

        imgs = []
        for key in ["frame1_path", "frame2_path", "frame3_path", "frame4_path"]:
            img = Image.open(row[key]).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        x = torch.cat(imgs, dim=0)  # (12, H, W)

        return {
            "quad": x,
            "label_id": int(row["label_id"]),
            "label_name": row["label_name"],
        }