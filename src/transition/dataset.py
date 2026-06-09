import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FrameTransitionDataset(Dataset):
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

        frame1 = Image.open(row["frame1_path"]).convert("RGB")
        frame2 = Image.open(row["frame2_path"]).convert("RGB")

        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)

        return {
            "current_frame": frame1,
            "next_frame": frame2,
            "label_id": int(row["label_id"]),
            "label_name": row["label_name"],
        }