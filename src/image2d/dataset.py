import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FrameClassDataset(Dataset):
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
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        label_id = int(row["label_id"])
        label_name = row["label_name"]

        return {
            "image": img,
            "label_id": label_id,
            "label_name": label_name,
        }