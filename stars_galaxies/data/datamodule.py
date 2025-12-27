from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GalaxyStarDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, num_workers: int, image_size: int
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        root = self.data_dir / "raw"

        self.train_dataset = datasets.ImageFolder(root / "train", transform=transform)
        self.val_dataset = datasets.ImageFolder(
            root / "validation", transform=transform
        )
        self.test_dataset = datasets.ImageFolder(root / "test", transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
