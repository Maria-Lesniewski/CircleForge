import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CircleData(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir="",
        target_columns=None,
        transform=None,
    ):
        """
        PyTorch Dataset for annotated .npy vector files.

        Args:
            csv_file:
                Path to CSV annotation file.
            root_dir:
                Optional prefix directory for .npy files.
            target_columns:
                List of CSV columns to use as labels.
                Example: ["ncircles"]
                or ["ncircles", "nbig", "nsmall"]
            transform:
                Optional transform applied to input vectors.
        """

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_columns = target_columns or ["ncircles"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        row = self.annotations.iloc[idx]

        # Load vector file
        vector_path = os.path.join(
            self.root_dir,
            row["access_filename"]
        )

        x = np.load(vector_path)

        # Convert numpy array -> torch tensor
        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        if self.transform:
            x = self.transform(x)

        # Extract labels
        y = row[self.target_columns].values.astype(np.float32)

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y
