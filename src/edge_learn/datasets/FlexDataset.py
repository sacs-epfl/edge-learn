import logging

from decentralizepy.datasets.Dataset import Dataset

from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch


class InnerDataset(TorchDataset):
    def __init__(self):
        self.data = []
        self.target = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def append(self, batch):
        self.data.extend(batch["data"])
        self.target.extend(batch["target"])


class FlexDataset(Dataset):
    def __init__(self):
        self.inner_dataset = InnerDataset()

    def add_batch(self, batch):
        self.inner_dataset.append(batch)

    def get_trainset(self, batch_size):
        if len(self.inner_dataset) == 0:
            return None
        return DataLoader(self.inner_dataset, batch_size=batch_size, shuffle=True)
