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
        self.data.append(batch['data'])
        self.target.append(batch['target'])

class FlexDataset(Dataset):
    def __init__(self):
        self.inner_dataset = InnerDataset()

    def add_batch(self, batch):
        self.inner_dataset.append(batch)

    def get_trainset(self):
        return DataLoader(self.inner_dataset, batch_size=1)
