from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from torch.utils.data import Dataset


class StratesfiedPartitioner:
    """
    Class partition data in i.i.d
    """

    def __init__(self, dataset, sizes=[1.0], seed=1234, num_classes=100):
        class_split = len(dataset) // num_classes
        class_start_idxs = [class_split * i for i in range(0, num_classes)]

        self.dataset = dataset
        labels = self.get_all_labels()

        if sum(sizes) != 1.0:
            raise ValueError("Sum of sizes must be 1.0")

        self.partitions = []
        np.random.seed(seed)

        dataset_indices = list(range(len(dataset)))

        sss = StratifiedShuffleSplit(
            n_splits=len(sizes), test_size=None, random_state=seed
        )

        stratisfied_indices = [[] for _ in sizes]

        for train_idx, _ in sss.split(dataset_indices, labels):
            for i, frac in enumerate(sizes):
                if i == 0:
                    start = 0
                else:
                    start = int(sum(sizes[: i - 1]) * len(train_idx))
                end = int(sum(sizes[:i]) * len(train_idx))
                stratisfied_indices[i].extend(train_idx[start:end])

        self.partitions = [
            IndexDataset(self.dataset, part) for part in stratisfied_indices
        ]

    def get_all_labels(self):
        labels = []
        for _, label in self.dataset:
            labels.append(label)
        return labels

    def use(self, index):
        return self.partitions[index]


class IndexDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
