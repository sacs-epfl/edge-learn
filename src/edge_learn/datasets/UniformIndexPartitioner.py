from torch.utils.data import Dataset


class UniformIndexPartitioner:
    """
    Class partition data in i.i.d where dataset composed of classes
    grouped together
    """

    def __init__(self, dataset, sizes=[1.0], num_classes=100):
        self.dataset = dataset
        class_split = len(dataset) // num_classes

        self.partitions = [[] for _ in range(len(sizes))]

        for class_idx in range(num_classes):
            class_start_idx = class_split * class_idx
            for i, _ in enumerate(sizes):
                if i == 0:
                    start_idx = class_start_idx
                else:
                    start_idx = class_start_idx + int(class_split * sizes[:i])
                end_idx = class_start_idx + int(class_split * sizes[: i + 1])
                self.partitions[i].extend(range(start_idx, end_idx))

    def use(self, duid: int):
        return IndexDataset(self.dataset, self.partitions[duid])


class IndexDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
