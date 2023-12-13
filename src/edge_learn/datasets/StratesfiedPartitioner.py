from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class StratesfiedPartitioner:
    """
    Class partition data in i.i.d
    """

    def __init__(self, data, labels, sizes=[1.0], seed=1234):
        self.data = data
        self.labels = labels
        self.partitions = []
        np.random.seed(seed)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=None, random_state=seed)
        for frac in sizes:
            for train_index, test_index in sss.split(self.data, self.labels):
                self.partitions.append(train_index[: int(frac * len(train_index))])
                self.labels = self.labels[test_index]
                self.data = self.data[test_index]

    def use(self, index):
        return self.partitions[index]
