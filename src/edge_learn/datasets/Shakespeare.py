import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model
from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.enums.LearningMode import LearningMode

VOCAB = list(
    "dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#'/37;?bfjnrvzBFJNRVZ\"&*.26:\naeimquyAEIMQUY]!%)-159\r{{}}<>"
)
VOCAB_LEN = len(VOCAB)
# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(VOCAB)}
idx2char = np.array(VOCAB)

EMBEDDING_DIM = 8
HIDDEN_DIM = 256
NUM_CLASSES = VOCAB_LEN
NUM_LAYERS = 2
SEQ_LENGTH = 80


class Shakespeare(Dataset):
    """
    Class for the Shakespeare dataset
    --  Based on https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
    """

    def __read_file__(self, file_path):
        """
        Read data from the given json file

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        tuple
            (users, num_samples, data)

        """
        with open(file_path, "r") as inf:
            client_data = json.load(inf)
        return (
            client_data["users"],
            client_data["num_samples"],
            client_data["user_data"],
        )

    def __read_dir__(self, data_dir):
        """
        Function to aggregate data across all JSON files in given data_dir

        Parameters
        ----------
        data_dir : str
            Path to the folder containing the data files

        Returns
        -------
        3-tuple
            A tuple containing list of users, number of samples per client,
            and the data items per client

        """
        users = []
        num_samples = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            u, n, d = self.__read_file__(file_path)
            users.extend(u)
            num_samples.extend(n)
            data.update(d)
        return users, num_samples, data

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        if not self.mapping.does_uid_generate_data(
            self.mapping.get_uid(self.rank, self.machine_id)
        ):
            self.train_x = []
            self.train_y = []
            return

        clients, num_samples, train_data = self.__read_dir__(self.train_dir)
        duid = self.mapping.get_duid_from_uid(
            self.mapping.get_uid(self.rank, self.machine_id)
        )
        assert clients.__len__() > duid
        print("duid: ", duid)
        print("tostring: ", str(duid))
        print("data_y: ", train_data[str(duid)]["y"])
        self.train_y = np.array(
            self.process(train_data[str(duid)]["y"]), dtype=np.dtype("int64")
        ).reshape(-1)
        self.train_x = np.array(
            self.process(train_data[str(duid)]["x"]), dtype=np.dtype("int64")
        )
        logging.info("train_x.shape: %s", str(self.train_x.shape))
        logging.info("train_y.shape: %s", str(self.train_y.shape))
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[0] > 0

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        if not self.mapping.does_uid_test_data(
            self.mapping.get_uid(self.rank, self.machine_id)
        ):
            self.test_x = []
            self.test_y = []
            return

        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            test_x.extend(self.process(test_data["x"]))
            test_y.extend(self.process(test_data["y"]))
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        self.test_x = np.array(test_x, dtype=np.dtype("int64"))
        logging.info("test_x.shape: %s", str(self.test_x.shape))
        logging.info("test_y.shape: %s", str(self.test_y.shape))
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        learning_mode: LearningMode,
        train=True,
        test=True,
        random_seed: int = 1234,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1024,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        random_seed : int, optional
            Random seed for the dataset. Default value is 1234
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str, optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            random_seed,
            only_local,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )
        self.__training__ = train
        self.__testing__ = test

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def process(self, x):
        output = list(
            map(lambda sentences: list(map(lambda c: char2idx[c], list(sentences))), x)
        )
        return output

    def get_client_ids(self):
        """
        Function to retrieve all the clients of the current process

        Returns
        -------
        list(str)
            A list of strings of the client ids.

        """
        return self.clients

    def get_client_id(self, i):
        """
        Function to get the client id of the ith sample

        Parameters
        ----------
        i : int
            Index of the sample

        Returns
        -------
        str
            Client ID

        Raises
        ------
        IndexError
            If the sample index is out of bounds

        """
        lb = 0
        for j in range(len(self.clients)):
            if i < lb + self.num_samples[j]:
                return self.clients[j]

        raise IndexError("i is out of bounds!")

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        if self.__training__:
            return DataLoader(
                Data(self.train_x, self.train_y), batch_size=batch_size, shuffle=shuffle
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            thirstiest = torch.arange(0, self.test_x.shape[0], 30)
            return DataLoader(
                Data(self.test_x[thirstiest], self.test_y[thirstiest]),
                batch_size=self.test_batch_size,
            )
        raise RuntimeError("Test set not initialized!")

    def test(self, model, loss):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple(float, float)

        """
        model.eval()
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                outputs = model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


class LSTM(Model):
    """
    Class for a RNN Model for Sent140

    """

    def __init__(self):
        """
        Constructor. Instantiates the RNN Model to predict the next word of a sequence of word.
        Based on the TensorFlow model found here: https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        """
        super().__init__()

        # input_length does not exist
        self.embedding = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, num_layers=NUM_LAYERS
        )
        # activation function is added in the forward pass
        # Note: the tensorflow implementation did not use any activation function in this step?
        # should I use one.
        self.l1 = nn.Linear(HIDDEN_DIM * SEQ_LENGTH, VOCAB_LEN)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = F.relu(x.reshape((-1, HIDDEN_DIM * SEQ_LENGTH)))
        x = self.l1(x)
        return x
