import logging

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import (
    DataPartitioner,
    DirichletDataPartitioner,
    KShardDataPartitioner,
    SimpleDataPartitioner,
)
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model
from edge_learn.enums.LearningMode import LearningMode

NUM_CLASSES = 10

class CIFAR10(Dataset):
    def load_trainset(self):
        logging.info("Loading training set.")
        trainset = torchvision.datasets.CIFAR10(
            root=self.train_dir, train=True, download=True, transform=self.transform
        )
        if self.learning_mode == LearningMode.BASELINE:
            self.trainset = trainset 
            return
        
        c_len = len(trainset)
        num_clients = self.mapping.get_num_clients()

        e = c_len // num_clients
        client_duid = self.mapping.get_duid_from_machine_and_rank(self.machine_id, self.rank)

        start_idx = e * client_duid
        end_idx = start_idx + e

        self.trainset = [trainset[i] for i in range(start_idx, end_idx)]

    def load_testset(self):
        logging.info("Loading testing set.")

        self.testset = torchvision.datasets.CIFAR10(
            root=self.test_dir, train=False, download=True, transform=self.transform
        )

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        learning_mode: LearningMode,
        train: bool,
        test: bool,
        random_seed: int = 1234,
        train_dir: str = "",
        test_dir: str = "",
        test_batch_size: int =1024,
    ):
        super().__init__(
            rank,
            machine_id,
            mapping,
            random_seed,
            False,
            train_dir,
            test_dir,
            "",
            test_batch_size,
        )

        self.__training__ = train
        self.__testing__ = test

        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.learning_mode = learning_mode

        self.num_classes = NUM_CLASSES

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()


    def get_trainset(self, batch_size=1, shuffle=False):
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        if self.__testing__:
            return DataLoader(self.testset, batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def test(self, model, loss):
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


class CNN(Model):

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        # 1.6 million params
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(Model):
    """
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(64 * 4 * 4, NUM_CLASSES)

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
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# Taken from: https://github.com/gong-xuan/FedKD/blob/master/models/resnet8.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8(Model):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        block = BasicBlock
        num_blocks = [1, 1, 1]
        self.num_classes = num_classes
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(2048, num_classes)
        self.linear2 = nn.Linear(2048, num_classes)
        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # b*128*32*32
        out = self.layer2(out)  # b*256*16*16
        out = self.layer3(out)  # b*512*8*8
        self.inner = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        self.flatten_feat = out  # b*2048
        out = self.linear1(out)
        return out

    def get_attentions(self):
        inner_copy = self.inner.detach().clone()  # b*512*8*8
        inner_copy.requires_grad = True
        out = F.avg_pool2d(inner_copy, 4)  # b*512*2*2
        out = out.view(out.size(0), -1)  # b*2048
        out = self.linear1(out)  # b*num_classes
        losses = out.sum(dim=0)  # num_classes
        cams = []
        # import ipdb;ipdb.set_trace()
        # assert losses.shape ==self.num_classes
        for n in range(self.num_classes):
            loss = losses[n]
            self.zero_grad()
            if n < self.num_classes - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grads_val = inner_copy.grad
            weights = grads_val.mean(dim=(2, 3), keepdim=True)  # b*512*1*1
            cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))  # b*8*8
        atts = torch.stack(cams, dim=1)
        return atts
