import logging

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models

import torch.nn.functional as F
import torch.nn as nn


from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.models.Model import Model
from edge_learn.mappings.EdgeMapping import EdgeMapping
from decentralizepy.datasets.Partitioner import DataPartitioner
from edge_learn.enums.LearningMode import LearningMode
from edge_learn.datasets.StratesfiedPartitioner import StratesfiedPartitioner

NUM_CLASSES = 1000
# MAX is 50
TEST_IMAGES_PER_CATEGORY = 50

"""
Download the dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
You need to download Development Kit (Tasks 1 & 2), Training images (Tasks 1 & 2), 
    and Validatio images (all tasks)
Place all in the same directory, do not extract. 
Launch ./docker_run.sh [DIRECTORY OF IMAGENET]. It will parse all the files for you.
"""


class ImageNet2012(Dataset):
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

        # Online Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.__training__:
            logging.info("Loading trainset")
            self.load_trainset()

        if self.__testing__:
            logging.info("loading testset")
            self.load_testset()

    def load_trainset(self):
        logging.info("Starting to load the trainset...")
        trainset = torchvision.datasets.ImageNet(
            root=self.train_dir, split="train", transform=self.transform
        )
        logging.info("Full trainset loaded.")

        # data = []
        # labels = []
        # for img, label in trainset:
        #     data.append(img)
        #     labels.append(label)

        i: int = 0
        for _, label in trainset:
            i += 1
            logging.info(f"label for i {i} is {label}")
            if i == 2500:
                exit(0)

        c_len = len(trainset)

        if self.sizes == None:
            e = c_len // self.mapping.get_number_of_nodes_read_from_dataset()
            frac = e / c_len
            self.sizes = [frac] * self.mapping.get_number_of_nodes_read_from_dataset()
            self.sizes[-1] += 1.0 - sum(self.sizes)  # Give the last one the rest
            logging.debug("Size fractions: {}".format(self.sizes))

        my_duid = self.mapping.get_duid_from_uid(
            self.mapping.get_uid(self.rank, self.machine_id)
        )
        self.trainset = StratesfiedPartitioner(
            trainset, sizes=self.sizes, num_classes=NUM_CLASSES
        ).use(my_duid)

        logging.debug(f"Dataset partition size: {self.sizes[my_duid]}")

    def load_testset(self):
        logging.info("Starting to load the testset...")

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load dataset without transformations to quickly get labels=
        full_testset = torchvision.datasets.ImageNet(
            root=self.test_dir, split="val", transform=test_transform
        )
        logging.info("Full testset loaded.")
        self.testset = full_testset

    def get_trainset(self, batch_size=8, shuffle=True):
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Train set not initialised!")

    def get_testset(self):
        if self.__testing__:
            return DataLoader(
                self.testset, batch_size=self.test_batch_size, num_workers=8
            )
        raise RuntimeError("Test set not initialised!")


class ResNet50(Model):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.resnet50 = models.resnet50()

        fc_in_features = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(fc_in_features, NUM_CLASSES)

    def forward(self, x):
        return self.resnet50(x)


class ResNet50Pretrained(Model):
    def __init__(self):
        super(ResNet50Pretrained, self).__init__()

        self.resnet50 = models.resnet50()

        self.resnet50.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        fc_in_features = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(fc_in_features, 100)

        self.resnet50.load_state_dict(
            torch.load("./datasets/weights/resnet50_cifar100.bin")
        )

        self.resnet50.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet50.fc = torch.nn.Linear(fc_in_features, NUM_CLASSES)

    def forward(self, x):
        return self.resnet50(x)


class EfficientNetB0(Model):
    def __init__(self):
        super(EfficientNetB0, self).__init__()

        self.efficientnet_b0 = models.efficientnet_b0(pretrained=False)

        num_ftrs = self.efficientnet_b0.classifier[1].in_features
        self.efficientnet_b0.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(num_ftrs, NUM_CLASSES)
        )

    def forward(self, x):
        return self.efficientnet_b0(x)


class MobileNetV2Pretrained(Model):
    def __init__(self):
        super(MobileNetV2Pretrained, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(weights="DEFAULT")
        # num_ftrs = self.mobilenet_v2.classifier[1].in_features
        # self.mobilenet_v2.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2), nn.Linear(num_ftrs, NUM_CLASSES)
        # )

    def forward(self, x):
        return self.mobilenet_v2(x)


class MobileNetV2(Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2()

    def forward(self, x):
        return self.mobilenet_v2(x)
