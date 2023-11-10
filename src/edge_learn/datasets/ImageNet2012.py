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

NUM_CLASSES = 100
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

        # Filter the dataset to include only the first `NUM_CLASSES` classes
        indices = [
            idx
            for idx, (_, target) in enumerate(trainset.samples)
            if target < NUM_CLASSES
        ]
        logging.info(f"Number of elements in training dataset: {len(indices)}")
        trainset = torch.utils.data.Subset(trainset, indices)

        c_len = len(trainset)

        if self.sizes == None:
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - sum(self.sizes)
            logging.debug("Size fractions: {}".format(self.sizes))

        self.trainset = DataPartitioner(
            trainset, sizes=self.sizes, seed=self.random_seed
        ).use(self.dataset_id)

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

        # Group image indices by category
        category_indices = {}
        for idx, (_, label) in enumerate(full_testset.imgs):
            if label < NUM_CLASSES:
                if label not in category_indices:
                    category_indices[label] = []
                category_indices[label].append(idx)
        logging.info("Grouped image indices by category.")

        # Select up to TEST_IMAGES_PER_CATEGORY images from each of the first NUM_CLASSES categories
        selected_indices = []
        for indices in category_indices.values():
            selected_indices.extend(indices[:TEST_IMAGES_PER_CATEGORY])

        logging.info(f"Number of elements in testing dataset: {len(selected_indices)}")

        self.testset = torch.utils.data.Subset(full_testset, selected_indices)

    def get_trainset(self, batch_size=1, shuffle=False):
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Train set not initialised!")

    def get_testset(self):
        if self.__testing__:
            return DataLoader(
                self.testset, batch_size=self.test_batch_size, num_workers=8
            )
        raise RuntimeError("Test set not initialised!")

    def test(self, model, loss):
        model.eval()
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        # Get the device of the model
        device = next(model.parameters()).device

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                # Move the data to the same device as the model
                elems = elems.to(device)
                labels = labels.to(device)

                outputs = model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label.item()] += 1
                        total_correct += 1
                    total_pred[label.item()] += 1
                    total_predicted += 1

            logging.debug("Predicted on the test set")

            for key, value in enumerate(correct_pred):
                if total_pred[key] != 0:
                    accuracy = 100 * float(value) / total_pred[key]
                else:
                    accuracy = 100.0
                logging.debug(
                    "Accuracy for class {} is: {:.1f} %".format(key, accuracy)
                )

            accuracy = 100 * float(total_correct) / total_predicted
            loss_val = loss_val / count
            logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
            model.train()
            return accuracy, loss_val


class ResNet50(Model):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        fc_in_features = self.resnet50.fc.in_features
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


class MobileNetV2Custom(Model):
    def __init__(self, num_classes=100):
        super(MobileNetV2Custom, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=False)
        num_ftrs = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.mobilenet_v2(x)
