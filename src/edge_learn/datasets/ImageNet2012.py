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

NUM_CLASSES = 1000


class ImageNet2012(Dataset):
    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
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
        c_len = len(trainset)

        if self.sizes == None:
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - frac * self.num_partitions
            logging.debug("Size fractions: {}".format(self.sizes))

        self.trainset = DataPartitioner(
            trainset, sizes=self.sizes, seed=self.random_seed
        ).use(self.dataset_id)

    def load_testset(self):
        logging.info("Starting to load the testset...")

        # Load dataset without transformations to quickly get labels=
        full_testset = torchvision.datasets.ImageNet(root=self.test_dir, split="val")
        logging.info("Full testset loaded.")

        # Group image indices by category
        category_indices = {}
        for idx, (_, label) in enumerate(full_testset.imgs):
            if label not in category_indices:
                category_indices[label] = []
            category_indices[label].append(idx)
        logging.info("Grouped image indices by category.")

        # Select one image from each category
        selected_indices = []
        for label, indices in category_indices.items():
            selected_indices.extend(indices[:1])

        # Apply transformations to selected images
        transformed_images = [
            self.transform(full_testset[i][0]) for i in selected_indices
        ]
        transformed_labels = [full_testset[i][1] for i in selected_indices]

        # Combine the transformed images and labels to create a custom dataset
        self.testset = list(zip(transformed_images, transformed_labels))

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


import torch
import torchvision.models as models


class ResNet18(models.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()

        # Step 1: Create the ResNet-18 model without any weights
        self.resnet18 = models.resnet18(weights=None)

        # Step 2: Store the original conv1 layer
        original_conv1 = self.resnet18.conv1

        if pretrained:
            # Step 3: Load the CIFAR-100 weights into the model (excluding the fully connected layer)
            state_dict = torch.load(
                "datasets/weights/resnet18_CIFAR100.bin", map_location="cpu"
            )
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            self.resnet18.load_state_dict(state_dict, strict=False)

            # Step 4: Restore the original conv1 layer back to the model
            self.resnet18.conv1 = original_conv1

            # Modify the last fully connected layer for 1000 classes of ImageNet
            fc_in_features = self.resnet18.fc.in_features
            self.resnet18.fc = torch.nn.Linear(fc_in_features, 1000)

    def forward(self, x):
        return self.resnet18(x)
