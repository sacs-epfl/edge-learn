import logging

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

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
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def load_trainset(self):
        trainset = torchvision.datasets.ImageNet(
            root=self.train_dir, split="train", transform=self.transform
        )
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
        self.testset = torchvision.datasets.ImageNet(
            root=self.test_dir, split="val", transform=self.transform
        )

    def get_trainset(self, batch_size=1, shuffle=False):
        if self.__training__:
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        raise RuntimeError("Train set not initialised!")

    def get_testset(self):
        if self.__testing__:
            return DataLoader(self.testset, batch_size=self.test_batch_size)
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


class ResNet18(Model):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        logging.debug(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet8(Model):
#     def __init__(self, num_classes=10):
#         super(ResNet8, self).__init__()
#         block = BasicBlock
#         num_blocks = [1, 1, 1]
#         self.num_classes = num_classes
#         self.in_planes = 128

#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
#         self.linear1 = nn.Linear(2048, num_classes)
#         self.linear2 = nn.Linear(2048, num_classes)
#         self.emb = nn.Embedding(num_classes, num_classes)
#         self.emb.weight = nn.Parameter(torch.eye(num_classes))

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)  # b*128*32*32
#         out = self.layer2(out)  # b*256*16*16
#         out = self.layer3(out)  # b*512*8*8
#         self.inner = out
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)

#         self.flatten_feat = out  # b*2048
#         out = self.linear1(out)
#         return out

#     def get_attentions(self):
#         inner_copy = self.inner.detach().clone()  # b*512*8*8
#         inner_copy.requires_grad = True
#         out = F.avg_pool2d(inner_copy, 4)  # b*512*2*2
#         out = out.view(out.size(0), -1)  # b*2048
#         out = self.linear1(out)  # b*num_classes
#         losses = out.sum(dim=0)  # num_classes
#         cams = []
#         # import ipdb;ipdb.set_trace()
#         # assert losses.shape ==self.num_classes
#         for n in range(self.num_classes):
#             loss = losses[n]
#             self.zero_grad()
#             if n < self.num_classes - 1:
#                 loss.backward(retain_graph=True)
#             else:
#                 loss.backward()
#             grads_val = inner_copy.grad
#             weights = grads_val.mean(dim=(2, 3), keepdim=True)  # b*512*1*1
#             cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))  # b*8*8
#         atts = torch.stack(cams, dim=1)
#         return atts
