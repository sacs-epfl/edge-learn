import logging

import torch
import json
import copy

from decentralizepy import utils


class Training:
    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        gpu_mapping_filepath="",
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = mapping.get_uid(self.rank, self.machine_id)
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

        logging.debug("My uid: " + str(self.uid))

        logging.debug("Gpu filepath: " + gpu_mapping_filepath)
        with open(gpu_mapping_filepath, "r") as f:
            self.gpu_mapping = json.load(f)

        self.gpus_to_use = self.gpu_mapping.get(str(self.uid), [])
        logging.debug("Using gpus: " + ", ".join(map(str, self.gpus_to_use)))

    def reset_optimizer(self, optimizer):
        self.optimizer = optimizer

    def eval_loss(self, dataset):
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    # One training step on minibatch
    def trainstep(self, data, target):
        # Ensure data and target are on CPU for easy splitting
        data, target = data.cpu(), target.cpu()

        # Split data and target based on the number of GPUs to use
        data_splits = torch.split(data, len(data) // len(self.gpus_to_use))
        target_splits = torch.split(target, len(target) // len(self.gpus_to_use))

        # Create a list to hold gradients from each GPU
        gradients = []

        primaryModelGpu = self.model.to(f"cuda:{self.gpus_to_use[0]}")

        for gpu, (data_split, target_split) in zip(
            self.gpus_to_use, zip(data_splits, target_splits)
        ):
            device = f"cuda:{gpu}"
            # Clone the model and optimizer for this GPU
            if gpu == self.gpus_to_use[0]:
                model_gpu = primaryModelGpu
            else:
                model_gpu = copy.deepcopy(primaryModelGpu).to(device)

            # Move data and target split to this GPU
            data_split, target_split = data_split.to(device), target_split.to(device)

            model_gpu.zero_grad()
            output = model_gpu(data_split)
            loss_val = self.loss(output, target_split)
            loss_val.backward()

            # Store gradients
            gradients.append(
                {name: param.grad for name, param in model_gpu.named_parameters()}
            )

        # Average the gradients across GPUs
        avg_gradients = {}
        for name in gradients[0].keys():
            # Stack gradients from all GPUs together
            stacked_grads = torch.stack(
                [grad[name].to(f"cuda:{self.gpus_to_use[0]}") for grad in gradients]
            )
            # Compute the average gradient
            avg_gradients[name] = stacked_grads.mean(dim=0)

        # Update the main model's gradients with the averaged gradients
        for name, param in self.model.named_parameters():
            param.grad = avg_gradients[name]

        # Optimize the main model
        self.optimizer.step()

        return loss_val.item()

    # One training iteration entire dataset
    def train_full(self, dataset):
        for epoch in range(self.rounds):
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                logging.debug(
                    "Starting minibatch {} with num_samples: {}".format(
                        count, len(data)
                    )
                )
                logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    # One training iteration
    def train(self, dataset):
        self.model.train()

        if self.full_epochs:
            self.train_full(dataset)
        else:
            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            while count < self.rounds:
                for data, target in trainset:
                    iter_loss += self.trainstep(data, target)
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break