import logging

import torch
import json
import copy
from torch.nn import DataParallel

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = mapping.get_uid(self.rank, self.machine_id)

        logging.debug("My uid: " + str(self.uid))

        logging.debug("Gpu filepath: " + gpu_mapping_filepath)
        with open(gpu_mapping_filepath, "r") as f:
            self.gpu_mapping = json.load(f)

        self.gpus_to_use = self.gpu_mapping.get(str(self.uid), [])
        logging.debug("Using gpus: " + ", ".join(map(str, self.gpus_to_use)))

        if len(self.gpus_to_use) != 0:
            primary_device = f"cuda:{self.gpus_to_use[0]}"
            self.model = self.model.to(primary_device)
            self.model = DataParallel(self.model, device_ids=self.gpus_to_use)

    # Assuming self.model is already wrapped with DataParallel
    def trainstep(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)  # data can be on CPU
        loss_val = self.loss(
            output, target.to(output.device)
        )  # Move target to the same device as output
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def eval_loss(self, dataset):
        trainset = dataset.get_trainset(64)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = self.model(data)  # data can be on CPU
                loss_val = self.loss(
                    output, target.to(output.device)
                )  # Move target to the same device as output
                epoch_loss += loss_val.item()
                count += len(data)
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss
