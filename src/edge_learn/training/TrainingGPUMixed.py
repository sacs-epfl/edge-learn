import logging

import torch
import json
import copy
from torch.nn import DataParallel
from apex import amp

from time import perf_counter

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
        time_step_start = perf_counter()
        self.model.train()
        self.optimizer.zero_grad()
        time_forward_start = perf_counter()
        output = self.model(data)  # data can be on CPU
        time_forward_end = perf_counter()
        time_loss_start = perf_counter()
        loss_val = self.loss(
            output, target.to(output.device)
        )  # Move target to the same device as output
        time_loss_end = perf_counter()
        time_backward_start = perf_counter()
        with amp.scale_loss(loss_val, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        time_backward_end = perf_counter()
        time_optimizer_start = perf_counter()
        self.optimizer.step()
        time_optimizer_end = perf_counter()
        time_step_end = perf_counter()

        logging.info(
            f"TRAINING BREAKDOWN\nForward pass {time_forward_end - time_forward_start}\nLoss calc {time_loss_end - time_loss_start}\nBackward pass {time_backward_end - time_backward_start}\nOptimizer step {time_optimizer_end - time_optimizer_start}\nEntire Training Step {time_step_end - time_step_start}"
        )
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

    def test(self, loss, testloader):
        self.model.eval()

        logging.debug("Test Loader instantiated.")

        correct_pred = {}
        total_pred = {}

        total_correct = 0
        total_predicted = 0

        # Get the device of the model
        device = next(self.model.parameters()).device
        logging.debug("Testing on: {}".format(device))
        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                # Move the data to the same device as the model
                elems = elems.to(device)
                labels = labels.to(device)

                outputs = self.model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    label = label.item()

                    if label not in correct_pred:
                        correct_pred[label] = 0
                        total_pred[label] = 0

                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

            logging.debug("Predicted on the test set")

            class_accuracies = {}
            for key, value in correct_pred.items():
                accuracy = (
                    100.0
                    if total_pred[key] == 0
                    else 100 * float(value) / total_pred[key]
                )
                class_accuracies[key] = accuracy
                logging.debug(
                    "Accuracy for class {} is: {:.1f} %".format(key, accuracy)
                )

            overall_accuracy = 100 * float(total_correct) / total_predicted
            loss_val = loss_val / count
            logging.info("Overall accuracy is: {:.1f} %".format(overall_accuracy))
            return overall_accuracy, loss_val
