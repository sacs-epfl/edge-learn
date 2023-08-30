from collections import deque
import logging
import torch
import os
import json
import importlib
import math
from matplotlib import pyplot as plt

from decentralizepy.node.Node import Node
from decentralizepy import utils

from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.datasets.FlexDataset import FlexDataset


class EdgeServer(Node):
    """
    Defines the edge server node
    Responsible for training model based on data from clients and sending the model to the
        primary cloud and returning the model to the clients
    """

    class DisconnectedException(Exception):
        pass

    def runHybrid(self):
        try:
            self.initialize_run()
            while self.connected_to_primary_cloud:
                self.get_model_from_primary_cloud()
                self.send_model_to_clients()
                self.get_data_from_clients()
                self.create_batch_and_cache()
                self.fill_batch_till_target()
                self.train()
                self.send_model_to_primary_cloud()
                self.collect_stats()
        except self.DisconnectedException:
            pass
        finally:
            self.finalize_run()

    def runOnlyWeights(self):
        try:
            self.initialize_run()
            while self.connected_to_primary_cloud:
                self.get_model_from_primary_cloud()
                self.send_model_to_clients()
                self.get_model_from_clients_and_store_in_peer_deque()
                self.average_model_from_peer_deques()
                self.send_model_to_primary_cloud()
                self.collect_stats()
        except self.DisconnectedException:
            pass
        finally:
            self.finalize_run()

    def runOnlyData(self):
        try:
            self.initialize_run()
            while self.connected_to_primary_cloud:
                self.get_model_from_primary_cloud()
                self.send_model_to_clients()
                self.get_data_from_clients()
                self.create_batch()
                self.send_data_to_primary_cloud()
                self.collect_stats()
        except self.DisconnectedException:
            pass
        finally:
            self.finalize_run()

    def initialize_run(self):
        self.rounds_to_test = self.test_after
        self.peer_deques = dict()

    def get_model_from_primary_cloud(self):
        sender, data = self.receive_channel("MODEL")

        if data["STATUS"] == "BYE":
            logging.debug("Received {} from {}".format("BYE", sender))
            self.connected_to_primary_cloud = False
            raise self.DisconnectedException()

        self.iteration = data["iteration"]
        self.model.load_state_dict(data["params"])
        self.sharing.communication_round += 1
        logging.debug("Received model")

    def send_model_to_clients(self):
        to_send = dict()
        to_send["params"] = self.model.state_dict()
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        for client in self.children:
            self.communication.send(client, to_send)

    def send_data_to_primary_cloud(self):
        to_send = dict()
        to_send["params"] = self.received_batch
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "DATA"
        to_send["STATUS"] = "OK"

    def get_data_from_clients(self):
        self.batches_received = dict()
        while not self.receive_from_all_clients():
            sender, data = self.receive_channel("DATA")

            if sender not in self.batches_received:
                self.batches_received[sender] = deque()

            if data["iteration"] == self.iteration:
                self.batches_received[sender].appendleft(data)
            else:
                self.batches_received[sender].append(data)
        logging.info("Received data from all clients")

    def receive_from_all_clients(self):
        for k in self.children:
            if (
                (k not in self.batches_received)
                or len(self.batches_received[k]) == 0
                or self.batches_received[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def get_model_from_clients_and_store_in_peer_deque(self):
        while not self.receive_from_all():
            sender, data = self.receive_channel("MODEL")

            if sender not in self.peer_deques:
                self.peer_deques[sender] = deque()

            if data["iteration"] == self.iteration:
                self.peer_deques[sender].appendleft(data)
            else:
                self.peer_deques[sender].append(data)
        logging.debug("Received model from each edge server")

    def receive_from_all(self):
        for k in self.children:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def average_model_from_peer_deques(self):
        averaging_deque = dict()
        for client in self.children:
            averaging_deque[client] = self.peer_deques[client]
        self.sharing._pre_step()
        self.sharing._averaging_server(averaging_deque)
        logging.info("Averaged model from each edge server")

    def create_batch_and_cache(self):
        self.create_batch()
        if len(self.received_batch["data"]) != 0:
            self.collected_dataset.add_batch(self.received_batch)

    def create_batch(self):
        batch_data = []
        batch_target = []

        for k in self.children:
            data, target = self.batches_received[k].popleft()["params"]
            batch_data.append(data)
            batch_target.append(target)

        self.received_batch = dict()
        self.received_batch["data"] = torch.cat(batch_data)
        self.received_batch["target"] = torch.cat(batch_target)

        logging.info("Created batch from data received from clients")
        logging.info("Type of data received: %s", self.received_batch["data"].dtype)

    def fill_batch_till_target(self):
        amountRecordsNeeded = (
            self.train_batch_size - self.received_batch["data"].shape[0]
        )
        data_loader = self.collected_dataset.get_trainset(amountRecordsNeeded)
        if data_loader is not None:
            iter_data_loader = iter(data_loader)
            relooked_batch = next(iter_data_loader)
            logging.info("Type of data relooked: %s", relooked_batch[0].dtype)
            logging.debug("Relooked batch size: {}".format(relooked_batch[0].shape))
            self.batch["data"] = torch.cat(
                (self.received_batch["data"], relooked_batch[0])
            )
            self.batch["target"] = torch.cat(
                (self.received_batch["target"], relooked_batch[1])
            )
        else:
            self.batch = self.received_batch

    def train(self):
        self.loss_amt = self.trainer.trainstep(
            self.batch["data"], self.batch["target"].long()
        )

        logging.info(
            "Trained model for one step with a loss of {}".format(self.loss_amt)
        )

    def send_model_to_primary_cloud(self):
        to_send = self.sharing.serialized_model()
        to_send["iteration"] = self.iteration
        to_send["degree"] = 1
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        self.communication.send(self.parents[0], to_send)

    def collect_stats(self):
        if self.iteration != 0:
            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                "r",
            ) as inf:
                results_dict = json.load(inf)
        else:
            results_dict = {
                "train_loss": {},
                "test_loss": {},
                "test_acc": {},
                "total_bytes": {},
                "total_meta": {},
                "total_data_per_n": {},
            }

        if self.loss_amt:
            results_dict["train_loss"][self.iteration + 1] = self.loss_amt

        results_dict["total_bytes"][self.iteration + 1] = self.communication.total_bytes

        if hasattr(self.communication, "total_meta"):
            results_dict["total_meta"][
                self.iteration + 1
            ] = self.communication.total_meta
        if hasattr(self.communication, "total_data"):
            results_dict["total_data_per_n"][
                self.iteration + 1
            ] = self.communication.total_data

        self.rounds_to_test -= 1

        if self.dataset.__testing__ and self.rounds_to_test == 0:
            self.rounds_to_test = self.test_after
            logging.info("Evaluating on test set.")
            ta, tl = self.dataset.test(self.model, self.loss)
            results_dict["test_acc"][self.iteration + 1] = ta
            results_dict["test_loss"][self.iteration + 1] = tl

        with open(
            os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
        ) as of:
            json.dump(results_dict, of)

    def finalize_run(self):
        self.disconnect_children()
        self.save_data()

    def disconnect_children(self):
        if not self.sent_disconnections:
            logging.info("Sending disconnection messages to all clients")
            for client in self.children:
                logging.info("Disconnecting from {}".format(client))
                self.communication.send(client, {"STATUS": "BYE", "CHANNEL": "MODEL"})
            self.sent_disconnections = True
        logging.info("All neighbors disconnected. Process complete!")

    def save_data(self):
        logging.info("Storing final weights and generating graphs")
        self.model.dump_weights(self.weights_store_dir, self.uid, self.iteration)
        with open(
            os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
            "r",
        ) as inf:
            results_dict = json.load(inf)
            self.save_graphs(results_dict)

    def save_graphs(self, results_dict):
        self.save_plot(
            results_dict["train_loss"],
            "train_loss",
            "Training Loss",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
        )
        self.save_plot(
            results_dict["test_acc"],
            "test_acc",
            "Test Accuracy",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_test_acc.png".format(self.rank)),
        )
        self.save_plot(
            results_dict["test_loss"],
            "test_loss",
            "Test Loss",
            "Communication Rounds",
            os.path.join(self.log_dir, "{}_test_loss.png".format(self.rank)),
        )

    def save_plot(self, l, label, title, xlabel, filename):
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_batch_size=32,
        learning_mode="H",
        *args
    ):
        total_threads = os.cpu_count()
        max_threads = max(total_threads - mapping.get_procs_per_machine() - 1, 1)
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(1)

        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_batch_size,
            learning_mode,
            *args
        )

        if self.learning_mode == "H":
            self.runHybrid()
        elif self.learning_mode == "OD":
            self.runOnlyData()
        elif self.learning_mode == "OW":
            self.runOnlyWeights()
        else:
            raise ValueError("Learning mode must be one of: H, OD, OW")
        logging.info("Edge Server finished running")

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str,
        weights_store_dir: str,
        log_level: int,
        test_after: int,
        train_batch_size: int,
        learning_mode: str,
        *args
    ):
        logging.info("Started process")

        self.init_log(log_dir, rank, log_level)
        self.cache_fields(
            rank,
            machine_id,
            mapping,
            log_dir,
            weights_store_dir,
            test_after,
            train_batch_size,
            learning_mode,
        )

        self.batches_received = dict()
        self.peer_deques = dict()

        self.collected_dataset = FlexDataset()
        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])

        self.message_queue = dict()
        self.barrier = set()
        self.my_neighbors = set()
        self.my_neighbors.update(self.parents)
        self.my_neighbors.update(self.children)
        self.connect_neighbors()
        self.connected_to_primary_cloud = True

        self.init_sharing(config["SHARING"])

        print("Initialized Edge Server")

    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        log_dir: str,
        weights_store_dir: str,
        test_after: int,
        train_batch_size: int,
        learning_mode: str,
    ):
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.graph = None
        self.uid = self.mapping.get_uid(self.rank, self.machine_id)
        self.parents = self.mapping.get_parents(self.uid)
        self.children = self.mapping.get_children(self.uid)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.test_after = test_after
        self.sent_disconnections = False
        self.train_batch_size = train_batch_size
        self.learning_mode = learning_mode

    def init_comm(self, comm_configs):
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, 1, **comm_params
        )
