import torch
import os
import json
from collections import deque
import logging
import importlib

from decentralizepy.node.Node import Node
from decentralizepy import utils

from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.datasets.FlexDataset import FlexDataset
from edge_learn.enums.LearningMode import LearningMode


class PrimaryCloud(Node):
    """
    Defines the primary cloud node
    Responsible for aggregating the models from the edge servers and redistributing them
    Defaulted to have a UID = -1
    """

    def runHybrid(self):
        for iteration in range(self.iterations):
            self.initialize_iteration(iteration)
            self.send_model_to_edge_servers()
            self.get_model_from_edge_servers_and_store_in_peer_deque()
            self.average_model_from_peer_deques()
            self.collect_stats()
        self.finalize_run()

    def runOnlyData(self):
        for iteration in range(self.iterations):
            self.initialize_iteration(iteration)
            self.send_model_to_edge_servers()
            self.get_data_from_edge_servers()
            self.create_batch_and_cache()
            self.fill_batch_till_target()
            self.train()
            self.collect_stats()
        self.finalize_run()

    def runOnlyWeights(self):
        self.runHybrid()

    def initialize_iteration(self, iteration: int):
        self.iteration = iteration
        self.peer_deques = dict()

    def send_model_to_edge_servers(self):
        to_send = dict()
        to_send["params"] = self.model.state_dict()
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        for edge in self.children:
            self.communication.send(edge, to_send)

    def get_model_from_edge_servers_and_store_in_peer_deque(self):
        while not self.receive_from_all(self.peer_deques):
            sender, data = self.receive_channel("MODEL")

            if sender not in self.peer_deques:
                self.peer_deques[sender] = deque()

            if data["iteration"] == self.iteration:
                self.peer_deques[sender].appendleft(data)
            else:
                self.peer_deques[sender].append(data)
        logging.debug("Received model from each edge server")

    def get_data_from_edge_servers(self):
        self.batches_received = dict()
        while not self.receive_from_all(self.batches_received):
            sender, data = self.receive_channel("DATA")

            if sender not in self.batches_received:
                self.batches_received[sender] = deque()

            if data["iteration"] == self.iteration:
                self.batches_received[sender].appendleft(data)
            else:
                self.batches_received[sender].append(data)
        logging.info("Received data from all clients")

    def receive_from_all(self, dict):
        for k in self.children:
            if (
                (k not in dict)
                or len(dict[k]) == 0
                or dict[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def average_model_from_peer_deques(self):
        averaging_deque = dict()
        for edge in self.children:
            averaging_deque[edge] = self.peer_deques[edge]
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
        current_batch_size = self.received_batch["data"].shape[0]
        amountRecordsNeeded = self.train_batch_size - current_batch_size
        if amountRecordsNeeded < 0:
            self.batch = self.received_batch
            return
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
                "total_bytes": {},
                "total_meta": {},
                "total_data_per_n": {},
            }

        if LearningMode(self.learning_mode) == LearningMode.ONLY_DATA:
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

        with open(
            os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
        ) as of:
            json.dump(results_dict, of)

    def finalize_run(self):
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, self.iteration)
        logging.info("All neighbors disconnected. Process complete!")

    def disconnect_neighbors(self):
        if not self.sent_disconnections:
            logging.info("Disconnecting neighbors")

            for edge in self.children:
                self.communication.send(edge, {"STATUS": "BYE", "CHANNEL": "MODEL"})
            self.sent_disconnections = True

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        train_batch_size=32,
        learning_mode="H",
        *args
    ):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
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
            raise ValueError("Learning mode must be one of H, OD, OW")
        logging.info("Primary cloud finished running")

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        iterations: int,
        log_dir: str,
        weights_store_dir: str,
        log_level: int,
        train_batch_size: int,
        learning_mode: str,
        *args
    ):
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)
        self.cache_fields(
            rank,
            machine_id,
            mapping,
            iterations,
            log_dir,
            weights_store_dir,
            train_batch_size,
            learning_mode,
        )

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

        self.init_sharing(config["SHARING"])

        print("Initialized primary cloud")

    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        iterations: int,
        log_dir: str,
        weights_store_dir: str,
        train_batch_size: int,
        learning_mode: str,
    ):
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.graph = None
        self.uid = -1
        self.parents = self.mapping.get_parents(self.uid)
        self.children = self.mapping.get_children(self.uid)
        self.iterations = iterations
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
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
