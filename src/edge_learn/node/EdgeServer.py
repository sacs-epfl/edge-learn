from collections import deque
import logging
import torch
import os
import json
import importlib
from ast import literal_eval

from decentralizepy.node.Node import Node
from decentralizepy import utils

from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.datasets.FlexDataset import FlexDataset
from edge_learn.enums.LearningMode import LearningMode
from edge_learn.utils.Graphing import create_and_save_plot


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
                self.create_mini_batches()
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

        before = self.communication.total_bytes
        for client in self.children:
            self.communication.send(client, to_send)
        self.amt_bytes_sent_to_client = (
            self.communication.total_bytes - before
        ) // len(self.children)

    def send_data_to_primary_cloud(self):
        to_send = dict()
        to_send["params"] = (self.received_batch["data"], self.received_batch["target"])
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "DATA"
        to_send["STATUS"] = "OK"

        before = self.communication.total_bytes
        self.communication.send(self.parents[0], to_send)
        self.amt_bytes_sent_to_cloud = self.communication.total_bytes - before

    def get_data_from_clients(self):
        self.batches_received = dict()
        while not self.receive_from_all_clients(self.batches_received):
            sender, data = self.receive_channel("DATA")

            if sender not in self.batches_received:
                self.batches_received[sender] = deque()

            if data["iteration"] == self.iteration:
                self.batches_received[sender].appendleft(data)
            else:
                self.batches_received[sender].append(data)
        logging.info("Received data from all clients")

    def get_model_from_clients_and_store_in_peer_deque(self):
        while not self.receive_from_all_clients(self.peer_deques):
            sender, data = self.receive_channel("MODEL")

            if sender not in self.peer_deques:
                self.peer_deques[sender] = deque()

            if data["iteration"] == self.iteration:
                self.peer_deques[sender].appendleft(data)
            else:
                self.peer_deques[sender].append(data)
        logging.debug("Received model from each client")

    def receive_from_all_clients(self, dict):
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
        for client in self.children:
            averaging_deque[client] = self.peer_deques[client]
        self.sharing._pre_step()
        self.sharing._averaging_server(averaging_deque)
        logging.info("Averaged model from each edge server")

    def create_batch_and_cache(self):
        self.create_batch()
        # if len(self.received_batch["data"]) != 0:
        #     self.collected_dataset.add_batch(self.received_batch)

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

    def create_mini_batches(self):
        self.mini_batches = []
        batch_data = self.received_batch["data"]
        batch_target = self.received_batch["target"]
        total_size = batch_data.size(0)

        for i in range(0, total_size, self.train_batch_size):
            mini_batch_data = batch_data[i : i + self.train_batch_size]
            mini_batch_target = batch_target[i : i + self.train_batch_size]

            # if len(mini_batch_data) < self.train_batch_size:
            #     amount_records_needed = self.train_batch_size - len(mini_batch_data)
            #     data_loader = self.collected_dataset.get_trainset(amount_records_needed)
            #     if data_loader is not None:
            #         iter_data_loader = iter(data_loader)
            #         extra_records = next(iter_data_loader)
            #         mini_batch_data = torch.cat(
            #             (mini_batch_data, extra_records[0]), dim=0
            #         )
            #         mini_batch_target = torch.cat(
            #             (mini_batch_target, extra_records[1]), dim=0
            #         )
            self.mini_batches.append(
                {"data": mini_batch_data, "target": mini_batch_target}
            )

        logging.info("Created mini-batches from received batch data")

        # mini_batch = []
        # while mini_batch.__len__() != 0:
        #     mini_batch.append(batch.remove(0))
        #     if mini_batch.__len__() == self.train_batch_size:
        #         self.mini_batches.append(mini_batch)
        #         mini_batch = []
        #     if mini_batch.__len__() != 0:
        #         amount_records_needed = self.train_batch_size - mini_batch.__len__()
        #         data_loader = self.collected_dataset.get_trainset(amount_records_needed)
        #         if data_loader is not None:
        #             iter_data_loader = iter(data_loader)
        #             extra_records = next(iter_data_loader)
        #             logging.info(
        #                 "Amount of extra records taken: {}".format(extra_records[0].shape)
        #             )
        #             mini_batch["data"] = torch.cat(
        #                 mini_batch["data"], extra_records["data"]
        #             )
        #             mini_batch["target"] = torch.cat(
        #                 mini_batch["target"], extra_records["target"]
        #             )

    # def fill_batch_till_target(self):
    #     self.batch = dict()
    #     current_batch_size = self.received_batch["data"].shape[0]
    #     amountRecordsNeeded = self.train_batch_size - current_batch_size
    #     if amountRecordsNeeded < 0:
    #         self.batch = self.received_batch
    #         return
    #     data_loader = self.collected_dataset.get_trainset(amountRecordsNeeded)
    #     if data_loader is not None:
    #         iter_data_loader = iter(data_loader)
    #         relooked_batch = next(iter_data_loader)
    #         logging.info("Type of data relooked: %s", relooked_batch[0].dtype)
    #         logging.debug("Relooked batch size: {}".format(relooked_batch[0].shape))
    #         self.batch["data"] = torch.cat(
    #             (self.received_batch["data"], relooked_batch[0])
    #         )
    #         self.batch["target"] = torch.cat(
    #             (self.received_batch["target"], relooked_batch[1])
    #         )
    #     else:
    #         self.batch = self.received_batch

    def train(self):
        logging.info("Starting training phase")
        if self.iteration % self.lr_scheduler_frequency == 0 and self.iteration != 0:
            self.lr_scheduler.step()

        losses = []
        for mini_batch in self.mini_batches:
            losses.append(
                self.trainer.trainstep(mini_batch["data"], mini_batch["target"].long())
            )

            logging.info(
                "Trained model for one mini-batch with a loss of {}".format(
                    losses[len(losses) - 1]
                )
            )
        self.loss_amt = sum(losses) / len(losses) if len(losses) != 0 else -1
        logging.info("Finished training phase")

    def send_model_to_primary_cloud(self):
        to_send = self.sharing.serialized_model()
        to_send["iteration"] = self.iteration
        to_send["degree"] = 1
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        before = self.communication.total_bytes
        self.communication.send(self.parents[0], to_send)
        self.amt_bytes_sent_to_cloud = self.communication.total_bytes - before

    def collect_stats(self):
        file_path = os.path.join(self.log_dir, "{}_results.json".format(self.rank))

        if os.path.exists(file_path):
            with open(
                file_path,
                "r",
            ) as inf:
                results_dict = json.load(inf)
        else:
            results_dict = {
                "train_loss": {},
                "bytes_sent_to_each_client": {},
                "bytes_sent_to_cloud": {},
                "total_meta": {},
                "total_data_per_n": {},
            }

        if LearningMode(self.learning_mode) == LearningMode.HYBRID:
            results_dict["train_loss"][self.iteration + 1] = self.loss_amt

        results_dict["bytes_sent_to_each_client"][
            self.iteration + 1
        ] = self.amt_bytes_sent_to_client
        results_dict["bytes_sent_to_cloud"][
            self.iteration + 1
        ] = self.amt_bytes_sent_to_cloud

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
        self.model.dump_weights(self.weights_store_dir, self.uid, self.iteration)
        logging.info("Stored final weights")

    def save_graphs(self):
        with open(
            os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
            "r",
        ) as inf:
            results_dict = json.load(inf)
            if LearningMode(self.learning_mode) == LearningMode.HYBRID:
                create_and_save_plot(
                    "Training Loss",
                    results_dict["train_loss"],
                    "Rounds",
                    "Training Loss",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

    @classmethod
    def create(
        cls,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        train_batch_size=32,
        learning_mode="H",
        num_threads=1,
    ):
        if LearningMode(learning_mode) == LearningMode.BASELINE:
            return None
        return cls(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            weights_store_dir,
            log_level,
            train_batch_size,
            learning_mode,
            num_threads,
        )

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        train_batch_size=32,
        learning_mode="H",
        num_threads=1,
    ):
        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            weights_store_dir,
            log_level,
            train_batch_size,
            learning_mode,
            num_threads,
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
        train_batch_size: int,
        learning_mode: str,
        num_threads: int,
    ):
        logging.info("Started process")

        self.init_log(log_dir, rank, log_level)
        self.cache_fields(
            rank,
            machine_id,
            mapping,
            log_dir,
            weights_store_dir,
            train_batch_size,
            learning_mode,
            num_threads,
        )

        self.batches_received = dict()
        self.peer_deques = dict()

        self.collected_dataset = FlexDataset()
        self.init_model(config["DATASET"])
        self.dataset = None
        self.init_comm(config["COMMUNICATION"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_lr_scheduler(config["LR_SCHEDULER"])
        self.init_trainer(config["TRAIN_PARAMS"])

        self.message_queue = dict()
        self.barrier = set()
        self.my_neighbors = set()
        self.my_neighbors.update(self.parents)
        self.my_neighbors.update(self.children)
        self.connect_neighbors()
        self.connected_to_primary_cloud = True

        self.init_sharing(config["SHARING"])

        self.set_threads()

        print("Initialized Edge Server")

    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        log_dir: str,
        weights_store_dir: str,
        train_batch_size: int,
        learning_mode: str,
        num_threads: int,
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
        self.sent_disconnections = False
        self.train_batch_size = train_batch_size
        self.learning_mode = learning_mode
        self.num_threads = num_threads

    def init_model(self, dataset_configs):
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = self.model_class()

    def init_comm(self, comm_configs):
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, 1, **comm_params
        )

    def init_optimizer(self, optimizer_configs):
        optimizer_module = importlib.import_module(
            optimizer_configs["optimizer_package"]
        )

        self.optimizer_class = getattr(
            optimizer_module, optimizer_configs["optimizer_class"]
        )

        # Prepare optimizer parameters, converting strings to tuples if necessary
        self.optimizer_params = {}
        for key, value in optimizer_configs.items():
            if key not in ["optimizer_package", "optimizer_class"]:
                # Here we handle the conversion of the 'betas' string to a tuple
                if key == "betas":
                    value = literal_eval(value)
                self.optimizer_params[key] = value

        # Instantiate the optimizer with the prepared parameters
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_params
        )

    def init_lr_scheduler(self, scheduler_configs):
        scheduler_module = importlib.import_module(
            scheduler_configs["scheduler_package"]
        )
        scheduler_class = getattr(
            scheduler_module, scheduler_configs["scheduler_class"]
        )
        self.lr_scheduler_frequency = scheduler_configs["frequency"]
        scheduler_params = utils.remove_keys(
            scheduler_configs, ["scheduler_package", "scheduler_class", "frequency"]
        )
        self.lr_scheduler = scheduler_class(self.optimizer, **scheduler_params)

    def set_threads(self):
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(1)
