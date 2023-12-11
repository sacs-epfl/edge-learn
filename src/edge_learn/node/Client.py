import torch
import importlib
import logging
import json
import os
from ast import literal_eval

from edge_learn.mappings.EdgeMapping import EdgeMapping
from decentralizepy.node.Node import Node
from decentralizepy import utils
from edge_learn.enums.LearningMode import LearningMode


class Client(Node):
    """
    Defines the client node (representing a consumer device)
    Responsible for distributing data collected to its edge server and receiving the model from the edge server
    """

    class DisconnectedException(Exception):
        pass

    def runHybrid(self):
        try:
            self.initialize_run()
            while self.connected_to_edge_server:
                self.get_model_from_edge_server()
                self.send_batch_to_edge_server()
                self.collect_stats()
        except self.DisconnectedException:
            pass
        finally:
            self.finalize_run()

    def runOnlyWeights(self):
        try:
            self.initialize_run()
            while self.connected_to_edge_server:
                self.get_model_from_edge_server()
                self.train()
                self.send_model_to_edge_server()
                self.collect_stats()
        except self.DisconnectedException:
            pass
        finally:
            self.finalize_run()

    def runOnlyData(self):
        self.runHybrid()

    def initialize_run(self):
        self.trainset = self.dataset.get_trainset(self.batch_size_to_send, True)
        self.dataiter = iter(self.trainset)

    def get_model_from_edge_server(self):
        sender, data = self.receive_channel("MODEL")

        if data["STATUS"] == "BYE":
            logging.debug("Received {} from {}".format("BYE", sender))
            self.connected_to_edge_server = False
            raise self.DisconnectedException()

        self.iteration = data["iteration"]
        del data["iteration"]
        del data["CHANNEL"]

        self.model.load_state_dict(data["params"])
        self.sharing.communication_round += 1

        logging.info("Received model from edge")

    def send_batch_to_edge_server(self):
        try:
            to_send = dict()
            to_send["CHANNEL"] = "DATA"
            to_send["iteration"] = self.iteration

            data, target = next(self.dataiter)
            if data.nelement() != 0:
                self.last_dtype_data = data.dtype
                logging.info("Last dtype data: %s", self.last_dtype_data)
            if target.nelement() != 0:
                self.last_dtype_target = target.dtype
                logging.info("Last dtype target: %s", self.last_dtype_target)
            to_send["params"] = (data, target)
            to_send["STATUS"] = "OK"
            if self.epoch_confirmation:
                to_send["epoch"] = True
                self.epoch_confirmation = False

            before = self.communication.total_bytes
            self.communication.send(self.parents[0], to_send)
            self.amt_bytes_sent_to_edge = self.communication.total_bytes - before

        except StopIteration:
            logging.debug("Ran out of data, recreating iterable")
            self.epoch_confirmation = True
            self.dataiter = iter(self.trainset)
            self.send_batch_to_edge_server()

    def send_model_to_edge_server(self):
        to_send = self.sharing.serialized_model()
        to_send["CHANNEL"] = "MODEL"
        to_send["iteration"] = self.iteration
        to_send["STATUS"] = "OK"
        to_send["degree"] = 1

        before = self.communication.total_bytes
        self.communication.send(self.parents[0], to_send)
        self.amt_bytes_sent_to_edge = self.communication.total_bytes - before

    def train(self):
        if self.epoch_confirmation:
            self.lr_scheduler.step()
            self.epoch_confirmation = False
        data, target = None, None
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.epoch_confirmation = True
            self.dataiter = iter(self.trainset)
            data, target = next(self.dataiter)
        self.loss_amt = self.trainer.trainstep(data, target.long())

        logging.info(
            "Trained model for one step with a loss of {}".format(self.loss_amt)
        )

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
                "bytes_sent_to_edge": {},
                "total_meta": {},
                "total_data_per_n": {},
            }

        results_dict["bytes_sent_to_edge"][
            self.iteration + 1
        ] = self.amt_bytes_sent_to_edge

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
        logging.info("Server disconnected. Process complete!")

    @classmethod
    def create(
        cls,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str = ".",
        log_level: logging = logging.INFO,
        batch_size_to_send: int = 64,
        learning_mode: str = "H",
        num_threads: int = 1,
    ):
        if LearningMode(learning_mode) == LearningMode.BASELINE:
            return None
        return cls(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            log_level,
            batch_size_to_send,
            learning_mode,
            num_threads,
        )

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str = ".",
        log_level: logging = logging.INFO,
        batch_size_to_send: int = 64,
        learning_mode: str = "H",
        num_threads: int = 1,
    ):
        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            log_level,
            batch_size_to_send,
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

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str,
        log_level: logging,
        batch_size_to_send: int,
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
            batch_size_to_send,
            learning_mode,
            num_threads,
        )
        self.last_dtype_data = torch.int64
        self.last_dtype_target = torch.int64
        self.init_dataset_model(config["DATASET"])
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
        self.connected_to_edge_server = True

        self.init_sharing(config["SHARING"])

        self.set_threads()

        print("Initialized client: ", self.uid)

    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        log_dir: str,
        batch_size_to_send: int,
        learning_mode: str,
        num_threads: int,
    ):
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.graph = None
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.parents = self.mapping.get_parents(self.uid)
        self.children = self.mapping.get_children(self.uid)
        self.log_dir = log_dir
        self.batch_size_to_send = batch_size_to_send
        self.learning_mode = learning_mode
        self.num_threads = num_threads

    def init_dataset_model(self, dataset_configs):
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        self.dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        random_seed = (
            dataset_configs["random_seed"] if "random_seed" in dataset_configs else 97
        )
        torch.manual_seed(random_seed)
        self.dataset_params = utils.remove_keys(
            dataset_configs,
            ["dataset_package", "dataset_class", "model_class", "random_seed"],
        )
        self.dataset = self.dataset_class(
            self.rank,
            self.machine_id,
            self.mapping,
            LearningMode(self.learning_mode),
            train=True,
            test=False,
            **self.dataset_params
        )
        logging.info("Dataset instantiation complete.")
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
        if self.learning_mode == LearningMode.ONLY_WEIGHTS:
            total_threads = os.cpu_count()
            max_threads = max(
                (total_threads - 2) // self.mapping.get_procs_per_machine(), 1
            )
            torch.set_num_threads(max_threads)
        else:
            torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
