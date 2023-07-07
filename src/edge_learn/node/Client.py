import torch
import importlib
import logging
import json
import os
from time import perf_counter

from edge_learn.mappings.EdgeMapping import EdgeMapping
from decentralizepy.node.Node import Node
from decentralizepy import utils

class Client(Node):
    """
    Defines the client node (representing a consumer device)
    Responsible for distributing data collected to its edge server and receiving the model from the edge server
    """

    class DisconnectedException(Exception):
        pass

    def run(self):
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
    
    def initialize_run(self):
        self.trainset = self.dataset.get_trainset(self.batch_size_to_send, True)
        self.dataiter = iter(self.trainset)
        self.start_time = perf_counter()
    
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
        to_send = dict()
        to_send["CHANNEL"] = "DATA"
        to_send["iteration"] = self.iteration
        to_send["params"] = next(self.dataiter) # TODO ensure this is proper way to get next batch
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
                "total_elapsed_time": {},
            }

        
        cur_time = perf_counter()
        results_dict["total_bytes"][self.iteration + 1] = self.communication.total_bytes
        results_dict["total_elapsed_time"][self.iteration + 1] = cur_time - self.start_time
        self.start_time = cur_time

        # Removed Evaluation TODO from here, add it to the edge server because all clients have the same model.
        # Will have to use the timestamp of the client but accuracy of the edge server.
        # Write model with a unique name using the round number and client uid to NFS
        # Evaluate on another machine
        
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

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str=".",
        log_level: logging=logging.INFO,
        batch_size_to_send: int= 64,
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        torch.set_num_threads(1) # No learning, so 1 thread is enough
        torch.set_num_interop_threads(1)
        
        self.instantiate(
            rank,
            machine_id,
            mapping,
            config,
            log_dir,
            log_level,
            batch_size_to_send,
            *args
        )

        self.run()

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        config: dict,
        log_dir: str,
        log_level: logging,
        batch_size_to_send: int,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        logging.info("Started process")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            log_dir,
            batch_size_to_send,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()
        self.barrier = set()
        self.my_neighbors = set()
        self.my_neighbors.update(self.parents)
        self.my_neighbors.update(self.children)
        self.connect_neighbors()
        self.connected_to_edge_server = True

        self.init_sharing(config["SHARING"])

        print("Initialized client: ", self.uid)

    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        log_dir: str,
        batch_size_to_send: int,
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

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, 1, **comm_params
        )