from collections import deque
from decentralizepy.node.Node import Node
from decentralizepy.communication.TCP import TCP
from EdgeMapping import EdgeMapping
from decentralizepy.graphs.Graph import Graph
import logging

import torch
import os
import json

class PrimaryCloud(Node):
    """
    Defines the primary cloud node
    Responsible for aggregating the models from the edge servers and redistributing them
    Defaulted to have a UID = -1
    """

    def run(self):
        for iteration in range(self.iterations):
            self.initialize_iteration(iteration)
            self.send_model_to_edge_servers()
            self.get_model_from_edge_servers_and_store_in_peer_deque()
            self.average_model_from_peer_deques()
            self.collect_stats()
        self.finalize_run()
                    
    
    def initialize_iteration(self, iteration: int):
        self.iteration = iteration
        self.peer_deques = dict()
    
    def send_model_to_edge_servers(self):
        to_send = dict()
        to_send["params"] = self.model.state_dict()
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        for edge in range(self.num_edge_servers):
            self.communication.send(edge, to_send)
    
    def get_model_from_edge_servers_and_store_in_peer_deque(self):
        while not self.receive_from_all():
            sender, data = self.receive_channel("MODEL")

            if sender not in self.peer_deques:
                self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendLeft(data)
                else:
                    self.peer_deques[sender].append(data)
        logging.debug("Received model from each edge server")
    
    def receive_from_all(self):
        for k in range(self.num_edge_servers):
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True
    
    def average_model_from_peer_deques(self):
        averaging_deque = deque()
        for edge in range(self.num_edge_servers):
            averaging_deque[edge] = self.peer_deques[edge]
        self.sharing._pre_step()
        self.sharing._averaging_server(averaging_deque)
     
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

            for edge in range(self.num_edge_servers):
                self.communication.send(
                    edge, {"STATUS": "BYE", "CHANNEL": "MODEL"}
                )
            self.sent_disconnections = True

    def __init__(
        self, 
        rank: int,
        machine_id: int,
        mapping: EdgeMapping, 
        num_edge_servers: int,
        config, 
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        *args
    ):
        super.__init__(
            rank, 
            machine_id, 
            mapping, 
            None, 
            config, 
            iterations, 
            log_dir, 
            log_level, 
            *args
        )

        self.instantiate(
            rank, 
            machine_id, 
            mapping, 
            num_edge_servers,
            config, 
            iterations, 
            log_dir, 
            weights_store_dir, 
            log_level, 
            *args
        )

        self.run()
        logging.info("Primary cloud finished running")
    
    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        num_edge_servers: int,
        config,
        iterations: int,
        log_dir:str,
        weights_store_dir: str,
        log_level: int,
        *args
    ):
        logging.info("Started process.")

        self.init_log(log_dir, log_level)
        self.cache_fields(
            rank,
            machine_id,
            mapping,
            num_edge_servers,
            iterations,
            log_dir,
            weights_store_dir,
        )

        self.message_queue = dict()
        self.barrier = set()
        self.peer_deques = dict()

        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])

        self.my_neighbors = set(range(self.num_edge_servers))
        self.connect_neighbors()

        self.init_sharing(config["SHARING"])

    def init_log(self, log_dir: str, log_level: int, force: bool=True):
        log_file = os.path.join(log_dir, "PrimaryCloud.log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
            level=log_level,
            force=force,
        )
    
    def cache_fields(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        num_edge_servers: int,
        iterations: int,
        log_dir: str,
        weights_store_dir: str,
    ):
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.num_edge_servers = num_edge_servers
        self.uid = -1
        self.iterations = iterations
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir