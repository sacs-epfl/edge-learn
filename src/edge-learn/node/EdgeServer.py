from decentralizepy.node.Node import Node
from decentralizepy.node.FederatedParameterServer import FederatedParameterServer
from decentralizepy.communication.TCP import TCP
from decentralizepy.graphs.Graph import Graph

import EdgeMapping

import logging
import torch
import os
import json

class EdgeServer(Node):
    """
    Defines the edge server node
    Responsible for training model based on data from clients and sending the model to the 
        primary cloud and returning the model to the clients
    """

    def run(self):
        self.initialize_run()
        while self.connected_to_primary_cloud:
            self.get_model_from_primary_cloud()
            self.get_data_from_clients()
            self.train()
            self.send_model_to_clients_and_primary_cloud()
            self.collect_stats()
        self.finalize_run()
    
    def initialize_run(self):
        self.rounds_to_train_evaluate = self.train_evaluate_after
    
    def get_model_from_primary_cloud(self):
        sender, data = self.receive_channel("MODEL")

        if data["STATUS"] == "BYE":
            logging.debug("Received {} from {}".format("BYE", sender))
            self.connected_to_primary_cloud = False
            return

        self.model = data["params"]
        self.iteration = data["iteration"]
    
    def get_data_from_clients(self):
        for client in self.clients:
            sender, data = self.receive_channel("DATA")
            batch = data["batch"]
            # Somehow add data to the dataset and data to be trained on

    def train(self):
        pass

    def send_model_to_clients_and_primary_cloud(self):
        to_send = dict()
        to_send["params"] = self.model.state_dict()
        to_send["iteration"] = self.iteration
        to_send["CHANNEL"] = "MODEL"
        to_send["STATUS"] = "OK"

        for client in self.clients:
            self.communication.send(client, to_send)
        
        self.communication.send(-1, to_send)
    
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

        self.rounds_to_train_evaluate -= 1

        if self.rounds_to_train_evaluate == 0:
            logging.info("Evaluating on train set")
            self.rounds_to_train_evaluate = self.train_evaluate_after
            loss = self.trainer.eval_loss(self.dataset)
            results_dict["train_loss"][self.iteration + 1] = loss
            self.save_plot(
                results_dict["train_loss"],
                "train_loss",
                "Training Loss",
                "Communication Rounds",
                os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
            )
        
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
            logging.info('Sending disconnection messages to all clients')
            for client in self.clients:
                logging.info("Disconnecting from {}".format(client))
                self.communication.send(client, {"STATUS": "BYE", "CHANNEL": "MODEL"})
            self.sent_disconnections = True

    def __init__(
        self, 
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,  
        config, 
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        train_evaluate_after=5,
        *args
    ):
        super.__init__(
            rank, 
            machine_id, 
            mapping, 
            None, 
            config, 
            1, 
            log_dir, 
            log_level, 
            *args
        )

        self.instantiate(
            rank, 
            machine_id, 
            mapping, 
            config, 
            log_dir, 
            weights_store_dir, 
            log_level, 
            train_evaluate_after,
            *args
        )

        self.run()
        logging.info("Edge Server finished running")
    
    def instantiate(
        self,
        rank: int, 
        machine_id: int, 
        mapping: EdgeMapping, 
        config, 
        log_dir: str, 
        weights_store_dir: str, 
        log_level: int, 
        train_evaluate_after: int,
        *args
    ):
        logging.info("Started process.")

        self.init_log(log_dir, log_level)
        self.cache_fields(
            rank,
            machine_id,
            mapping,
            log_dir,
            weights_store_dir,
            train_evaluate_after,
        )

        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])

        self.my_neighbors = set(map(lambda x: self.mapping.get_uid(x, self.machine_id), range(1, self.mapping.get_procs_per_machine() + 1)))
        self.my_neighbors.add(-1)
        self.connect_neighbors()
        self.connected_to_primary_cloud = True

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
        train_evaluate_after: int,
    ):
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.num_edge_servers = num_edge_servers
        self.uid = self.mapping.get_uid(self.rank, self.machine_id)
        self.iterations = iterations
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir     
        self.train_evaluate_after = train_evaluate_after   