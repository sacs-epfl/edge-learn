from decentralizepy.node.DPSGDNodeFederated import DPSGDNodeFederated
from decentralizepy.training.Training import Training
from decentralizepy.communication.TCP import TCP
import torch
import logging
import json
from EdgeMapping import EdgeMapping
from decentralizepy.graphs.Graph import Graph
from collections import deque
import os
from time import perf_counter

class Client(DPSGDNodeFederated):
    """
    Perform training and exchange data and models with other the (closest) edge servers.

    TCP conveniently measures the bytes exchanged.
    """


    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        batch_size_to_send = 64,
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
        graph : decentralizepy.graphs
            The object containing the global graph
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
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        args : optional
            Other arguments

        """
        torch.set_num_threads(1) # No learning, so 1 thread is enough
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            batch_size_to_send,
            *args
        )

        self.message_queue["PEERS"] = deque()

        self.edge_server_uid = machine_id
        self.connect_neighbor(self.edge_server_uid)
        self.wait_for_hello(self.edge_server_uid)

        self.run()

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        batch_size_to_send,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.sent_disconnections = False
        self.batch_size_to_send = batch_size_to_send

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: EdgeMapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        batch_size_to_send = 64,
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
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            batch_size_to_send,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()

        self.participated = 0

        self.init_sharing(config["SHARING"])


    def run(self):
            """
            Start the decentralized learning

            """
            trainset = self.dataset.get_trainset(self.batch_size, True)
            start_time = perf_counter()
            while len(self.barrier):
                sender, data = self.receive_channel("WORKER_REQUEST")

                if "BYE" in data:
                    logging.debug("Received {} from {}".format("BYE", sender))
                    self.barrier.remove(sender)
                    break

                iteration = data["iteration"]
                del data["iteration"]
                del data["CHANNEL"]

                self.model.load_state_dict(data["params"])
                self.sharing._post_step()
                self.sharing.communication_round += 1

                logging.debug(
                    "Received worker request at node {}, global iteration {}, local round {}".format(
                        self.uid, iteration, self.participated
                    )
                )

                # Send update to server
                to_send = dict()
                # TODO: Add the batch from trainset here
                to_send["CHANNEL"] = "DPSGD"
                to_send["iteration"] = iteration
                self.communication.send(self.edge_server_uid, to_send)

                if self.participated > 0:
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
                results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes
                results_dict["total_elapsed_time"][iteration + 1] = cur_time - start_time
                start_time = cur_time

                # Removed Evaluation TODO from here, add it to the edge server because all clients have the same model.
                # Will have to use the timestamp of the client but accuracy of the edge server.
                # Write model with a unique name using the round number and client uid to NFS
                # Evaluate on another machine
                

                if hasattr(self.communication, "total_meta"):
                    results_dict["total_meta"][
                        iteration + 1
                    ] = self.communication.total_meta
                if hasattr(self.communication, "total_data"):
                    results_dict["total_data_per_n"][
                        iteration + 1
                    ] = self.communication.total_data

                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
                ) as of:
                    json.dump(results_dict, of)

                self.participated += 1

            logging.info("Server disconnected. Process complete!")