import importlib
import logging

import torch


class Sharing:
    """
    API defining who to share with and what, and what to do on receiving

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.graph = graph
        self.model = model
        self.dataset = dataset
        self.communication_round = 0
        self.log_dir = log_dir

        self.shapes = []
        self.lens = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():
                self.shapes.append(v.shape)
                t = v.flatten().cpu().numpy()
                self.lens.append(t.shape[0])

        self.compress = compress

        if compression_package and compression_class:
            compressor_module = importlib.import_module(compression_package)
            compressor_class = getattr(compressor_module, compression_class)
            self.compressor = compressor_class(float_precision=float_precision)
            logging.debug(f"Using the {compressor_class} to compress the data")
        else:
            assert not self.compress

    def compress_data(self, data):
        result = dict(data)
        if self.compress:
            if "params" in result:
                result["params"] = self.compressor.compress_float(result["params"])
        return result

    def decompress_data(self, data):
        if self.compress:
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        return data

    def serialized_model(self):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        dict
            Model converted to dict

        """
        to_cat = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                to_cat.append(t)
        flat = torch.cat(to_cat)
        data = dict()
        data["params"] = flat.numpy()
        logging.info("Model sending this round: {}".format(data["params"]))
        return self.compress_data(data)

    def deserialized_model(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        state_dict = dict()
        m = self.decompress_data(m)
        T = m["params"]
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            state_dict[key] = torch.from_numpy(
                T[start_index:end_index].reshape(self.shapes[i])
            )
            start_index = end_index

        return state_dict

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        pass

    def _post_step(self):
        """
        Called at the end of step.

        """
        pass

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            total = dict()
            weight_total = 0
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data)
                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for key, value in data.items():
                    if key in total:
                        total[key] += value * weight
                    else:
                        total[key] = value * weight

            for key, value in self.model.state_dict().items():
                total[key] += (1 - weight_total) * value  # Metro-Hastings

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, degree=None):
        self._pre_step()
        data = self.serialized_model()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["degree"] = degree if degree != None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data

    def _averaging_server(self, peer_deques):
        """
        Averages the received models of all working nodes

        """
        with torch.no_grad():
            total = dict()
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data)
                weight = 1 / len(peer_deques)
                for key, value in data.items():
                    if key in total:
                        total[key] += weight * value
                    else:
                        total[key] = weight * value

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1
        return total
