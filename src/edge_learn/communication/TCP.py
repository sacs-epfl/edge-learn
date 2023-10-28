import json
import logging
import pickle
from collections import deque
from time import sleep

import socket

import zmq

from decentralizepy.communication.Communication import Communication

HELLO = b"HELLO"
BYE = b"BYE"


class TCP(Communication):
    """
    TCP Communication API

    """

    def addr(self, rank, machine_id):
        """
        Returns TCP address of the process.

        Parameters
        ----------
        rank : int
            Local rank of the process
        machine_id : int
            Machine id of the process

        Returns
        -------
        str
            Full address of the process using TCP

        """
        machine_config = self.ip_config[str(machine_id)]
        machine_addr = socket.gethostbyname(machine_config["host"])

        # Check if it's a cloud, edge, or client node and fetch the appropriate port
        if rank == -1:  # cloud
            port = machine_config["cloud_port"]
        elif rank == 0:  # edge
            port = machine_config["edge_port"]
        else:  # client
            # Note: you may need to handle cases where rank exceeds the number of client ports available.
            port = machine_config["client_ports"][rank - 1]

        assert port > 0
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        total_procs,
        ip_config_filepath,
        offset=9000,
        recv_timeout=50,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank of the process
        machine_id : int
            Machine id of the process
        mapping : decentralizepy.mappings.Mapping
            uid, rank, machine_id invertible mapping
        total_procs : int
            Total number of processes
        addresses_filepath : str
            JSON file with machine_id -> ip mapping
        compression_package : str
            Import path of a module that implements the compression.Compression.Compression class
        compression_class : str
            Name of the compression class inside the compression package

        """
        super().__init__(rank, machine_id, mapping, total_procs)

        with open(ip_config_filepath) as ip_config:
            self.ip_config = json.load(ip_config)

        self.total_procs = total_procs
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.offset = offset
        self.recv_timeout = recv_timeout
        self.uid = mapping.get_uid(rank, machine_id)
        self.identity = str(self.uid).encode()
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.IDENTITY, self.identity)
        self.router.setsockopt(zmq.RCVTIMEO, self.recv_timeout)
        self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.router.bind("tcp://0.0.0.0:1000")

        self.total_data = 0
        self.total_meta = 0

        self.peer_deque = deque()
        self.peer_sockets = dict()

        # sleep(2) # Sleep for socket creation everywhere

    def __del__(self):
        """
        Destroys zmq context

        """
        self.context.destroy(linger=0)

    def encrypt(self, data):
        """
        Encode data as python pickle.

        Parameters
        ----------
        data : dict
            Data dict to send

        Returns
        -------
        byte
            Encoded data

        """
        data_len = 0
        if "params" in data:
            data_len = len(pickle.dumps(data["params"]))
        output = pickle.dumps(data)
        self.total_meta += len(output) - data_len
        self.total_data += data_len
        return output

    def decrypt(self, sender, data):
        """
        Decode received pickle data.

        Parameters
        ----------
        sender : byte
            sender of the data
        data : byte
            Data received

        Returns
        -------
        tuple
            (sender: int, data: dict)

        """
        sender = int(sender.decode())
        data = pickle.loads(data)
        return sender, data

    def init_connection(self, neighbor):
        """
        Initiates a socket to a given node.

        Parameters
        ----------
        neighbor : int
            neighbor to connect to

        """
        logging.debug("Connecting to my neighbour: {}".format(neighbor))
        id = str(neighbor).encode()
        req = self.context.socket(zmq.DEALER)
        req.setsockopt(zmq.IDENTITY, self.identity)
        req.connect(self.addr(*self.mapping.get_machine_and_rank(neighbor)))
        self.peer_sockets[id] = req

    def destroy_connection(self, neighbor, linger=None):
        id = str(neighbor).encode()
        if self.already_connected(neighbor):
            self.peer_sockets[id].close(linger=linger)
            del self.peer_sockets[id]

    def already_connected(self, neighbor):
        id = str(neighbor).encode()
        return id in self.peer_sockets

    def receive(self, block=True):
        """
        Returns ONE message received.

        Returns
        ----------
        dict
            Received and decrypted data

        Raises
        ------
        RuntimeError
            If received HELLO

        """
        while True:
            try:
                sender, recv = self.router.recv_multipart()
                s, r = self.decrypt(sender, recv)
                return s, r
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    if not block:
                        return None
                    else:
                        continue
                else:
                    raise

    def send(self, uid, data, encrypt=True):
        """
        Send a message to a process.

        Parameters
        ----------
        uid : int
            Neighbor's unique ID
        data : dict
            Message as a Python dictionary

        """

        if encrypt:
            to_send = self.encrypt(data)
        else:
            to_send = data
        data_size = len(to_send)
        self.total_bytes += data_size
        id = str(uid).encode()
        self.peer_sockets[id].send(to_send)
        logging.debug("{} sent the message to {}.".format(self.uid, uid))
        logging.debug("Sent message size: {}".format(data_size))
