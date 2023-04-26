from decentralizepy.node.Node import Node
from decentralizepy.node.FederatedParameterServer import FederatedParameterServer
from decentralizepy.communication.TCP import TCP

class EdgeServer(Node):
    """
    This should ideally be similar to FederatedParameterServer

    For starters the graph here should represent a tree (or any other heirarchy).
    UID 1->2^D

    Heads up: The Graph is used by Communication module to instantiate socket connections.
    There is also a synchronization barrier.

    Define simple API for now

    Pointers: Can exchange data: up or down. Can also exchange models: up or down.

    Can also act as aggregator for the models.
    """