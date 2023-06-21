from decentralizepy.node.Node import Node
from decentralizepy.communication.TCP import TCP

class PrimaryCloud(Node):
    """
    Defines the primary cloud node, Would be the same as FederatedParameterServer, with compatibility modifications.

    Connect only to edge servers.

    __init__() will be called when the process is spawned

    UID = -1
    """