from decentralizepy.node.Node import Node
from decentralizepy.node.FederatedParameterServer import FederatedParameterServer
from decentralizepy.communication.TCP import TCP

class EdgeServer(Node):
    """
    This should ideally be a mix of DPSGDNodeFederated FederatedParameterServer

    TODO: Connect to primary cloud server and clients. See EdgeMapping.py for UIDs of primary cloud and clients.
    
    There is also a synchronization barrier.

    Send model down to clients, receive data samples from clients all clients, train, send model back to clients and primary server, download model from primary server, repeat.

    """