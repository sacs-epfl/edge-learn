from decentralizepy.node.DPSGDNodeFederated import DPSGDNodeFederated
from decentralizepy.training.Training import Training
from decentralizepy.communication.TCP import TCP

class Client(DPSGDNodeFederated):
    """
    Perform training and exchange data and models with other the (closest) edge servers.

    TCP conveniently measures the bytes exchanged.
    """
