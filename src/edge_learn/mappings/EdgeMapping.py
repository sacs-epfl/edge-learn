import logging

from decentralizepy.mappings.Mapping import Mapping
from edge_learn.enums.LearningMode import LearningMode


class EdgeMapping(Mapping):
    """
    This class defines the mapping for edge-learn
    n_machines edge servers, each with local rank 0  and uid = 0 .. n_machines-1
    procs_per_machine clients with 1 -> procs_per_machine+1 ranks

    """

    def __init__(
        self,
        n_machines,
        procs_per_machine,
        learning_mode,
        global_service_machine=0,
        current_machine=0,
    ):
        """
        Constructor

        Parameters
        ----------
        n_machines : int
            Number of machines involved in learning
        procs_per_machine : list(int)
            A list of number of processes spawned per machine
        global_service_machine: int, optional
            Machine ID on which the server/services are hosted
        current_machine: int, optional
            Machine ID of local machine

        """

        super().__init__(n_machines * procs_per_machine + n_machines + 1)
        self.n_machines = n_machines
        self.edge_servers = n_machines
        self.num_clients = n_machines * procs_per_machine
        self.n_procs = self.num_clients + self.edge_servers + 1  # the singular cloud
        self.procs_per_machine = procs_per_machine
        self.local_clients = procs_per_machine
        self.global_service_machine = global_service_machine
        self.current_machine = current_machine
        self.learning_mode = LearningMode(learning_mode)

    def get_procs_per_machine(self):
        """
        Gives the number of processes per machine

        Returns
        -------
        int
            number of processes per machine

        """
        return self.procs_per_machine

    def get_num_clients(self):
        return self.num_clients

    def get_duid_from_machine_and_rank(self, machine, rank):
        uid = self.get_uid(rank, machine)
        duid = self.get_duid_from_uid(uid)
        assert duid >= 0
        return duid

    def get_uid(self, rank: int, machine_id: int):
        """
        Gives the global unique identifier of the node

        Parameters
        ----------
        rank : int
            Node's rank on its machine
        machine_id : int
            node's machine in the cluster

        Returns
        -------
        int
            the unique identifier

        """
        if rank < 0:
            return rank
        elif rank == 0:
            return machine_id
        return (
            self.edge_servers + machine_id * self.local_clients + rank - 1
        )  # -1 to account for rank 0

    def get_parents(self, uid: int) -> list:
        if uid < 0:
            return []
        elif uid < self.edge_servers:
            return [-1]
        else:
            _, machine_id = self.get_machine_and_rank(uid)
            return [machine_id]

    def get_children(self, uid: int) -> list:
        if uid < 0:
            return list(range(self.edge_servers))
        elif uid < self.edge_servers:
            return list(
                range(
                    self.edge_servers + uid * self.local_clients,
                    self.edge_servers
                    + uid * self.local_clients
                    + self.procs_per_machine,
                )
            )
        else:
            return []

    def does_uid_generate_data(self, uid: int) -> bool:
        if uid < self.n_machines:
            return False
        return True

    def does_uid_test_data(self, uid: int) -> bool:
        if uid >= self.n_machines:
            return False
        return True

    def get_duid_from_uid(self, uid: int) -> int:
        """
        Returns the data unique identifier, useful for the dataset
        """
        if self.learning_mode == LearningMode.BASELINE:
            if uid == -1:
                return 0
            else:
                return -1
        else:
            return uid - self.n_machines

    def get_number_of_nodes_read_from_dataset(self) -> int:
        if self.learning_mode == LearningMode.BASELINE:
            return 1
        else:
            return self.n_machines * self.procs_per_machine

    def get_machine_and_rank(self, uid: int):
        """
        Gives the rank and machine_id of the node

        Parameters
        ----------
        uid : int
            globally unique identifier of the node

        Returns
        -------
        2-tuple
            a tuple of rank and machine_id

        """
        if uid < 0:
            return uid, self.global_service_machine
        elif uid < self.edge_servers:
            return 0, uid
        uid -= self.edge_servers
        return (1 + uid % self.local_clients), (uid // self.local_clients)

    def get_local_procs_count(self):
        """
        Useless here, just 1 client performs learning. Gives number of processes that run on the node

        Returns
        -------
        int
            the number of local processes

        """

        return self.local_clients + 1
