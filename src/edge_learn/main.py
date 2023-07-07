from torch import multiprocessing as mp

# Start with everything for one machine, then scale to multiple machines

if __name__ == "__main__":
    # Specify args similar to src/decentralizepy/utils.py
    
    # Instantiate the multiprocessing context

    # Define a format for how many and which processes to instantiate on which machine
    # Assign unique ids to each process (all kinds)
    # Look at src/decentralizepy/mappings

    # Instantiate 2^D processes for edge servers
    # Loot at eval/testing.py for how to instantiate processes

    # Instantiate 1 process for the primary cloud on one of the machines

    # Instantiate N client processes
    