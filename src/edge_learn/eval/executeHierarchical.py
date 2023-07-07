import logging
from pathlib import Path
from shutil import copy
import datetime
import argparse
import os
import sys

from localconfig import LocalConfig
from torch import multiprocessing as mp

# Add the project root to sys.path before importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.node.Client import Client
from edge_learn.node.EdgeServer import EdgeServer
from edge_learn.node.PrimaryCloud import PrimaryCloud

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ta", type=int, help="Test after", default=20)
    parser.add_argument("-ld", type=str, help="Log directory", default="logs")
    parser.add_argument("-mid", type=int, help="Machine ID", default=0)
    parser.add_argument("-ps", type=int, help="Processes per machine", default=3)
    parser.add_argument("-ms", type=int, help="Number of machines", default=1)
    parser.add_argument("-its", type=int, help="Number of iterations", default=100)
    parser.add_argument("-cf", type=str, help="Config file", default="config.ini")
    parser.add_argument("-ll", type=str, help="Log level", default="INFO")
    parser.add_argument("-sm", type=int, help="Server machine", default=0)
    parser.add_argument("-bs", type=int, help="Batch size", default=32)
    parser.add_argument("-wsd", type=str, help="Weights store directory",
        default="./{}_ws".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    return parser.parse_args()
    
def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config

if __name__ == "__main__":
    args = parseArgs()

    Path(args.ld).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.cf)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.cf, args.ld)

    n_machines = args.ms
    procs_per_machine = args.ps
    
    m_id = args.mid

    sm = args.sm

    mapping = EdgeMapping(n_machines, procs_per_machine, sm, m_id)

    processes = []
    if sm == m_id:
        # Primary Cloud
        processes.append(
            mp.Process(
                target=PrimaryCloud,
                args=[
                    -1,
                    m_id,
                    mapping,
                    my_config,
                    args.its,
                    args.ld,
                    args.wsd,
                    log_level[args.ll],
                ],
            )
        )
    
    # Edge Server
    processes.append(
        mp.Process(
            target=EdgeServer,
            args=[
                0,
                m_id,
                mapping,
                my_config,
                args.ld,
                args.wsd,
                log_level[args.ll],
                args.ta,
            ],
        )
    )

    
    # Clients
    for r in range(1, procs_per_machine+1):
        processes.append(
            mp.Process(
                target=Client,
                args=[
                    r,
                    m_id,
                    mapping,
                    my_config,
                    args.ld,
                    log_level[args.ll],
                    args.bs,
                ],
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()


