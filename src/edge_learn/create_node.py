import argparse
from localconfig import LocalConfig
import json
import logging

from edge_learn.mappings.EdgeMapping import EdgeMapping
from edge_learn.node.Client import Client
from edge_learn.node.EdgeServer import EdgeServer
from edge_learn.node.PrimaryCloud import PrimaryCloud


def create_node(node_type, rank, config_dir):
    config = read_ini("{}/config.ini".format(config_dir))
    params = read_params("{}/params.json".format(config_dir))
    mapping = EdgeMapping(
        params["number_of_machines"],
        params["clients_per_machine"],
        params["cloud_machine_id"],
        params["machine_id"],
    )

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    cur_log_level = log_level[params["log_level"]]

    if node_type == "cloud":
        PrimaryCloud(
            rank,
            params["machine_id"],
            mapping,
            config,
            params["number_of_iterations"],
            "/results",
            "/results",
            cur_log_level,
            params["test_frequency"],
            params["train_batch_size"],
            params["learning_mode"],
            params["num_threads_cloud"],
        )
    elif node_type == "edge":
        EdgeServer.create(
            rank,
            params["machine_id"],
            mapping,
            config,
            "/results",
            "/results",
            cur_log_level,
            params["train_batch_size"],
            params["learning_mode"],
            params["num_threads_edge"],
        )
    elif node_type == "client":
        Client.create(
            rank,
            params["machine_id"],
            mapping,
            config,
            "/results",
            cur_log_level,
            params["batch_size_to_send_to_edge"],
            params["learning_mode"],
            params["num_threads_cloud"],
        )
    else:
        print(f"Invalid node type: {node_type}")
        return


def read_ini(file_path):
    config = LocalConfig(file_path)
    parsed_config = dict()
    for section in config:
        parsed_config[section] = dict(config.items(section))
    print(parsed_config)
    return parsed_config


def read_params(file_path):
    with open(file_path, "r") as f:
        result = json.load(f)
        print(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_type", help="Type of node to create (cloud, edge, client)"
    )
    parser.add_argument("--config_dir", help="Path to configuration files")
    parser.add_argument("--rank", type=int, help="Rank of node")
    args = parser.parse_args()

    create_node(args.node_type, int(args.rank), args.config_dir)
