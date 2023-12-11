import json
import argparse
import random

# HYPERPARAMETERS
# Bandwith in MB/s and Latency in seconds
# Assuming bidirectional performance
CLIENT_EDGE_LATENCY = [0.01, 0.07]
CLIENT_EDGE_BANDWIDTH = [37.5, 125]
EDGE_CLOUD_LATENCY = [0.06, 0.25]
EDGE_CLOUD_BANDWIDTH = [300, 600]
WARMUP = 10

# PARSING
parser = argparse.ArgumentParser(description="Take data and turn into time series")
parser.add_argument("dir_path", type=str, help="Path to folder with files")
args = parser.parse_args()


def read_json(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found. Will work without.")
        return None


CLOUD = read_json(f"{args.dir_path}/cloud.json")
EDGE = read_json(f"{args.dir_path}/edge.json")
CLIENT = read_json(f"{args.dir_path}/client.json")


def include_warmup_to_time(round_time_dict):
    post_warmup_rounds = {k: v for k, v in round_time_dict.items() if int(k) > WARMUP}
    total_time_across_non_warmup = sum(post_warmup_rounds.values())
    average_round_time = total_time_across_non_warmup / len(post_warmup_rounds)

    for i in range(1, WARMUP + 1):
        round_time_dict[str(i)] = average_round_time
    return round_time_dict


def calculate_time(data_size, latency_range, bandwidth_range):
    latency = random.uniform(*latency_range)
    bandwidth_time = data_size / random.uniform(*bandwidth_range) / 1e6
    return latency + bandwidth_time


def create_time_series():
    test_acc_dict = CLOUD["test_acc"]
    round_time = include_warmup_to_time(CLOUD["total_elapsed_time"])
    bytes_sent_to_edge_from_client = CLIENT["bytes_sent_to_edge"] if CLIENT else None
    bytes_sent_to_cloud_from_edge = EDGE["bytes_sent_to_cloud"] if EDGE else None
    bytes_sent_to_edge_from_cloud = CLOUD["bytes_sent_to_each_edge"]
    bytes_sent_to_client_from_edge = EDGE["bytes_sent_to_each_client"] if EDGE else None
    last_round_time_included: int = 1
    total_time: float = 0
    values = {}

    for i in test_acc_dict.keys():
        i_int = int(i)
        while last_round_time_included < i_int:
            last_round_time_included += 1
            total_time += round_time[str(last_round_time_included)]
            if bytes_sent_to_edge_from_client:
                total_time += calculate_time(
                    bytes_sent_to_edge_from_client[str(last_round_time_included)],
                    CLIENT_EDGE_LATENCY,
                    CLIENT_EDGE_BANDWIDTH,
                )
            if bytes_sent_to_cloud_from_edge:
                total_time += calculate_time(
                    bytes_sent_to_cloud_from_edge[str(last_round_time_included)],
                    EDGE_CLOUD_LATENCY,
                    EDGE_CLOUD_BANDWIDTH,
                )
            total_time += calculate_time(
                bytes_sent_to_edge_from_cloud[str(last_round_time_included)],
                EDGE_CLOUD_LATENCY,
                EDGE_CLOUD_BANDWIDTH,
            )
            if bytes_sent_to_client_from_edge:
                total_time += calculate_time(
                    bytes_sent_to_client_from_edge[str(last_round_time_included)],
                    CLIENT_EDGE_LATENCY,
                    CLIENT_EDGE_BANDWIDTH,
                )

        values[total_time] = test_acc_dict[i]

    return list(values.keys()), list(values.values())


def main():
    with open("output.txt", "w") as file:
        keys, values = create_time_series()
        first_line = True
        for i in range(0, len(keys)):
            if not first_line:
                file.write("\n")
            else:
                first_line = False
            file.write(f"{keys[i]} {values[i]}")


main()
