import json

import matplotlib.pyplot as plt
import argparse


# PARSING
parser = argparse.ArgumentParser(
    description="Plot testing accuracy over time across different organisations"
)
parser.add_argument(
    "dir_path", type=str, help="Path to folder with all subfolders organised with data"
)
parser.add_argument("title", type=str, help="Name to give plot")
args = parser.parse_args()


def read_json(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found. Quitting.")
        exit(1)


# LOADING ALL THE DATA
BASELINE_CLOUD = read_json(f"{args.dir_path}/baseline/cloud/stats.json")
HYBRID_CLOUD = read_json(f"{args.dir_path}/hybrid/cloud/stats.json")
HYBRID_EDGE = read_json(f"{args.dir_path}/hybrid/edge/stats.json")
HYBRID_CLIENT = read_json(f"{args.dir_path}/hybrid/client/stats.json")
ONLY_DATA_CLOUD = read_json(f"{args.dir_path}/only_data/cloud/stats.json")
ONLY_DATA_EDGE = read_json(f"{args.dir_path}/only_data/edge/stats.json")
ONLY_DATA_CLIENT = read_json(f"{args.dir_path}/only_data/client/stats.json")
ONLY_WEIGHTS_CLOUD = read_json(f"{args.dir_path}/only_weights/cloud/stats.json")
ONLY_WEIGHTS_EDGE = read_json(f"{args.dir_path}/only_weights/edge/stats.json")
ONLY_WEIGHTS_CLIENT = read_json(f"{args.dir_path}/only_weights/client/stats.json")

DATA = {
    "HYBRID": {"CLOUD": HYBRID_CLOUD, "CLIENT": HYBRID_CLIENT, "EDGE": HYBRID_EDGE},
    "ONLY_DATA": {
        "CLOUD": ONLY_DATA_CLOUD,
        "CLIENT": ONLY_DATA_CLIENT,
        "EDGE": ONLY_DATA_EDGE,
    },
    "ONLY_WEIGHTS": {
        "CLOUD": ONLY_WEIGHTS_CLOUD,
        "CLIENT": ONLY_WEIGHTS_CLIENT,
        "EDGE": ONLY_WEIGHTS_EDGE,
    },
}


def get_baseline_x_y():
    test_acc_dict = BASELINE_CLOUD["test_acc"]
    round_time = include_warmup_to_time(BASELINE_CLOUD["total_elapsed_time"])
    last_round_time_included: int = 0
    total_time: float = 0
    values = {}

    for i in test_acc_dict.keys():
        i_int = int(i)
        while last_round_time_included + 1 < i_int:
            last_round_time_included += 1
            total_time += round_time[str(last_round_time_included)]
        values[total_time] = test_acc_dict[i]

    return list(values.keys()), list(values.values())


x_baseline, y_baseline = get_baseline_x_y()
x_hybrid, y_hybrid = get_x_y_from_category("HYBRID")
x_only_data, y_only_data = get_x_y_from_category("ONLY_DATA")
x_only_weights, y_only_weights = get_x_y_from_category("ONLY_WEIGHTS")

# PLOTTING
x_baseline_minutes = [x / 60 for x in x_baseline]
x_hybrid_minutes = [x / 60 for x in x_hybrid]
x_only_data_minutes = [x / 60 for x in x_only_data]
x_only_weights_minutes = [x / 60 for x in x_only_weights]

plt.plot(x_baseline_minutes, y_baseline, color="blue", label="Baseline")
plt.plot(x_hybrid_minutes, y_hybrid, color="red", label="Hybrid")
plt.plot(x_only_data_minutes, y_only_data, color="yellow", label="Only-Data")
plt.plot(x_only_weights_minutes, y_only_weights, color="green", label="Only-Weights")
plt.xlabel("Time (minutes)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title(args.title)
plt.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.8)
plt.savefig(f"output/output.png", dpi=350)
