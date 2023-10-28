import json
import random
import matplotlib.pyplot as plt
import argparse

# Constants for simulation
TRAIN_TIME_MIN = 0.7 / 60
TRAIN_TIME_MAX = 0.71 / 60
COMMUNICATION_LATENCY_MIN = 0.2 / 60
COMMUNICATION_LATENCY_MAX = 0.4 / 60
MODEL_SIZE = 1  # in MB
BANDWIDTH_MIN = 5.0 * 60  # in MB/s
BANDWIDTH_MAX = 10.0 * 60  # in MB/s

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Plot test_loss and test_acc against simulated time."
)
parser.add_argument("file_path", type=str, help="Path to the JSON file")
parser.add_argument("output_plot", type=str, help="Path to the output PDF plot file")
args = parser.parse_args()

# Read the JSON file
with open(args.file_path, "r") as json_file:
    data = json.load(json_file)

test_loss_dict = data["test_loss"]
test_acc_dict = data["test_acc"]

# Extract round numbers and values from the dictionaries
round_numbers = [int(round_num) for round_num in test_loss_dict.keys()]
test_loss_values = [test_loss_dict[str(round_num)] for round_num in round_numbers]
test_acc_values = [test_acc_dict[str(round_num)] for round_num in round_numbers]

# Initialize lists for simulated time, loss, and accuracy
simulated_time = []
simulated_loss = []
simulated_acc = []

current_time = 0.0
last_round_num = 0

# Simulate time and communication for each round
for i in range(len(round_numbers)):
    round_num = round_numbers[i]
    # Add random training time
    training_time = random.uniform(TRAIN_TIME_MIN, TRAIN_TIME_MAX)
    current_time += (round_num - last_round_num) * training_time

    # Add random communication latency
    communication_latency = random.uniform(
        COMMUNICATION_LATENCY_MIN, COMMUNICATION_LATENCY_MAX
    )
    current_time += (round_num - last_round_num) * communication_latency

    # Calculate time to send model
    bandwidth = random.uniform(BANDWIDTH_MIN, BANDWIDTH_MAX)
    time_to_send_model = MODEL_SIZE / bandwidth
    current_time += (round_num - last_round_num) * time_to_send_model

    # Add simulated data to lists
    simulated_time.append(current_time)
    simulated_loss.append(test_loss_values[i])
    simulated_acc.append(test_acc_values[i])
    last_round_num = round_num

fig, ax1 = plt.subplots(figsize=(10, 6))

color = "tab:red"
ax1.set_xlabel("Simulated Time (minutes)")
ax1.set_ylabel("Accuracy (%)", color=color)
ax1.plot(simulated_time, simulated_acc, label="test_acc", color=color, marker="o")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel(
    "Loss (CrossEntropy)", color=color
)  # we already handled the x-label with ax1
ax2.plot(simulated_time, simulated_loss, label="test_loss", color=color, marker="x")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("SHAKESPEARE Hybrid Post-Processed Time")

fig.tight_layout()  # ensure that the different y-axes labels are not overlapping

# Save the plot to a PDF file
plt.savefig(args.output_plot)

# Close the plot to release resources
plt.close()
