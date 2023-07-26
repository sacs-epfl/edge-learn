import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--filename", type=str, help="Location of file")
parser.add_argument("--stat", type=str, help="The key in the json you want to average")
parser.add_argument(
    "--start", type=int, help="Index to start averaging from", default=0
)
parser.add_argument(
    "--skip", type=int, help="modulo int index to skip", default=100000000000
)
parser.add_argument(
    "--skip_offset",
    type=int,
    help="offset to subtract iteration number to calculate skip",
    default=0,
)

args = parser.parse_args()

with open(args.filename, "r") as file:
    data = json.load(file)

sum = 0
count = 0
for key, value in data[args.stat].items():
    if int(key) >= args.start and ((int(key) - args.skip_offset) % args.skip) != 0:
        count += 1
        sum += value

print(sum / count)
