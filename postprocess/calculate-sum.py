import argparse
import json

parser = argparse.ArgumentParser(description="Get sum")
parser.add_argument("file_path", type=str, help="Path to file")
parser.add_argument("key", type=str, help="key")
parser.add_argument("-d", "--divisor", type=float, help="What to divide sum by")
args = parser.parse_args()

with open(args.file_path, "r") as file:
    js = json.load(file)
    x = [float(xx) for xx in js[args.key].values()]
    sum_x = sum(x)
    print(f"sum: {sum_x}")
    if args.divisor:
        print(f"div: {sum_x / args.divisor}")
