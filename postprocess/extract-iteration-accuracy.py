import json
import argparse

parser = argparse.ArgumentParser(description="Take data and save as two column")
parser.add_argument("file_path", type=str, help="Path to cloud file json")
args = parser.parse_args()


def read_json(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found. Quiting.")
        exit(1)


def main():
    with open("output.txt", "w") as file:
        FILE = read_json(args.file_path)
        first_line = True
        for k, v in FILE["test_acc"].items():
            if not first_line:
                file.write("\n")
            else:
                first_line = False
            file.write(f"{k} {v}")


main()
