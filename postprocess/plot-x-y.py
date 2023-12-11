import matplotlib.pyplot as plt
import argparse

# PARSING
parser = argparse.ArgumentParser(
    description="Plot testing accuracy over time across different organisations"
)
parser.add_argument(
    "title", type=str, help="Path to folder with all subfolders organised with data"
)
parser.add_argument("x_label", type=str, help="Label for x axis")
parser.add_argument("y_label", type=str, help="Label for y axis")
parser.add_argument(
    "file_paths_with_colors_and_legend_name",
    nargs="*",
    type=str,
    help="List of file paths with their corresponding colors and legend name, separated by a comma. For example: 'path/to/file1,red,CIFAR path/to/file2,blue,ImageNet'",
)
args = parser.parse_args()

file_paths_colors_legends = []
for triplet in args.file_paths_with_colors_and_legend_name:
    try:
        file_path, color, legend_name = triplet.split(",")
        file_paths_colors_legends.append((file_path, color, legend_name))
    except ValueError:
        print(
            f"Error: Each file path must be accompanied by a color, separated by a comma. Received: {pair}"
        )


def read_x_y_file(filename):
    try:
        with open(filename, "r") as file:
            x, y = [], []
            for line in file.readlines():
                splt = line.split(" ")
                x.append(splt[0])
                y.append(splt[1])
            return x, y
    except FileNotFoundError:
        print(f"Could not find file: {filename}")
        exit(1)


def main():
    for file_path, color, legend_name in file_paths_colors_legends:
        x, y = read_x_y_file(file_path)
        plt.plot(
            [float(xx) for xx in x],
            [float(yy) for yy in y],
            color=color,
            label=legend_name,
        )
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.legend()
    plt.title(args.title)
    plt.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.8)
    plt.savefig(f"output/output.png", dpi=350)


main()
