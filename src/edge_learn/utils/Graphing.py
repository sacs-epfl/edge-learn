from matplotlib import pyplot as plt


def create_and_save_plot(
    title: str, dict: dict, xlabel: str, ylabel: str, filename: str
):
    plt.clf()
    y_axis = [dict[key] for key in dict.keys()]
    x_axis = list(map(int, dict.keys()))
    plt.plot(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
