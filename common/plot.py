import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_scatter(xs, ys, labels, category_num):
    colors = cm.rainbow(np.linspace(0, 1, category_num))
    labels = np.array(labels)
    xs = np.array(xs)
    ys = np.array(ys)
    f, ax = plt.subplots(1)
    for i in range(category_num):
        cur_xs = xs[labels == i]
        cur_ys = ys[labels == i]
        plt.scatter(cur_xs, cur_ys, c=colors[i], label=i)
    ax.legend()
    plt.show()
