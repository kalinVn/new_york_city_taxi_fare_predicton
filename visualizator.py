import matplotlib
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")


def hist_plot(column, params=None, xlabel='', title=''):
    column.plot.hist(**params)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def scatter_plot(df, params=None):
    df.plot.scatter(*params)
    plt.show()


