import random

import pandas as pd
from scipy.stats import norm

from model import *
from utils.distribution_params import get_dist_params
from visualisation.dynamic_plots import plot_dist_over_time


def model_normal_visualisation():
    """
    Example for plotting a dynamic graph assuming the data should have a normal distribution.
    The plot has the graphs of normal distribution's pdf function for all the time points
    """
    data = {'B1': [0.5] * 50 + [0.9] * 100,  # generating data
            'B2': [random.uniform(0, 1) for x in range(50)] + [0.9] * 100,
            'B3': [random.uniform(0, 1) for x in range(150)]}
    model = NormalDistributionModel()  # instantiating the model
    data = pd.DataFrame(data)
    myu_sigma_lists = get_dist_params(model, data)  # get the myu and sigma parameters for all the time points
    fig = plot_dist_over_time(myu_sigma_lists, norm, (0, 1))  # plot
    fig.show()


if __name__ == "__main__":
    model_normal_visualisation()
