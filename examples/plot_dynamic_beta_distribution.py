import random

import pandas as pd
from scipy.stats import beta

from thompson_sampling.model import *
from thompson_sampling.utils.distribution_params import get_dist_params
from thompson_sampling.visualisation.dynamic_plots import plot_dist_over_time


def model_beta_visualisation():
    """
    Example for plotting a dynamic graph assuming the data should have a beta distribution.
    The plot has the graphs of beta distribution pdf function for all the time points
    """
    data = {'B1': [random.randint(0, 1) for x in range(50)] + [1] * 50 + [0] * 50,  # generating data
            'B2': [random.randint(0, 1) for x in range(50)] + [1] * 100,
            'B3': [random.randint(0, 1) for x in range(150)]}
    model = BetaDistributionModel()  # instantiating the model
    data = pd.DataFrame(data)
    a_b_lists = get_dist_params(model, data)  # get a b parameters for Beta distribution for each timestamp
    fig = plot_dist_over_time(a_b_lists, beta)  # plot the dynamic graph
    fig.show()


if __name__ == "__main__":
    model_beta_visualisation()
