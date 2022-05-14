import random

import pandas as pd

from thompson_sampling.model import *


def model_beta_visualisation():
    """
    Plot the graph of Beta distribution's pdf function for each arm after fitting the model.
    """
    data = {'B1': [random.randint(0, 1) for x in range(50)] + [1] * 50 + [0] * 50,  # generating sample data
            'B2': [random.randint(0, 1) for x in range(50)] + [0] * 100,
            'B3': [random.randint(0, 1) for x in range(150)]}
    model = BetaDistributionModel()  # instantiating the model
    data = pd.DataFrame(data)
    model.fit(data)
    model.plot_current_pdf()  # plotting the current pdf function


def model_normal_visualisation():
    """
    Plot the graph of Normal distribution's pdf function for each arm after fitting the model.
    """
    data = {'B1': [random.randint(0, 1) for x in range(50)] + [1] * 50 + [0] * 50,  # generating sample data
            'B2': [random.randint(0, 1) for x in range(50)] + [0] * 100,
            'B3': [random.randint(0, 1) for x in range(150)]}
    model = NormalDistributionModel()  # instantiating the model
    data = pd.DataFrame(data)
    model.fit(data)  # fitting the model
    model.plot_current_pdf(range_of_plot=(-1, 1), plot_frequency=100)  # plotting the current pdf function


if __name__ == "__main__":
    model_beta_visualisation()
    model_normal_visualisation()
