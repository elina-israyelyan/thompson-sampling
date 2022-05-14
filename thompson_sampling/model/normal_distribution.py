import logging
import pickle
import random
from statistics import mode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from thompson_sampling.model.base import BaseModel


class NormalDistributionModel(BaseModel):
    """
    Model for implementing Thompson Sampling with Normal distributions.
    """

    def __init__(self):
        super().__init__()
        self.arm_dist_params = None
        self.sum_of_rewards = None
        self.arm_labels = []
        self.predicted_best_rewards = None

    @property
    def arm_labels(self):
        return self._arm_labels

    @arm_labels.setter
    def arm_labels(self, arm_labels):
        self.number_of_plays = [0] * len(arm_labels)
        self.arm_dist_params = [(0, 1)] * len(arm_labels)  # giving the starting number_of_plays, means and sigmas
        self.sum_of_rewards = [0] * len(arm_labels)
        self._arm_labels = arm_labels

    def fit(self, data: pd.DataFrame, prefit: bool = True, exploration_time: int = 10):
        """
        Method to fit the data to the Binomial Thompson Sampling model
        Parameters
        ----------
        data : pandas.DataFrame
            Data to fit the model.
        prefit : bool
            If True use the previous, trained  parameters of normal distribution for each arm.
        exploration_time: int
            The amount of time points to explore before updating the distribution parameters.
        Returns
        -------
        None
        """
        if not self.arm_labels or not prefit:
            self.arm_labels = data.columns.tolist()
        for i in range(len(data)):
            best_arm_label = self.predict()  # get best reward giving arm
            best_arm = self.arm_labels.index(best_arm_label)
            try:
                reward = data[best_arm_label].tolist()[i]
            except KeyError as e:
                logging.warning("best arm selected was not in the new data, "
                                "so we dont know if there is a reward or not")
                continue
            self.number_of_plays[best_arm] += 1
            self.sum_of_rewards[best_arm] += reward
            if sum(self.number_of_plays) % exploration_time == 0:
                # check if exploration time is over and update the parameters
                for arm in range(len(self.arm_labels)):
                    number_of_plays = self.number_of_plays[arm]
                    self.arm_dist_params[arm] = (
                        (1 / (number_of_plays + 1)) * self.sum_of_rewards[arm], 1 / (number_of_plays + 1))

    def predict(self):
        """
        Predict which arm is the most reward bringing at current time.
        Returns
        -------
        str
            The name of the arm which gave the most probability to have a reward.
        """
        max_value = float("-inf")
        best_arm = -1
        for arm in range(len(self.arm_labels)):  # for each arm get a sample from its distribution
            myu, sigma = self.arm_dist_params[arm]
            arm_reward = np.random.normal(myu, sigma)
            if arm_reward > max_value:  # check if current arm gave the maximum reward rate and update the maximum.
                max_value = arm_reward
                best_arm = arm
        return self.arm_labels[best_arm]

    def predict_reward(self):
        """
        Predict which arm is the most reward bringing at current time.
        Returns
        -------
        str
           The name of the arm which gave the most probability to have a reward.
        """
        max_value = float("-inf")
        best_arm = -1
        for arm in range(len(self.arm_labels)):
            myu, sigma = self.arm_dist_params[arm]
            arm_reward = np.random.normal(myu, sigma)
            if arm_reward > max_value:
                max_value = arm_reward
                best_arm = arm
        return self.arm_labels[best_arm], max_value

    def save_model(self, save_path: str = "./", version: str = "latest"):
        """
        Save the model parameters in the mentioned path.
        Parameters
        ----------
        save_path : str
            Path where the model needs to be saved.
        version : str
            The version suffix which will be added to the model path.

        Returns
        -------
        None
            Saves the model in the save_path.

        """
        with open(save_path + "model_" + version + ".pkl", 'wb') as f:  # pickle the model's parameters
            pickle.dump({
                "arm_dist_params": self.arm_dist_params,
                "arm_labels": self.arm_labels,
                "number_of_plays": self.number_of_plays,
                "sum_of_rewards": self.sum_of_rewards}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, load_path: str = "./", version: str = "latest"):
        """
        Load model from the mentioned path.
        Parameters
        ----------
        load_path : str
            Path from which the model should be loaded.
        version : str
            The version of the model which should be loaded.

        Returns
        -------
        None
            Loads the parameters of the model from the path.
        """
        with open(load_path + "model_" + version + ".pkl", 'rb') as f:  # load pickle file
            model = pickle.load(f)
        self.arm_dist_params, self.arm_labels, self.number_of_plays, self.sum_of_rewards = (
            model["arm_dist_params"],
            model["arm_labels"],
            model["number_of_plays"],
            model["sum_of_rewards"])  # update parameters

    def plot_current_pdf(self, label: str = None, range_of_plot: tuple = (-1, 1), plot_frequency: int = 100):
        """
        Plot the pdf function of the current timestamp.
        Parameters
        ----------
        label : str
            The label of the arm that needs to be plotted. If None the distributions of all labels will be plotted.
        range_of_plot: tuple
            The range of figure for which the plot needs to be shown.
        plot_frequency: int
            The frequency of plot on the figure.

        Returns
        -------
        None
        """
        fig = go.Figure()
        if label is None:
            for label_name, dist_args in zip(self.arm_labels, self.arm_dist_params):
                hexadecimal = ["#" + ''.join([random.choice('ABCDEF0123456789')
                                              for _ in range(6)])][0]  # choosing random colors for each label
                distribution_func = norm(*dist_args)  # get the distribution
                fig.add_trace(go.Scatter(visible=True,  # create the figure of distribution's pdf
                                         line=dict(color=hexadecimal, width=6),
                                         name=label_name,
                                         x=np.linspace(*range_of_plot, 100),
                                         y=distribution_func.pdf(np.linspace(*range_of_plot, plot_frequency))))

        else:
            ind_of_arm = self.arm_labels.index(label)
            hexadecimal = ["#" + ''.join([random.choice('ABCDEF0123456789') for _ in range(6)])][0]
            dist_args = self.arm_dist_params[ind_of_arm]
            distribution_func = norm(*dist_args)
            fig.add_trace(go.Scatter(visible=True,
                                     line=dict(color=hexadecimal, width=6),
                                     name=label,
                                     x=np.linspace(*range_of_plot, 100),
                                     y=distribution_func.pdf(np.linspace(*range_of_plot, plot_frequency))))
        fig.show()

    def get_best_reward(self, n: int = 200):
        """
        Get best reward along n predictions.
        Parameters
        ----------
        n : int
            The number of iterations for prediction.
        Returns
        -------
        str
            The list of best rewards for each iteration is saved to self.predicted_best_rewards
            and the best reward is returned.
        """
        predicts_best = [self.predict() for _ in range(n)]  # do prediction n times
        self.predicted_best_rewards = predicts_best
        return mode(predicts_best)

    def plot_best_rewards(self):
        """
        Plot the histogram of best rewards for n predicts.
        Returns
        -------
        go.Figure
            The histogram of best rewards
        """
        predicts_best = self.predicted_best_rewards  # get the n predictions from best reward calculation
        fig = go.Figure(data=[go.Histogram(x=predicts_best)])  # make a histogram
        return fig
