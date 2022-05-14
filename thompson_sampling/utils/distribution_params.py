import pandas as pd

from thompson_sampling.model.base import BaseModel


def get_dist_params(model: BaseModel = None, data: pd.DataFrame = None, exploration_time: int = 10):
    """
    Dynamic plot for plotting beta distributions for a,b values given in a_b_lists.
    Parameters
    ----------
    model: NormalDistributionModel or BetaDistributionModel
        The model of the distribution.
    data : dict
        Each column is the label of one choice for which the distributions will be calculated over time.
        Each row represents a single timestamp.
    exploration_time: int
        The exploration time used for fitting the model.
    Returns
    -------
    dict
        Each key represents the label name and the values are arrays, each index having
        the distribution parameters of one timestamp.
    """
    dist_params = {k: [] for k in data.columns}
    exploration_time = exploration_time
    for i in range(len(data)):  # fitting data by each timestamp one by one
        model.fit(data.iloc[[i]], exploration_time=exploration_time)
        if i % exploration_time == 0:
            for j in range(len(list(data.columns))):
                dist_params[model.arm_labels[j]].append(model.arm_dist_params[j])  # getting distribution parameters
    return dist_params
