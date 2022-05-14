import random

import numpy as np
import plotly.graph_objects as go
from scipy.stats import rv_continuous


def plot_dist_over_time(dist_params: dict, distribution: rv_continuous, range_of_plot: tuple = (0, 1)):
    """
    Plotting distribution's pdf function as a dynamic graph.
    Parameters
    ----------
    dist_params : dict
        Each key is the label of one choice for which the given distributions will be plotted.
        Each row represents a timestamp.
    distribution : rv_continuous
        The distribution type to plot.
    range_of_plot: tuple
        The range of interval for which the plot should be shown.
    Returns
    -------
    plotly.graph_objects.Figure()
        The figure that combines all the distributions' pdf function over time, for different labels.
    """

    fig = go.Figure()
    for label_name, args_list in dist_params.items():
        hexadecimal = ["#" + ''.join([random.choice('ABCDEF0123456789')
                                      for _ in range(6)])][0]  # choosing random colors for each label
        for args in args_list:
            distribution_func = distribution(*args)
            fig.add_trace(go.Scatter(visible=False,
                                     line=dict(color=hexadecimal, width=6),
                                     name=label_name,
                                     x=np.linspace(*range_of_plot, 100),
                                     y=distribution_func.pdf(np.linspace(*range_of_plot, 100))))

    # Create and add slider
    steps = []
    for i in range(round(len(fig.data) / len(dist_params.keys()))):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "PDF function switched to timestamp: " + str(i)}],  # layout attribute
        )
        for j in range(
                len(dist_params.keys())):  # to get the traces of the same timestamp (that's why we use i+j*len)
            step["args"][0]["visible"][
                i + j * len(list(dist_params.values())[0])] = True  # make the trace visible
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Timestamp: "},
        pad={"t": 100},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig
