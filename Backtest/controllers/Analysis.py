from vectorbt.portfolio import Portfolio
from typing import Any, Optional
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd

def plot_table_statistics(data: np.ndarray, **kwargs):   
    """
    Generate a nice looking table of basic statistics for the data.

    Parameters:
        - data (np.ndarray): The input data array.

        Returns:
        None
    """

    benchmark_stats = stats.describe(data)

    table_data = [
        ["Number of Observations", benchmark_stats.nobs],
        ["Minimum", benchmark_stats.minmax[0]],
        ["Maximum", benchmark_stats.minmax[1]],
        ["Mean", benchmark_stats.mean],
        ["Variance", benchmark_stats.variance],
        ["Skewness", benchmark_stats.skewness],
        ["Kurtosis", benchmark_stats.kurtosis]
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Statistic", "Value"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[[row[0] for row in table_data], [row[1] for row in table_data]],
                fill_color='lavender',
                align='left'))
    ])

    fig.show()

def plot_statistics(data: np.ndarray, target: float = None, **kwargs):
    """
    Plots the histogram of the data and overlays the standard normal distribution curve.

    Args:
        data (np.ndarray): The data to plot. It should be a 1-dimensional array of standard normal variables.
        **kwargs: Additional keyword arguments to pass to the figure's layout. These can be any valid Plotly layout options.

    """
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.sqrt(variance)

    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    fitted_normal_curve = stats.norm.pdf(x, mean, std_dev)

    mean = np.mean(data)

    normal_curve = go.Scatter(
        x=x,
        y=fitted_normal_curve,
        mode='lines',
        name='Fit Normal Distribution',
        line=dict(color='red')
    )

    histogram = go.Histogram(
        x=data,
        nbinsx=30,
        histnorm='probability density',
        name='Data',
        opacity=0.75
    )

    fig = go.Figure(data=[histogram, normal_curve])

    fig.add_shape(
        type="line",
        x0=mean,
        y0=0,
        x1=mean,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color="green", width=2, dash="dash")
    )

    fig.add_annotation(
        x=mean,
        y=1.02,
        xref='x',
        yref='paper',
        text="mean",
        showarrow=False,
        font=dict(color="green")
    )

    if target:
        fig.add_shape(
            type="line",
            x0=target,
            y0=0,
            x1=target,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color="blue", width=2, dash="dash")
        )

        fig.add_annotation(
            x=target,
            y=1.02,
            xref='x',
            yref='paper',
            text="target",
            showarrow=False,
            font=dict(color="blue")
        )

    fig.update_layout(
        xaxis_title='Value',
        yaxis_title='Density',
        bargap=0.2,
        **kwargs
    )

    fig.show()

def plot_datetime_splits(data: np.ndarray, **kwargs):
    datetime_ranges = [(pd.to_datetime(split[0]), pd.to_datetime(split[-1])) for split in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (start, end) in enumerate(datetime_ranges):
        ax.plot([start, end], [i, i], color='blue', marker='o', linewidth=5)

    ax.set_yticks(range(len(datetime_ranges)))
    ax.set_yticklabels([f'Range {i+1}' for i in range(len(datetime_ranges))])

    ax.xaxis_date()
    fig.autofmt_xdate()

    ax.set_xlabel('Date')
    ax.set_ylabel('Ranges')
    ax.set_title('Datetime Ranges')

class BaseAnalysis():
    price_data: Any
    portfolio: Optional[Portfolio] = None

    

