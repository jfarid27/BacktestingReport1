import vectorbt as vbt
from vectorbt.portfolio import Portfolio
import numpy as np
from BaseAnalysis import BaseAnalysis

def divergence_indicator(price_data, level=2):
    """
    Calculates the divergence indicator for a given price data.
    Parameters:
    - price_data (list of tuples): List of tuples containing the price data for two pairs.
    - level (int, optional): The level at which to consider a divergence. Default is 2.
    Returns:
    - entries (numpy.ndarray): Boolean array indicating the entries where a divergence occurs.
    """
    pair1 = np.array([val[0] for val in price_data])
    pair2 = np.array([val[1] for val in price_data])
    
    percent_change_pair1 = np.diff(pair1) / pair1[:-1] * 100
    percent_change_pair2 = np.diff(pair2) / pair2[:-1] * 100

    divergences = np.divide(percent_change_pair1, percent_change_pair2)

    entries = np.logical_or(divergences > level, divergences < 1/level)

    return entries


class PairTradingAnalysis(BaseAnalysis):
    """Class to perform correlated pair analysis on price_data."""
    pass

    def _generate_pairs(self):
        """Returns strongly correlated pairs from the price_data."""
        pass

    