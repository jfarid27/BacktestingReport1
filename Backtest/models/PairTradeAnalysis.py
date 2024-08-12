import vectorbt as vbt
from vectorbt.portfolio import Portfolio
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Tuple
from Backtest.models.Analysis import BaseAnalysis

def divergence_indicator(price_data, level=2):
    """
    Calculates the divergence indicator for a given price data. Assumes two pairs and returns
    entries and exits respective of the first item in the pair.

    Args:
        price_data (list of tuples): List of tuples containing the price data for two pairs.
        level (int, optional): The level at which to consider a divergence. Default is 2.
    
    Returns:
        entries (numpy.ndarray): Boolean array indicating the entries where a divergence occurs.
    """

    pair1 = np.array([val[0] for val in price_data])
    pair2 = np.array([val[1] for val in price_data])
    
    percent_change_pair1 = np.diff(pair1) / pair1[:-1] * 100
    percent_change_pair2 = np.diff(pair2) / pair2[:-1] * 100
    divergences = np.divide(percent_change_pair1, percent_change_pair2)
    
    entries = divergences < 1/level
    exits = divergences > level

    return (entries, exits)

def generate_pairs(price_data, corr_threshold=0.7):
    """
    Returns labels of strongly correlated pairs from the price_data.

    This function calculates the percentage changes of the given price data,
    computes the correlation matrix of these changes, and identifies pairs of
    assets with a correlation coefficient greater than 0.7. It returns the labels
    of these strongly correlated pairs.

    Args:
        price_data (pd.DataFrame): A pandas DataFrame where each column represents
                                   the price data of an asset over time.
        corr_threshold (float): The correlation threshold above which pairs are
                                considered strongly correlated. Default is 0.7.

    Returns:
        list of tuple: A list of tuples, where each tuple contains the labels of
                       two strongly correlated assets.
    """

    percent_changes = price_data.pct_change().dropna()
    corr_matrix = percent_changes.corr()
    pairs = np.where(corr_matrix > corr_threshold)
    pairs = [(i, j) for i, j in zip(*pairs) if i < j]
    asset_labels = [(corr_matrix.index[i], corr_matrix.columns[j]) for i, j in pairs]
    return asset_labels

class PairTradeAnalysis(BaseAnalysis):
    """Class to perform correlated pair analysis on price_data."""
    portfolio: Optional[Portfolio] = None
    price_data: Optional[List[Any]] = None
    
    def __init__(self, price_data):
        """Initialize the MomentumAnalysis class.

        Args:
            price_data (Any): The price data on which the analysis is to be performed.
        """
        self.price_data = price_data
        self.portfolio = None
    
    def PairCorrLongOnly(self, pairs: Tuple, portfolio_cash: float = 100000, overwrite: bool = False):
        """Return vbt portfolio object after applying pair diversion strategy on price_data.

        Long only strategy. When the price ratio of two assets is below the lower bound,
        enters long position. Assumes total available cash is shared among all assets.

        Returns:
            Portfolio: The portfolio object after applying the pair correlation strategy.
        """

        if self.portfolio is None or overwrite:
            (asset1, asset2) = pairs
            pair_price_data = self.price_data[[asset1, asset2]]
            
            (entries1, exits1) = divergence_indicator(pair_price_data.values)

            entries = pd.DataFrame({
                asset1: entries1,
                asset2: exits1
            })

            exits = pd.DataFrame({
                asset1: exits1,
                asset2: entries1
            })

            pair_price_data = pair_price_data.drop(pair_price_data.index[0])

            self.portfolio = Portfolio.from_signals(
                pair_price_data,
                entries,
                exits,
                cash_sharing=True,
                init_cash=portfolio_cash,
            )

        return self.portfolio

    