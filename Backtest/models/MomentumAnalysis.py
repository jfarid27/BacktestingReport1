import vectorbt as vbt
from vectorbt.portfolio import Portfolio
from Backtest.models.Analysis import BaseAnalysis


class MomentumAnalysis(BaseAnalysis):
    """Class to perform momentum analysis on price_data."""

    def __init__(self, price_data):
        """Initialize the MomentumAnalysis class.

        Args:
            price_data (Any): The price data on which the analysis is to be performed.
        """
        self.price_data = price_data
        self.portfolio = None

    def _MAStrategy(self, short_window: int=15, long_window: int=50):
        """Return vbt entries and exits after applying MA strategy on price_data.

        This method calculates the moving averages (MA) for the given short and long windows
        and generates entry and exit signals based on the crossover of these moving averages.

        Args:
            short_window (int): The window size for the short-term moving average.
            long_window (int): The window size for the long-term moving average.

        Returns:
            list: A list containing two elements:
                - entries: A boolean array indicating where the short-term MA crosses above the long-term MA.
                - exits: A boolean array indicating where the short-term MA crosses below the long-term MA.
        """

        fast_ma = vbt.MA.run(self.price_data, short_window, short_name='fast')
        slow_ma = vbt.MA.run(self.price_data, long_window, short_name='slow')
        entries = fast_ma.ma_above(slow_ma)
        exits = fast_ma.ma_below(slow_ma)
        return (entries, exits)

    def MomentumBasedLongOnly(self, init_cash: float = 100000, overwrite: bool = False):
        """Return vbt portfolio object after applying MA strategy on price_data.

        Long only strategy. When the fast moving average is above the slow moving average,
        enters long position. If portfolio is already created, returns the same portfolio
        unless overwrite is True. Assumes total available cash is shared among all assets.

        Args:
            init_cash (float): Initial cash to be used for the portfolio. Defaults to 100000.
            overwrite (bool): If True, overwrite the existing portfolio even if it exists. Defaults to False.

        Returns:
            Portfolio: The portfolio object after applying the MA strategy.
        """

        if self.portfolio is None or overwrite:
            (entries, exits) = self._MAStrategy(15, 50)
            self.portfolio = Portfolio.from_signals(
                self.price_data,
                entries,
                exits, 
                init_cash=init_cash,
                cash_sharing=True
            )
        return self.portfolio
    
    def MomentumBasedLongShort(self, init_cash: float = 100000, overwrite: bool = False):
        """Return vbt portfolio object after applying MA strategy on price_data.

        Long short strategy. When the fast moving average is above the slow moving average,
        enters long position and closes shorts, and vice versa. If portfolio is already created,
        returns the same portfolio unless overwrite is True. Assumes total available cash
        is shared among all assets.

        Args:
            init_cash (float): Initial cash to be used for the portfolio. Defaults to 100000.
            overwrite (bool): If True, overwrite the existing portfolio even if it exists. Defaults to False.

        Returns:
            Portfolio: The portfolio object after applying the MA strategy.
        """

        if self.portfolio is None or overwrite:
            (entries, exits) = self._MAStrategy(10, 50)
            (short_entries, short_exits) = (exits, entries)
            self.portfolio = Portfolio.from_signals(
                self.price_data,
                entries,
                exits, 
                short_entries,
                short_exits,
                init_cash=init_cash,
                cash_sharing=True
            )
        return self.portfolio