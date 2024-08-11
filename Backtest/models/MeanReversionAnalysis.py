import vectorbt as vbt
from vectorbt.portfolio import Portfolio
from BaseAnalysis import BaseAnalysis


class MeanReversionAnalysis(BaseAnalysis):
    """Class to perform mean-reversion analysis on price_data."""

    def _MRStrategy(self, window: int=15, level: int=30):
        """Return vbt entries and exits after applying RSI strategy on price_data.

        This method calculates the Relative Strength Index (RSI) for the given window size
        and generates entry and exit signals based on the RSI crossing specified levels.

        Args:
            window (int): The window size for calculating the RSI. Defaults to 15.
            level (int): The RSI level used to generate entry and exit signals. Defaults to 30.

        Returns:
            list: A list containing two elements:
                - entries: A boolean array indicating where the RSI crosses below the specified level.
                - exits: A boolean array indicating where the RSI crosses above the level (100 - specified level).
        """


        rsi = vbt.RSI.run(self.price_data, window=window)
        entries = rsi.rsi_crossed_below(level)
        exits = rsi.rsi_crossed_above(100-level)
        return [entries, exits]
    
    def MeamReversionBasedLongOnly(self, init_cash: float = 100000, overwrite: bool = False):
        """Return vbt portfolio object after applying MR strategy on price_data.

        Long only strategy. When the rsi indicator has cross into oversold, enters positions with
        available cash, and vice versa if overbought. If portfolio is already created, returns the same portfolio
        unless overwrite is True. Assumes total available cash is shared among all assets.

        Args:
            init_cash (float): Initial cash to be used for the portfolio. Defaults to 100000.
            overwrite (bool): If True, overwrite the existing portfolio even if it exists. Defaults to False.

        Returns:
            Portfolio: The portfolio object after applying the MR strategy.
        """

        if self.portfolio is None or overwrite:
            [entries, exits] = self._MRStrategy()
            self.portfolio = Portfolio.from_signals(
                self.price_data,
                entries,
                exits, 
                init_cash=init_cash,
                cash_sharing=True
            )
        return self.portfolio