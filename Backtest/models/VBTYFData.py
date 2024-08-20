import os
import pandas as pd
from Backtest.models.LocalDataStorage import LocalDataStore
import vectorbt as vbt

        
class VBTYFData(LocalDataStore):
    """LocalDataStore wrapper for fetching and storing VectorBT Yahoo Finance data."""

    file_path: str

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self, *args, **kwargs) -> pd.DataFrame:
        """
        Load the data from a file or fetch it if the file does not exist.
        Parameters:
            **kwargs: Additional keyword arguments to be passed to the fetch method.
        Returns:
            pd.DataFrame: The loaded or fetched data as a pandas DataFrame.
        """
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
            df = df["Close"]
        else:
            df = self.fetch(*args, **kwargs)
            df.to_csv(self.file_path)
        
        self.data = df
        return df


    def fetch(self, ticker, debug=False, **kwargs) -> pd.DataFrame:
        """Fetch close data using the yahoo finance API for ticker."""
        df = vbt.YFData.download(ticker, **kwargs).get("Close")
        return df