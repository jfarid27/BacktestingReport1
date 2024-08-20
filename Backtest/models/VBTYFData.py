import pandas as pd
from Backtest.models.LocalDataStorage import LocalDataStore
import vectorbt as vbt

        
class VBTYFData(LocalDataStore):
    """LocalDataStore wrapper for fetching and storing VectorBT Yahoo Finance data."""

    file_path: str

    def __init__(self, file_path: str):
        super().__init__(file_path)


    def fetch(self, ticker, debug=False, **kwargs) -> pd.DataFrame:
        """Fetch close data using the yahoo finance API for ticker."""
        df = vbt.YFData.download(ticker, **kwargs).get("Close")
        return df