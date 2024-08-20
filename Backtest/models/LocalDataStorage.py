import os
import pandas as pd
from typing import Optional, Any

class LocalDataStore():
    """
    Takes a named file path and store data in it. Assumes data is a pandas DataFrame with index as the first column.
    """

    file_path: str
    data: Optional[pd.DataFrame] = None

    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch(self) -> pd.DataFrame:
        """Child classes should implement this method to fetch data from a source."""
        pass

    def load(self, *args, **kwargs) -> pd.DataFrame:
        """
        Load the data from a file or fetch it if the file does not exist.
        Parameters:
            debug (bool): Flag indicating whether to enable debug mode. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the fetch method.
        Returns:
            pd.DataFrame: The loaded or fetched data as a pandas DataFrame.
        """
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        else:
            df = self.fetch(*args, **kwargs)
            df.to_csv(self.file_path)
        
        self.data = df
        return df