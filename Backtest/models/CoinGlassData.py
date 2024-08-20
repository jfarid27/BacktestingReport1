import requests
import os
import pandas as pd
from Backtest.models.LocalDataStorage import LocalDataStore
from dotenv import load_dotenv
from typing import Optional, Any

load_dotenv()
api_key = os.getenv('COINGLASS_API_KEY')

        
class CoinGlassOI(LocalDataStore):
    """Class for fetching and storing CoinGlass Open Interest data."""

    coin: str

    def __init__(self, file_path: str, coin: str):
        super().__init__(file_path)
        self.coin = coin

    def process_response(self, json_data: dict) -> pd.DataFrame:
        """Process the JSON response from the CoinGlass API."""
        if json_data["data"]:
            data = pd.DataFrame(json_data["data"])
            data["t"] = pd.to_datetime(data["t"], unit="ms")
            data.set_index(data["t"], inplace=True)
            data = data[["t", "h"]]
            return data

    def fetch(self, days=1000, debug=False) -> pd.DataFrame:
        """
        Fetches historical open interest data for a specific coin.

        Args:
            days (int, optional): Number of days of historical data to fetch. Defaults to 1000.
            debug (bool, optional): If True, fetches a limited number of data for debugging purposes. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the fetched historical data.
        """

        url = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-aggregated-history"

        params = {
            "symbol": self.coin,
            "interval": "1d",
            "limit": 10 if debug else days,
        }

        with requests.Session() as session:
            response = session.get(url, params=params, headers={"CG-API-KEY": api_key})
            if response.status_code == 200:
                if debug:
                    print(response.json())
                df = self.process_response(response.json())
                return df