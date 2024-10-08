from Backtest.models.CoinGlassData import CoinGlassOI, CoinGlassFearGreedIndex
from pathlib import Path
from scipy import stats
import vectorbt as vbt

current_dir = Path.cwd()

oi_data_file = current_dir
fg_data_file = current_dir / "data/coin_glass_fg.csv"

def fetch_oi(coin: str, **kwargs):
    """
    Fetches the open interest data for a specific coin.

    Args:
        coin (str): The name of the coin.

    Returns:
        CoinGlassOI: The open interest data for the specified coin.
    """
    coin_file = oi_data_file / "data/coin_glass_{}_oi.csv".format(coin)
    data = CoinGlassOI(file_path=coin_file, coin=coin)
    data.load(**kwargs)
    return data

def fetch_fear_greed_index(**kwargs):
    """
    Fetches the fear and greed index data.

    Returns:
        CoinGlassFearGreedIndex: The fear and greed index data.
    """
    data = CoinGlassFearGreedIndex(file_path=fg_data_file)
    data.load(**kwargs)
    return data