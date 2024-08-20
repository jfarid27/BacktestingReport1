from Backtest.models.CoinGlassData import CoinGlassOI
from pathlib import Path
from scipy import stats
import vectorbt as vbt

current_dir = Path.cwd()

oi_data_file = current_dir / "data/coin_glass_oi_{}.csv"

def fetch_oi(coin: str, **kwargs):
    """
    Fetches the open interest data for a specific coin.

    Args:
        coin (str): The name of the coin.

    Returns:
        CoinGlassOI: The open interest data for the specified coin.
    """
    data = CoinGlassOI(file_path=oi_data_file.format(coin), coin=coin)
    data.load(**kwargs)
    return data