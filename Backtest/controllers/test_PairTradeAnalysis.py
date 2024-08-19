import numpy as np
import pandas as pd

from Backtest.controllers.PairTradeAnalysis import divergence_indicator, generate_pairs

def test_divergence_indicator():
    price_data = [(100, 105), (102, 107), (101, 106), (103, 108)]
    entries, exits = divergence_indicator(price_data, level=2)
    expected_entries = np.array([False, False, False])
    expected_exits = np.array([False, False, False])
    np.testing.assert_array_equal(entries, expected_entries)
    np.testing.assert_array_equal(exits, expected_exits)

def test_generate_pairs():
    price_data = pd.DataFrame({
        'AAPL': [100, 105, 110, 115, 120],
        'MSFT': [200, 210, 220, 230, 240],
        'GOOG': [-300, -290, -280, -270, -260],
        'BTC': [101, 102, 101, 102, 101]
    })

    pairs = generate_pairs(price_data)

    expected_pairs = [('AAPL', 'MSFT'), ('AAPL', 'GOOG'), ('MSFT', 'GOOG')]
    assert pairs == expected_pairs