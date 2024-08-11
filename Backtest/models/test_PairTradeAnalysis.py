import numpy as np
import pandas as pd

from PairTradeAnalysis import divergence_indicator, generate_pairs

def test_divergence_indicator():
    price_data = [(10, 20), (15, 25), (12, 18), (8, 16), (20, 40)]
    level = 2

    entries = divergence_indicator(price_data, level)

    expected_entries = np.array([False, False, True, False])
    assert np.array_equal(entries, expected_entries)

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