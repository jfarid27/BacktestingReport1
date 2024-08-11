import numpy as np
from PairTradeAnalysis import divergence_indicator

def test_divergence_indicator():
    price_data = [(10, 20), (15, 25), (12, 18), (8, 16), (20, 40)]
    level = 2

    entries = divergence_indicator(price_data, level)

    expected_entries = np.array([False, False, True, False])
    assert np.array_equal(entries, expected_entries)

test_divergence_indicator()