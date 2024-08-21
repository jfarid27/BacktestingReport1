import numpy as np
from Backtest.models.EvolutionaryModel import blend_signals, generate_weights

def test_blend_signals():
    entries = [
        np.array([True, False, True, False]),
        np.array([False, True, False, True]),
        np.array([True, True, False, False])
    ]
    exits = [
        np.array([False, True, False, True]),
        np.array([True, False, True, False]),
        np.array([False, False, True, True])
    ]
    weights = [0.33, 0.33, 0.33]
    entry_threshold = 0.5
    exit_threshold = 0.5

    blended_entries, blended_exits = blend_signals(entries, exits, weights, entry_threshold, exit_threshold)

    expected_entries = np.array([True, True, False, False])
    expected_exits = np.array([False, False, True, True])

    np.testing.assert_array_equal(blended_entries, expected_entries)
    np.testing.assert_array_equal(blended_exits, expected_exits)

def test_generate_weights():
    mutation_rate = 0.01

    # Test with initial weights provided
    initial_weights = np.array([0.2, 0.3, 0.5])
    mutated_weights = generate_weights(initial_weights, mutation_rate)

    assert np.allclose(np.sum(mutated_weights), 1.0)  # Check if weights sum to 1
    assert np.all(mutated_weights >= 0)  # Check if weights are non-negative

    # Test without initial weights
    mutated_weights = generate_weights(mutation_rate=mutation_rate)

    assert np.allclose(np.sum(mutated_weights), 1.0)  # Check if weights sum to 1
    assert np.all(mutated_weights >= 0)  # Check if weights are non-negative