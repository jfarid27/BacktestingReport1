from typing import Any, Optional, List
from vectorbt.portfolio import Portfolio
import numpy as np
import copy

def blend_signals(entries, exits, weights, entry_threshold=0.5, exit_threshold=0.5):
    """Take list of vectorbt entries and exits, and blend them into a single signal.
    
    New signal is just a weighted average of the entries and exits. Tunable with threshold
    where new signal is True if the weighted average is above threshold, and vice versa
    for the exit.
    
    Args:
        entries (list of np.ndarray): List of entry signals (boolean arrays).
        exits (list of np.ndarray): List of exit signals (boolean arrays).
        weights (list of float): List of weights for each signal.
        entry_threshold (float): Threshold for the blended entry signal. Defaults to 0.5.
        exit_threshold (float): Threshold for the blended exit signal. Defaults to 0.5.

    Returns:
        tuple: Blended entry and exit signals as numpy arrays.
    """

    weights = np.array(weights)
    
    # Calculate weighted average for entries and exits
    weighted_entries = np.average(entries, axis=0, weights=weights)
    weighted_exits = np.average(exits, axis=0, weights=weights)
    
    # Generate blended signals
    blended_entries = weighted_entries > entry_threshold
    blended_exits = weighted_exits > exit_threshold
    
    return blended_entries, blended_exits

def generate_weights(weights=None, mutation_rate=0.01):
    """Mutates weight array vector by adding random noise to weight array elements and normalizing.
    
    Args:
        weights (np.ndarray or None): Initial weights vector. If None, a random vector is generated.
        mutation_rate (float): Rate at which weights are mutated. Defaults to 0.1.
    
    Returns:
        np.ndarray: Mutated and normalized weights vector.
    """
    if weights is None:
        weights = np.random.rand(10) 
        weights /= np.sum(weights)

    # Mutate weights by adding random noise
    noise = np.random.normal(0, mutation_rate, size=weights.shape)
    mutated_weights = weights + noise

    # Ensure weights are non-negative
    mutated_weights = np.clip(mutated_weights, 0, None)

    # Normalize weights to sum to 1
    normalized_weights = mutated_weights / np.sum(mutated_weights)

    return normalized_weights

class EvolutionaryPortfolio:
    """A vectorbt portfolio object that stores information and methods for evolving the portfolio.
    Assumes the entries and exits are standard signal strategies with weights and appropriate indecies.
    """

    portfolio: Optional[Any] = None
    data: Any
    weights: List[float]
    entries: List[Any]
    exits: List[Any]
    weighted_entries: List[Any]
    weighted_exits: List[Any]
    init_cash: float
    
    def __init__(self, data, weights, entries, exits, init_cash=100000):
        self.data = data
        self.weights = weights
        self.entries = entries
        self.exits = exits
        self.init_cash = init_cash
        weighted_entries, weighted_exits = blend_signals(entries, exits, weights)
        self.weighted_entries = weighted_entries
        self.weighted_exits = weighted_exits

    def evolve_portfolio(self, mutation_rate=0.01):
        """Evolve the portfolio by mutating the weights."""
        new_weights = generate_weights(self.weights, mutation_rate)
        weighted_entries, weighted_exits = blend_signals(self.entries, self.exits, self.weights)
        portfolio = Portfolio.from_signals(
            self.data,
            entries=self.weighted_entries,
            exits=self.weighted_exits, 
            init_cash=self.init_cash,
            cash_sharing=True
        )

        if (portfolio.sharpe_ratio() > self.portfolio.sharpe_ratio()):
            self.weights = new_weights
            self.weighted_entries = weighted_entries
            self.weighted_exits = weighted_exits
            self.portfolio = portfolio
    
    def clone(self):
        """Create a deep copy of the current portfolio."""
        return copy.deepcopy(self)

class EvolutionaryPortfolioFamily:
    """Store a family of portfolios and methods related to evolving the family."""
    initial_weights: List[float]
    data: Any
    entries: List[Any]
    exits: List[Any]
    evolutionary_portfolios: Optional[List[EvolutionaryPortfolio]] = None

    def __init__(self, data, portfolio, weights, entries, exits, num_portfolios=10):
        self.data = data
        self.initial_weights = weights
        self.entries = entries
        self.exits = exits
        self.evolutionary_portfolios = [EvolutionaryPortfolio(data, weights, entries, exits) for _ in range(num_portfolios)]

    def evolve_family(self, mutation_rate=0.01):
        """Evolve the family of portfolios."""
        for portfolio in self.evolutionary_portfolios:
            portfolio.evolve_portfolio(mutation_rate)

    def optimize_genes(self):
        """Evolve the portfolio by selecting the current portfolio with highest fitness, and cloning it across the family.
        
        The current portfolio is selected based on the sharpe ratio of the portfolio.
        """
        sharpe_ratios = [portfolio.calculate_sharpe_ratio() for portfolio in self.evolutionary_portfolios]

        best_index = np.argmax(sharpe_ratios)

        best_portfolio = self.evolutionary_portfolios[best_index]

        # Clone the best portfolio across the family
        self.evolutionary_portfolios = [best_portfolio.clone() for _ in range(len(self.evolutionary_portfolios))]

    
    def run_simulation(self, n_steps=20, generation_size=10):
        """Run a simulation of the family of portfolios."""
        step = 0
        generation_counter = 0

        while step < n_steps:
            self.evolve_family()
            step += 1
            generation_counter += 1
            if generation_counter == generation_size:
                self.optimize_genes()
                generation_counter = 0