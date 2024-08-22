from typing import Any, Optional, List
from vectorbt.portfolio import Portfolio
import numpy as np
from math import isfinite
import copy

def blend_signals(entries, exits, weights, entry_threshold=0.5, exit_threshold=0.5, debug=False):
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
    
    if debug:
        print(entries)
        print(exits)
        print(weights)
    weighted_entries = np.average(entries, axis=0, weights=weights)
    weighted_exits = np.average(exits, axis=0, weights=weights)
    
    blended_entries = weighted_entries > entry_threshold
    blended_exits = weighted_exits > exit_threshold
    
    return blended_entries, blended_exits

def generate_weights(weights=None, weight_length=0, mutation_rate=0.01, debug=False):
    """Mutates weight array vector by adding random noise to weight array elements and normalizing.
    
    Args:
        weights (np.ndarray or None): Initial weights vector. If None, a random vector is generated.
        mutation_rate (float): Rate at which weights are mutated. Defaults to 0.1.
    
    Returns:
        np.ndarray: Mutated and normalized weights vector.
    """
    if weights is None:
        weights = np.ones(weight_length)
        weights /= weight_length
        return weights

    noise = np.random.normal(0, mutation_rate, size=weights.shape)
    mutated_weights = weights + noise


    mutated_weights = np.clip(mutated_weights, 0.01, None)


    normalized_weights = mutated_weights / np.sum(mutated_weights)

    if debug:
        print("Normalized Weights: ", normalized_weights)

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
    entry_threshold: float
    exit_threshold: float
    init_cash: float
    
    def __init__(self, data, weights, entries, exits, entry_threshold=0.5, exit_threshold=0.5, init_cash=100000):
        self.data = data
        self.weights = weights
        self.entries = entries
        self.exits = exits
        self.init_cash = init_cash
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.weighted_entries, self.weighted_exits = blend_signals(
            entries, exits, weights,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )
        self.entries = entries
        self.exits = exits
        self.portfolio = Portfolio.from_signals(
            self.data,
            entries=self.weighted_entries,
            exits=self.weighted_exits, 
            init_cash=self.init_cash,
            cash_sharing=True
        )

    def evolve_portfolio(self, mutation_rate=0.01, debug=False):
        """
        Evolve the portfolio by mutating the weights.
        Parameters:
        - mutation_rate (float): The rate at which the weights are mutated. Default is 0.01.
        - debug (bool): If True, print the old and new Sharpe ratios. Default is False.
        Returns:
            None
        """
        new_weights = generate_weights(self.weights, mutation_rate)
        weighted_entries, weighted_exits = blend_signals(
            self.entries, self.exits, new_weights,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold
        )
        
        portfolio = Portfolio.from_signals(
            self.data,
            entries=weighted_entries,
            exits=weighted_exits, 
            init_cash=self.init_cash,
            cash_sharing=True
        )

        if debug:
            print("Old Sharpe Ratio: ", self.portfolio.sharpe_ratio())
            print("New Sharpe Ratio: ", portfolio.sharpe_ratio())

        no_prev_trade = not isfinite(self.portfolio.sharpe_ratio())
        new_is_finite = isfinite(portfolio.sharpe_ratio())
        old_is_finite = isfinite(self.portfolio.sharpe_ratio())

        if (no_prev_trade or 
            (new_is_finite and not old_is_finite) or
            (new_is_finite and
                portfolio.sharpe_ratio() >= self.portfolio.sharpe_ratio()
        )):
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

    def __init__(self, data, weights, entries, exits, num_portfolios=10, entry_threshold=0.5, exit_threshold=0.5,):
        self.data = data
        self.initial_weights = weights
        self.entries = entries
        self.exits = exits
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.evolutionary_portfolios = [
            EvolutionaryPortfolio(data, weights, entries, exits,
                                  entry_threshold=self.entry_threshold,
                                  exit_threshold=self.exit_threshold) 
            for _ in range(num_portfolios)
        ]

    def evolve_family(self, mutation_rate=0.01):
        """Evolve the family of portfolios."""
        for portfolio in self.evolutionary_portfolios:
            portfolio.evolve_portfolio(mutation_rate)

    def optimize_genes(self):
        """Clone the best portfolio currently in the family and replace all other portfolios with it."""

        best_portfolio = self.fetch_best_portfolio()
        self.evolutionary_portfolios = [best_portfolio.clone() for _ in range(len(self.evolutionary_portfolios))]

    def fetch_best_portfolio(self):
        """Fetch the portfolio with the highest sharpe ratio."""
        sharpe_ratios = [
            portfolio.portfolio.sharpe_ratio() if isfinite(portfolio.portfolio.sharpe_ratio()) else 0
            for portfolio in self.evolutionary_portfolios
        ]
        best_index = np.argmax(sharpe_ratios)
        return self.evolutionary_portfolios[best_index]
    
    def run_simulation(self, n_steps=20, generation_size=10, temperature=1, delta=1, results_log=None, debug=False):
        """
        Run a simulation of the family of portfolios.

        Parameters:
        - n_steps (int): Number of steps to run the simulation for. Default is 20.
        - generation_size (int): Size of each generation. Default is 10.
        - temperature (int): Initial temperature for mutation rate. Default is 1.
        - delta (int): Temperature decay factor. Default is 1.
        - results_log (file): File object to write simulation results to. Default is None.
        - debug (bool): Flag to enable debug mode. Default is False.
        """
        step = 0
        generation_counter = 0

        while step < n_steps:
            self.evolve_family(mutation_rate=temperature)
            step += 1
            generation_counter += 1
            if generation_counter == generation_size:
                temperature *= delta
                self.optimize_genes()
                generation_counter = 0
                if results_log:
                    best_portfolio = self.fetch_best_portfolio()
                    sharpe = best_portfolio.portfolio.sharpe_ratio()
                    weights = best_portfolio.weights
                    weights_str = ",".join(map(str, weights))
                    csv_row = f"{step},{sharpe},{weights_str}\n"
                    if debug: 
                        print(csv_row)
                    with results_log.open(mode='a') as file:
                        file.write(csv_row)
                        pass