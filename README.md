# Backtester

## Requirements

It is assumed you are in an environment with python, and poetry as a dependency manager. It's recommended to
run with miniconda environments.

- Python3

### Install Poetry with bash

```bash
    curl -sSL https://install.python-poetry.org | python3 -
```

### Install Poetry with pip

```bash
    pip install poetry
```

## Installation

After poetry is installed, just run `poetry install`.

## Testing

Testing is done with pytest. Pytest should have been installed as a dep, and tests can be run with `poetry run pytest`.

## Analysis Code

A sample analysis report can be found in the `BacktestsReport.ipynb`. Inside of Backtests.models is a set
of strategy abstractions pre-set using `vectorbt`. One can use jupyter notebooks to view and run backtests,
and start their own strategy development. To run jupyter notebooks, run `poetry run jupyter lab`.

