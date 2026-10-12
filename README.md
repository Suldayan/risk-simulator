# Risk Simulator

A Monte Carlo portfolio risk simulator that uses Geometric Brownian Motion (GBM) to model correlated asset price paths, estimate portfolio risk, and optionally accelerate simulations with GPU computing.

## Quick Start

```bash
pip install -e ./app
jupyter notebook
```

Open `notebooks/starter.ipynb` to get started.

## What It Does

- Fetches historical stock price data from Yahoo Finance.
- Aligns trading dates across multiple assets.
- Estimates drift (\(\mu\)), volatility (\(\sigma\)), and asset correlations from historical returns.
- Simulates correlated asset price paths using GBM and Cholesky decomposition.
- Computes portfolio risk metrics across simulated paths.
- Supports CPU execution with NumPy and optional GPU acceleration with CuPy.

## Example

```python
from gpu_risk_simulator.simulation.pipeline import simulate
from gpu_risk_simulator.metrics.risk import compute_risk_metrics
import numpy as np

# Run simulation
result = simulate(
    tickers=["AAPL", "MSFT"],
    T=1.0,
    n_steps=252,
    n_paths=10_000,
    engine="host",
)

# Combine simulated asset prices using portfolio weights
portfolio = np.average(result.paths, axis=0, weights=[0.6, 0.4])

# Calculate risk metrics
metrics = compute_risk_metrics(portfolio, x0=10_000)

print(f"95% VaR: ${metrics.var:,.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

## Architecture

| Module | Responsibility |
|---|---|
| `market_data/` | Yahoo Finance data ingestion, caching, and retry handling |
| `portfolio/fetcher.py` | Concurrent multi-ticker fetching and trading-date alignment |
| `brownian/models.py` | GBM parameter estimation, including drift, volatility, and correlation |
| `brownian/execution/` | Hardware abstraction for CPU (NumPy) and GPU (CuPy) execution |
| `brownian/gbm.py` | Unified single-asset and multi-asset GBM simulator |
| `metrics/risk.py` | Value at Risk (VaR), Conditional Value at Risk (CVaR), probability of loss, and maximum drawdown |
| `simulation/pipeline.py` | High-level simulation orchestration through `simulate()` |

## Hardware Acceleration

The simulator supports CPU and GPU execution through a configurable engine.

| Engine | Behavior | Backend |
|---|---|---|
| `host` | Runs on the CPU | NumPy |
| `device` | Explicitly requests GPU execution | CuPy (requires CUDA) |
| `auto` | Selects an available backend based on workload size | CuPy or NumPy |

In automatic mode, workloads with at least 1,000 paths are eligible for GPU execution when a compatible GPU backend is available. Smaller workloads use the CPU.

## Testing

Run the test suite with:

```bash
pytest -v
```

The project includes 43 tests covering unit tests, integration tests, and mocked external API calls.

## Requirements

- Python 3.11+
- NumPy
- Pandas
- yfinance
- Optional: CuPy and a compatible CUDA environment for GPU acceleration

## Future Directions

- **Stochastic volatility:** Extend beyond constant-volatility GBM with models such as Heston.
- **Jump-diffusion models:** Explore Merton or Bates models to better capture discontinuous price movements and tail risk.
- **FastAPI integration:** Expose `simulate()` through a REST API for dashboard integration.
- **Docker deployment:** Containerize the simulator for reproducible environments and cloud deployment.

---
