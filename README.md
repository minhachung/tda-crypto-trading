# TDA-Based Crypto Trading System

Research-driven crypto trading system using Topological Data Analysis (TDA) to detect market regimes, price reversal structures, and exchange manipulation patterns.

**Author:** Minha Chung · **Status:** Research / paper-trading

## Quick Start

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run full pipeline (CoinGecko, daily data)
python main.py BTC 365

# Run rigorous v2 validation (Coinbase hourly data, 5-fold CV, grid search)
python examples/run_validation_v2.py BTC 180 1h

# Strategy 2 (Mapper exchange manipulation detection)
python examples/run_strategy2.py
```

## Validation: Two Tiers

| Tier | Script | Data | Method | Use case |
|------|--------|------|--------|----------|
| v1 (basic) | `examples/run_validation.py` | CoinGecko daily | Train/val/test split | Quick sanity check |
| **v2 (rigorous)** | `examples/run_validation_v2.py` | **Coinbase hourly** | **5-fold CV + grid search** | Statistical inference |

The v2 framework uses 24× more samples (hourly vs daily), grid-searches over 3 hyperparameters, runs 5-fold time-series CV, and reports Wilson confidence intervals + bootstrap Sharpe + t-tests. See `VALIDATION_REPORT_V2.md` after running.


## Project Overview

This project implements two complementary TDA-based trading strategies synthesized from 6 research papers on blockchain analytics and financial time series analysis:

### Strategy 1: Persistent Homology of Price-Volume Manifolds ⭐ (Best Overall)

**Goal:** Detect market regime changes through topological structure

**Data Architecture:**
- OHLCV (Open, High, Low, Close, Volume)
- Technical indicators: RSI, MACD, Bollinger Bands
- On-chain metrics: active addresses, transaction count, exchange flows

**Method:**
- Build point clouds where each point = time window with multi-dimensional coordinates
- Apply persistent homology to detect topological features (connected components, loops, voids)
- Track persistence diagrams to identify when 1-cycles (loops) die = regime changes

**Trading Signal:**
- Buy/Sell triggers when persistence landscapes peak or topology changes
- Validated against Bitcoin/Ethereum historical data

**Expected Output:**
- Trading signal system with probability estimates
- Performance metrics vs. traditional indicators

---

### Strategy 2: Mapper Algorithm for Exchange Flow Networks ⭐⭐ (Highest Impact)

**Goal:** Detect wash trading, price manipulation, whale movements

**Data Architecture:**
- Transaction data across multiple exchanges (volume, velocity, direction)
- Exchange-to-exchange flows
- Time-series transaction patterns

**Method:**
- Build Mapper graph (simplified topological network):
  - Nodes = clusters of similar exchange activity
  - Edges = exchanges that frequently transact together
- Analyze network topology for:
  - Bottleneck structures (critical exchanges)
  - Community detection (coordinated trading)
  - Anomalies (suspicious ring-trading patterns)

**Detection System:**
- Flag suspicious exchange activity in real-time
- Explain network topology changes
- Track whale movements and coordination

**Expected Output:**
- Anomaly detection system
- Explainable network analysis
- Risk scoring for suspicious patterns

---

## Key Concepts from Research Papers

### Persistent Homology Basics
- Track topological features (connected components H₀, loops H₁, voids H₂) across multi-scale filtrations
- Persistence diagrams show feature "births" and "deaths"
- C1-norm of persistence landscapes peaks before crashes (Gidea et al. 2020)

### TDA Features for Forecasting
- Entropy of persistence diagrams
- Amplitude of persistent features  
- Number of significant points in persistence diagram
- Boosts forecasting accuracy when combined with N-BEATS (Jesus Jr. et al. 2025)

### Ethereum Transaction Network Analysis
- Local topology predicts token price movements
- Functional data depth captures network patterns
- Real-time transaction graph available (unlike traditional finance)

### Ransomware Detection Baseline
- TDA successfully identifies malicious address clusters in Bitcoin
- Topology-based clustering outperforms heuristics
- Transferable to exchange manipulation detection

---

## Project Structure

```
tda-crypto-trading/
├── README.md                          # This file
├── docs/
│   ├── 01_literature_summary.md      # Synthesis of 6 research papers
│   ├── 02_architecture.md            # System design
│   ├── 03_tda_primer.md              # TDA concepts & math
│   └── 04_implementation_guide.md    # Step-by-step technical guide
├── notebooks/
│   ├── 01_data_pipeline.ipynb        # Fetch & preprocess price/on-chain data
│   ├── 02_persistent_homology.ipynb  # Compute persistence diagrams
│   ├── 03_trading_signals.ipynb      # Generate buy/sell signals
│   ├── 04_mapper_graphs.ipynb        # Build exchange flow networks
│   └── 05_backtesting.ipynb          # Performance evaluation
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py              # Data fetching & preprocessing
│   ├── persistent_homology.py        # TDA persistence computations
│   ├── mapper_algorithm.py           # Mapper graph construction
│   ├── trading_signals.py            # Signal generation logic
│   ├── exchange_anomaly.py           # Manipulation detection
│   └── backtester.py                 # Performance evaluation
├── data/
│   ├── raw/                          # Raw price/on-chain data
│   ├── processed/                    # Cleaned point clouds
│   └── persistence/                  # Computed diagrams & landscapes
├── models/
│   ├── persistence_models/           # Trained persistence-based models
│   └── mapper_graphs/                # Computed Mapper networks
└── tests/
    ├── test_persistence_homology.py
    ├── test_mapper_algorithm.py
    └── test_trading_signals.py
```

---

## Data Sources

### Price & OHLCV Data
- CoinGecko API (free, historical)
- Binance API (real-time)
- Kraken API (multiple exchange comparison)

### On-Chain Metrics
- Glassnode API (active addresses, transaction count, exchange flows)
- IntoTheBlock (whale movements)
- Nansen (smart contract interactions for Ethereum)

### Exchange Flow Data
- Blockchain.com (transaction graph)
- Etherscan (Ethereum transactions)
- Exchange-specific APIs (Binance, Kraken, Coinbase)

---

## Technology Stack

- **TDA Computation:** 
  - `giotto-tda` (Persistent homology, Mapper)
  - `ripser` (Fast persistent homology)
  - `persim` (Persistence metric computations)

- **Data Processing:**
  - `pandas`, `numpy`
  - `ta` (Technical indicators)
  - `networkx` (Graph analysis)

- **ML/Forecasting:**
  - `scikit-learn` (K-means, anomaly detection)
  - `statsmodels` (Time series)
  - `tensorflow`/`pytorch` (N-BEATS for enhanced forecasting)

- **Backtesting:**
  - `backtesting.py` or custom backtester

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (CoinGecko, Glassnode, etc.)
   ```

3. **Run data pipeline:**
   ```bash
   python src/data_pipeline.py --ticker BTC --days 365
   ```

4. **Generate trading signals:**
   ```bash
   jupyter notebook notebooks/02_persistent_homology.ipynb
   ```

5. **Backtest strategy:**
   ```bash
   jupyter notebook notebooks/05_backtesting.ipynb
   ```

---

## Research Papers Included

1. **Topological Data Analysis for Portfolio Management of Cryptocurrencies** (2019)
   - Authors: Rivera-Castro, Pilyugina, Burnaev
   - Focus: Persistence landscapes for portfolio selection

2. **Ethereum Price Prediction using TDA** (2022)
   - Authors: Hafez, ElNainay, et al.
   - Focus: TDA features from blockchain interaction networks

3. **Topological Data Analysis for Identifying Critical Transitions** (2018)
   - Authors: Saengduean, Noisagool, Chamchod
   - Focus: Early warning signals before crashes

4. **Enhancing Financial Time Series Forecasting through TDA** (2025)
   - Authors: Jesus Jr., Fernández-Navarro, Carbonero-Ruz
   - Focus: TDA + N-BEATS for forecasting

5. **Dissecting Ethereum Blockchain Analytics** (2020)
   - Authors: Li, Islambekov, Akcora, et al.
   - Focus: Transaction network topology & price prediction

6. **BitcoinHeist: TDA for Ransomware Detection** (2019)
   - Authors: Akcora, Li, Gel, Kantarcioglu
   - Focus: TDA-based anomaly detection on blockchain

---

## Key Metrics

### Strategy 1: Persistent Homology
- Sharpe Ratio (risk-adjusted return)
- Win Rate (% profitable trades)
- Max Drawdown
- Detection Accuracy (regime changes predicted correctly)

### Strategy 2: Mapper Exchange Networks
- Precision/Recall (manipulative trades detected)
- False positive rate
- Network metrics (bottleneck stability, community size)

---

## Next Steps

1. Review literature synthesis (`docs/01_literature_summary.md`)
2. Understand TDA fundamentals (`docs/03_tda_primer.md`)
3. Follow implementation guide (`docs/04_implementation_guide.md`)
4. Run notebooks in order (01 → 05)
5. Backtest and optimize parameters

---

## License

Research project. See individual papers for citations.

## Contact

Based on synthesis of 6 academic papers on TDA for cryptocurrency analysis (2019-2025).
