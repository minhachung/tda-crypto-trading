# System Architecture: TDA-Based Crypto Trading

## High-Level System Design

```
DATA INGESTION → PREPROCESSING → TDA COMPUTATION → SIGNAL GENERATION → EXECUTION
     ↓                ↓                ↓                 ↓                ↓
  APIs        Point Clouds    Persistence        Trading Rules      Orders/Alerts
  Blockchain  Normalization    Diagrams          Backtesting        Dashboard
```

---

## Module 1: Data Ingestion & Preprocessing

### Input Data Sources

#### Strategy 1: Price-Volume Manifolds
```
OHLCV Data:
  - Open, High, Low, Close, Volume (5-min, hourly, daily)
  - Source: Binance API, Kraken API, CoinGecko

Technical Indicators:
  - RSI (14-period momentum)
  - MACD (trend)
  - Bollinger Bands (volatility)
  - Source: Computed from OHLCV

On-Chain Metrics:
  - Active addresses
  - Transaction count
  - Exchange inflows/outflows
  - Source: Glassnode API, IntoTheBlock

Combined Features: 5 OHLCV + 3 indicators + 3 on-chain = 11 dimensions
```

#### Strategy 2: Exchange Flow Networks
```
Transaction Data:
  - Exchange ID, volume, direction (buy/sell/transfer)
  - Timestamp, price
  - Counterparty exchange
  - Source: Blockchain.com, Etherscan, Exchange APIs

Network Structure:
  - Nodes: Exchange identifiers
  - Edges: Transaction flows (volume, frequency)
  - Attributes: Velocity, direction, timing
```

### Preprocessing Pipeline

```python
Data Flow:
1. Fetch raw data (APIs, blockchain)
   ↓
2. Data validation (null checks, outliers)
   ↓
3. Normalization (z-score per feature)
   ↓
4. Feature engineering (technical indicators)
   ↓
5. Point cloud construction (sliding windows)
   ↓
6. Store in data/processed/
```

**Key Parameters:**
- **Window size:** 5-30 days (Takens embedding)
- **Stride:** 1 day (sliding window)
- **Normalization:** z-score (zero mean, unit variance)

---

## Module 2: Persistent Homology Computation

### Rips Complex Construction

```
Input: n points in d-dimensional space (point cloud)
     ↓
Epsilon filtration:
  - ε = 0: n disconnected points (H₀ = n components)
  - ε = δ₁: first edges form (some H₀ merge)
  - ε = δ₂: first triangles form (first H₁ loop appears)
  - ...
  - ε = ∞: fully connected (H₀ = 1, H₁ = 0)
     ↓
Persistent homology:
  - Track birth/death of features
  - Connected components: H₀ (dimension 0)
  - Loops: H₁ (dimension 1)
  - Voids: H₂ (dimension 2) [often empty]
     ↓
Output: Persistence diagram (birth, death pairs)
        or Barcode (interval representation)
```

### Key TDA Features Extracted

#### 1. Persistence Diagram
```
Diagram = {(b₁, d₁), (b₂, d₂), ..., (bₖ, dₖ)}
where:
  - bᵢ = birth time (filtration value where feature appears)
  - dᵢ = death time (filtration value where feature disappears)
  - persistence = dᵢ - bᵢ (how long feature survives)

Interpretation for trading:
  - Large persistence = robust structure (signal)
  - Small persistence = noise
  - Diagram changes = regime change
```

#### 2. Persistence Landscape
```
Convert diagram to functional data:

landscape(t) = max(0, min(t - bᵢ, dᵢ - t))

Results in:
  - Continuous function (smoother than diagram)
  - Can be treated as time series
  - Comparable across windows

Key metrics:
  - L₁-norm: ∫ landscape(t) dt (total "area")
  - C₁-norm: max(landscape(t)) (peak height)
  - Entropy: -∫ p(t) log p(t) dt (distribution complexity)
```

#### 3. Barcode Representation
```
Visual: Each feature as horizontal line (birth → death)
  ────────────── Feature 1 (long persistence)
  ───── Feature 2 (short persistence)
  ──────────────────────── Feature 3 (very long)

Metrics:
  - Number of bars (total features)
  - Number of "long" bars (>threshold)
  - Clustering of bars (separate or mixed?)
```

### Implementation Details

```python
# Pseudocode for persistent homology
def compute_persistent_homology(point_cloud):
    """
    Input: n points in d dimensions
    Output: Persistence diagram
    """
    # Build Rips complex (all distances)
    distances = pairwise_distances(point_cloud)
    
    # Add edges in order of distance
    edges = []
    for i, j in all_pairs:
        edges.append((distances[i,j], i, j))
    edges.sort()
    
    # Union-find for connected components (H₀)
    uf = UnionFind(n)
    for dist, i, j in edges:
        if not uf.connected(i, j):
            uf.union(i, j)
            # Record H₀ birth
        else:
            # Merging creates H₁ loop (records death)
    
    return persistence_pairs
```

**Libraries:**
- `giotto-tda`: Full TDA suite
- `ripser`: Fast C++ implementation
- `persim`: Persistence metrics

---

## Module 3: Signal Generation

### Strategy 1: Price-Volume Manifold Signals

```
Input: Time series of price data
     ↓
Sliding window:
  - Window size: W days (e.g., 10 days)
  - Stride: 1 day
  - Extract 11-dimensional feature vector each day
     ↓
Point cloud: {P₁, P₂, ..., Pₙ}
  - Each point = time window
  - n = length of price series - W
     ↓
Persistent homology:
  - Compute diagram
  - Extract C₁-norm(t), entropy(t), count(t)
     ↓
Signal generation:
  - BUY signal: When C₁-norm peaks
  - SELL signal: When C₁-norm drops sharply OR
                 When persistence diagram changes significantly
  - HOLD: Stable persistence
     ↓
Risk adjustment:
  - Confidence = entropy level
  - Position size ∝ C₁-norm magnitude
```

### Strategy 2: Exchange Flow Network Signals

```
Input: Transaction flows between exchanges
     ↓
Network construction:
  - Nodes = {Exchange A, Exchange B, Exchange C, ...}
  - Edges = {(A,B,vol), (B,C,vol), ...}
  - Attributes: volume, frequency, direction
     ↓
Mapper algorithm:
  - Lens function: Exchange total volume, velocity
  - Cover: Overlapping bins
  - Clustering: k-means on lens values
     ↓
Simplified network:
  - Nodes = clusters of similar activity
  - Edges = cluster interactions
     ↓
Anomaly detection:
  - ALERT: New bottleneck structure appears
  - ALERT: Sudden community formation
  - ALERT: Unusual ring-trading pattern (cycle detection)
  - RISK_SCORE: Based on network persistence
     ↓
Whale detection:
  - Track large single transfers
  - Correlation with price movements
```

---

## Module 4: Backtesting Framework

### Time Series Cross-Validation

```
Data: [————————————————————————————————————————————————]
       Train    Val   Test Train    Val   Test

Split 1: [─────][───][────────────]
Split 2:            [─────][───][────────────]
Split 3:                   [─────][───][────────────]

Prevents: Look-ahead bias, overfitting on recent data
```

### Backtest Metrics

```python
Metrics to track:

Performance:
  - Sharpe ratio = (ret - risk_free) / std(ret)
  - Win rate = % trades profitable
  - Profit factor = sum(wins) / sum(losses)
  - CAGR = compound annual growth

Risk:
  - Max drawdown = largest peak-to-trough
  - Calmar ratio = CAGR / max_drawdown
  - Sortino ratio = CAGR / downside_std

Detection quality:
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1 = 2 * precision * recall / (precision + recall)

Statistical:
  - n_trades
  - avg_trade_duration
  - avg_win / avg_loss
```

---

## Module 5: Real-Time Execution Layer

### Streaming Architecture

```
Live data:
  [Price feed] → [Buffer] → [Preprocessing] → [TDA] → [Signal]
                                              ↓
                                         Decision → [Risk Manager]
                                                        ↓
                                                   Order Execution
```

**Latency Requirements:**
- Data fetch: <100ms
- TDA computation: <500ms (rolling window)
- Signal generation: <100ms
- Total: ~700ms acceptable

**Optimization:**
- Use incremental/streaming TDA (giotto-learn)
- Cache previous computations
- Batch updates (every hour, not every tick)

---

## Data Flow Diagrams

### Strategy 1: Price Manifolds

```
API Fetch (Binance, Glassnode)
    ↓
Normalize OHLCV + Indicators
    ↓
Create 10-day rolling point clouds
    ↓
Compute persistent homology (Rips)
    ↓
Extract C₁-norm, entropy, count
    ↓
Generate trading signals
    ↓
Backtest against historical data
    ↓
Execute on live market (if profitable)
```

### Strategy 2: Exchange Networks

```
Scrape transaction data (Etherscan, Blockchain.com)
    ↓
Identify exchange addresses
    ↓
Build directed transaction graph
    ↓
Apply Mapper algorithm (lens + clustering)
    ↓
Analyze topology for anomalies
    ↓
Detect ring-trading, whale movements
    ↓
Generate risk scores & alerts
    ↓
Validate against known manipulation cases
```

---

## Configuration & Parameters

### Strategy 1: Price Manifolds

```yaml
data:
  source: binance  # or kraken, coingecko
  interval: 1h     # 5m, 15m, 1h, 1d
  lookback: 365    # days
  
preprocessing:
  window_size: 10        # days (Takens embedding)
  stride: 1              # day
  normalization: zscore  # or minmax
  
tda:
  complex_type: rips
  max_edge_length: 100   # controls filtration
  
signals:
  c1_threshold: 2.0      # std deviations above mean
  entropy_weight: 0.5    # confidence adjustment
  position_size: 0.1     # % of portfolio
```

### Strategy 2: Exchange Networks

```yaml
data:
  source: etherscan    # or blockchain.com
  interval: hourly     # or daily
  lookback: 90         # days
  
mapper:
  lens_dimension: 2      # 1 or 2
  overlap: 0.3           # 30% overlap between bins
  n_clusters: 3          # per bin
  
anomaly:
  ring_transaction_depth: 3    # cycles to track
  bottleneck_threshold: 0.8    # importance score
  community_min_size: 2        # exchanges
```

---

## Deployment Options

### Development
```
Local machine
  ↓
Jupyter notebooks
  ↓
Backtesting only (no live trading)
```

### Staging
```
Cloud VM (AWS, GCP)
  ↓
Scheduled Python scripts
  ↓
Backtest + paper trading
  ↓
Email alerts only
```

### Production
```
Kubernetes cluster
  ↓
Streaming data pipeline (Kafka)
  ↓
Real-time TDA computation
  ↓
Automated execution (with kill-switches)
  ↓
Monitoring & logging
```

---

## Risk Management

### Hard Stops

```
1. Max position size: 10% of portfolio per signal
2. Daily loss limit: Stop trading if -5% down
3. Max leverage: 2x (or 1x for first version)
4. Cooldown period: No trades for 1 hour after execution
```

### Soft Stops

```
1. Confidence threshold: Only trade if confidence > 70%
2. Volatility filter: Reduce size if IV > 80th percentile
3. Liquidity check: Min order book depth before execution
4. Exchange health: Skip if API latency > 1sec
```

### Monitoring

```
Real-time alerts on:
  - Unusual persistence diagram changes
  - Failed API connections
  - Execution errors
  - Drawdown > threshold
  - Win rate < historical baseline
```

---

## Testing Strategy

### Unit Tests
```
test_data_pipeline.py
  - Fetch validation
  - Normalization correctness
  - Feature engineering
  
test_persistent_homology.py
  - Diagram computation
  - Landscape extraction
  - Feature metrics
  
test_signals.py
  - Signal generation logic
  - Threshold behavior
  - Edge cases
```

### Integration Tests
```
test_end_to_end.py
  - Full pipeline: data → signal → backtest
  - Multiple assets
  - Different time periods
```

### Validation Tests
```
backtest_validation.py
  - Forward testing (unseen data)
  - Walk-forward optimization
  - Parameter sensitivity
  - Statistical significance
```

---

## Monitoring & Metrics

### System Health
- API uptime
- Data quality (completeness, latency)
- Computation time per update
- Memory usage

### Trading Performance
- Daily P&L
- Sharpe ratio (rolling)
- Win rate (rolling)
- Max drawdown (rolling)

### TDA Quality
- Persistence diagram stability
- Feature count trends
- Entropy trends
- Anomaly score distribution
