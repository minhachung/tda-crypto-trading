# Literature Synthesis: TDA for Cryptocurrency Trading

## Overview
This document synthesizes 6 research papers (2019-2025) applying Topological Data Analysis to cryptocurrency markets.

---

## 1. Topological Data Analysis for Portfolio Management of Cryptocurrencies
**Rivera-Castro, Pilyugina, Burnaev (2019)**

### Key Contributions
- Applied TDA to 1561 cryptocurrencies across 6 years (2013-2019)
- Used persistence landscapes to identify investment opportunities
- Compared traditional portfolio methods vs. TDA-based selection

### Methodology
1. Create time-series point cloud from cryptocurrency price features
2. Compute persistent homology across multiple filtration levels
3. Extract persistence landscapes (functional data)
4. Use landscape properties to score and rank cryptocurrencies
5. Select portfolio based on topological stability

### Key Findings
- TDA outperformed traditional portfolio methods without feature engineering
- Persistence landscapes capture hidden structures in cryptocurrency correlations
- Method works with limited feature engineering (raw OHLCV)

### Applicable to Project
- **Strategy 1 foundation:** Uses persistence landscapes for signal generation
- Validates that TDA works without domain knowledge
- Provides proof-of-concept for 1500+ asset comparison

---

## 2. Ethereum Price Prediction using Topological Data Analysis
**Hafez, ElNainay, Abougabal, Kosba (2022)**

### Key Contributions
- Applied TDA to Ethereum network interaction patterns
- Extracted TDA features from 3 interaction types: transactions, smart contracts, tokens
- Achieved 0.75% MAPE (hourly), 4.9% MAPE (daily), 13.75% MAPE (weekly)
- Extended to predict 8 Ethereum token prices

### Methodology
1. Extract interaction networks from Ethereum blockchain:
   - User transactions (Ether transfers)
   - Smart contract interactions
   - Token transfers (ERC-20 events)
2. Compute persistence diagrams for each network
3. Extract TDA features:
   - Number of significant features
   - Entropy
   - Amplitude
4. Feed TDA features + traditional indicators to forecasting model
5. Compare against baseline (no TDA features)

### Key Findings
- TDA features significantly improve price prediction accuracy
- Different network types have different predictive power
- Classification of steep price movements improved with TDA
- Token predictions work with modified method

### Applicable to Project
- **Strategy 1:** Framework for extracting TDA features as trading indicators
- **Data sources:** Shows how to structure blockchain data for TDA
- **Validation:** Proof that TDA improves prediction vs. traditional ML

---

## 3. Topological Data Analysis for Identifying Critical Transitions in Cryptocurrency Time Series
**Saengduean, Noisagool, Chamchod (2018)**

### Key Contributions
- Detected financial crashes using TDA on Bitcoin/Ethereum (2018 crash)
- L1-norm and C1-norm of persistence landscapes peak BEFORE crashes
- Combined TDA with k-means clustering for regime detection
- Identified optimal time window and embedding dimension

### Methodology
1. Create embedded point cloud from price time-series:
   - Takens' embedding (time-delay reconstruction)
   - Variable window sizes (5-30 days optimal)
   - Variable embedding dimension (3-10 optimal)
2. Compute persistent homology
3. Extract persistence landscapes
4. Monitor C1-norm (sum of point heights)
5. Trigger alert when C1-norm spikes

### Key Findings
- **Critical finding:** C1-norm peaks 10-20 days BEFORE crash
- Early warning signals detectable from topology alone
- Window size and embedding dimension crucial for performance
- Works for both mini-crashes and major crashes

### Applicable to Project
- **Strategy 1 - Core signal:** Use C1-norm peaks as sell signals
- Explains WHEN topology changes matter
- Provides parameter optimization framework
- Validates early warning capability

---

## 4. Enhancing Financial Time Series Forecasting through Topological Data Analysis
**Jesus Jr., Fernández-Navarro, Carbonero-Ruz (2025)**

### Key Contributions
- Integrated TDA features into N-BEATS neural network
- Tested on 6 cryptocurrencies + 4 traditional assets
- Achieved best mean performance across MAPE, MAE, RMSE
- Compared against decomposition and time-delay embedding baselines

### Methodology
1. Compute traditional time-series features + N-BEATS baseline
2. Additionally extract TDA features:
   - Entropy of persistence diagram
   - Amplitude (maximum height)
   - Number of significant points
3. Create hybrid model: N-BEATS + TDA features
4. Compare: baseline, +decomposition, +time-delay, +TDA

### Key Findings
- TDA features statistically significantly improve forecasting (α=0.10)
- TDA outperforms traditional decomposition and embedding methods
- Consistent improvement across 32 test scenarios
- Amplitude feature most important; entropy also valuable

### Applicable to Project
- **Strategy 1:** Use entropy, amplitude, number-of-points as feature set
- Validates hybrid ML approach: traditional models + TDA
- Shows TDA beats manual feature engineering
- Parameter selection validated across datasets

---

## 5. Dissecting Ethereum Blockchain Analytics: Topology and Geometry of the Ethereum Graph
**Li, Islambekov, Akcora, Smirnova, Gel, Kantarcioglu (2020)**

### Key Contributions
- Applied TDA + functional data analysis to Ethereum transaction graph
- Showed local topology predicts token price movements
- Demonstrated that global network features alone insufficient
- Real-time transaction graph availability (new in finance)

### Methodology
1. Build transaction graph from Ethereum blockchain:
   - Nodes = addresses
   - Edges = transactions
2. Compute topological properties:
   - Local clustering patterns
   - Ego-network density
   - Functional data depth (FDA)
3. Extract features for price prediction
4. Compare: global features vs. local topology vs. combined

### Key Findings
- **Critical finding:** Local topology essential; global features miss patterns
- Ethereum enables fine-grained transaction analysis (real-time, public)
- Functional data analysis captures network evolution better than snapshots
- Topology reveals hidden co-movements between tokens

### Applicable to Project
- **Strategy 2:** Foundation for exchange flow network analysis
- Validates that local structure (not just global stats) matters
- Shows how to apply TDA to transaction networks
- Provides FDA framework for time-evolving networks

---

## 6. BitcoinHeist: Topological Data Analysis for Ransomware Detection on the Bitcoin Blockchain
**Akcora, Li, Gel, Kantarcioglu (2019)**

### Key Contributions
- Automated ransomware detection using TDA on Bitcoin
- Identified malicious address clusters via topology
- High precision/recall compared to heuristic baselines
- Zero-day ransomware family detection capability

### Methodology
1. Build Bitcoin transaction graph from blockchain
2. Identify known ransomware addresses (labeled dataset)
3. Extract neighborhoods around known malicious addresses
4. Compute persistent homology on address neighborhoods
5. Use topological features for anomaly scoring
6. Cluster similar patterns (topology-based clustering)

### Key Findings
- TDA-based clustering outperforms heuristic-based approaches
- Topology captures hidden relationships in transaction flows
- Works for new ransomware families (generalization)
- Real-time detection feasible

### Applicable to Project
- **Strategy 2 - Direct application:** Same approach for exchange manipulation
- Validates that topology detects coordinated malicious behavior
- Provides clustering & anomaly scoring framework
- Shows how to scale to large graphs

---

## Synthesis: Common Threads

### Consistent Findings Across Papers
1. **TDA captures hidden structures** that traditional methods miss
2. **Multi-scale perspective matters** - persistence diagrams show robust features
3. **Local topology crucial** - neighborhood structure beats global statistics
4. **Real-time feasible** - blockchain data available immediately
5. **Generalization works** - trained on known patterns → detects new patterns

### Parameter Patterns
- **Embedding dimension:** 3-10 for price series
- **Window size:** 5-30 days optimal for crash detection
- **Feature selection:** Entropy, amplitude, C1-norm are robust
- **Threshold tuning:** Context-dependent; backtesting required

### TDA Features That Work
1. **Entropy** - captures distribution complexity
2. **Amplitude** - max persistence (strongest signal)
3. **C1-norm** - sum of all persistence (cumulative signal)
4. **Number of features** - count of significant components
5. **Barcode length** - lifetime of topological features

---

## Implementation Priorities

### Must-Have (Core Algorithm)
1. Persistent homology computation (Rips complex)
2. Persistence diagram extraction
3. Persistence landscape computation
4. C1-norm calculation

### Should-Have (Enhanced Signals)
1. Multi-scale filtration (vary window sizes)
2. Embedding dimension optimization (Takens)
3. K-means clustering on landscapes
4. Time-delay reconstruction

### Nice-to-Have (Advanced)
1. Mapper algorithm (simplified networks)
2. Functional data analysis (FDA)
3. Entropy regularization
4. Wasserstein metric (persistence comparison)

---

## Risk/Limitations Identified in Literature

1. **Parameter sensitivity** - window size, embedding dim, filtration threshold matter
2. **Computational cost** - O(n²) for point cloud; needs optimization for real-time
3. **Cryptocurrency specificity** - training data crucial (not transferable)
4. **Lag risk** - early warning systems have 10-20 day lead; trading window tight
5. **Overfitting** - backtesting on known crashes risky; forward validation essential

---

## Recommended Reading Order

1. Start: Rivera-Castro (portfolio foundation)
2. Then: Saengduean (crash detection concept)
3. Then: Li et al. (blockchain-specific TDA)
4. Then: Jesus Jr. (feature integration)
5. Deep-dive: Hafez (implementation detail)
6. Reference: Akcora (anomaly scoring)

---

## Paper Statistics

| Paper | Year | Focus | Key Metric | 
|-------|------|-------|-----------|
| Rivera-Castro | 2019 | Portfolio | Outperformed classic methods |
| Hafez et al. | 2022 | Price prediction | 0.75% MAPE hourly |
| Saengduean et al. | 2018 | Crash detection | 10-20 day early warning |
| Jesus Jr. et al. | 2025 | Forecasting | Best mean performance (α=0.10) |
| Li et al. | 2020 | Ethereum analytics | Local topology essential |
| Akcora et al. | 2019 | Anomaly detection | High precision/recall |

---

## Dataset Benchmarks from Papers

- **Bitcoin:** 2019, public addresses, 10 years available
- **Ethereum:** 2015+, all transactions public, 3 interaction types
- **Cryptocurrencies tested:** 1561 (Rivera), 4 major (Gidea), 6 + 4 assets (Jesus Jr.)
- **Time periods:** 5-10 years historical, real-time for streaming

---

## Validation Metrics to Implement

From the papers, these metrics validated TDA effectiveness:

### Trading Performance
- Sharpe ratio
- Win rate  
- Maximum drawdown
- MAPE/MAE/RMSE (forecasting)

### Anomaly Detection
- Precision / Recall
- F1 score
- ROC-AUC
- False positive rate

### Topology Quality
- Persistence stability (Wasserstein distance)
- Feature robustness (vary parameters)
- Cross-validation (temporal splits)
