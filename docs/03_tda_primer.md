# Topological Data Analysis Primer

## What is TDA?

Topological Data Analysis is a mathematical framework for extracting meaningful structures from high-dimensional data by studying its shape and connectivity patterns.

### Core Insight

Traditional statistics describe: central tendency, variance, distribution
TDA describes: holes, loops, voids, connected components, clustering

```
Traditional stats:     μ = 5.2, σ = 1.3, normal distribution
TDA adds:            "Data has 2 clusters + 1 loop connecting them"
```

### Why TDA for Crypto?

1. **Cryptocurrency is high-dimensional:** OHLCV + 10+ indicators + on-chain metrics
2. **Noise-robust:** Topology survives small perturbations
3. **Captures non-linear structure:** Finds loops and voids traditional methods miss
4. **Multi-scale:** Sees structure at all zoom levels simultaneously
5. **Novel for finance:** Relatively unexplored advantage vs. competitors

---

## Key Concepts

### 1. Point Clouds

A point cloud is a finite set of points in high-dimensional space.

```
Example: Price time series
Date    Open  High  Low   Close  Vol      RSI   MACD   ...
2024-01 100   101   99    100.5  1000000  55.2  0.3    → Point 1 in 11D space
2024-02 100.5 102   99.8  101.2  950000   58.1  0.5    → Point 2 in 11D space
2024-03 101.2 103   100   102.8  1100000  62.3  0.8    → Point 3 in 11D space
...

Visualization in 3D (project via PCA):
    *
   / \
  *   *
 / \ / \
*   *   *
```

### 2. Simplicial Complexes

A simplicial complex is a higher-dimensional generalization of a graph.

```
Dimension:
  0-simplex = point
  1-simplex = edge (line segment connecting 2 points)
  2-simplex = triangle (face connecting 3 points)
  3-simplex = tetrahedron (4 points)
  ...

Construction: Rips complex
  Start with all points
  Add edges if distance < ε
  Add triangles if all 3 edges exist
  Add tetrahedra if all 6 edges exist
  Repeat for increasing ε
```

### 3. Homology

Homology counts topological features at each dimension.

```
H₀ (0-dimensional homology):
  Counts connected components
  Usually: initially n components, decreases as ε grows

H₁ (1-dimensional homology):
  Counts loops (1-cycles)
  Appears when triangle closes (form a hole)
  Dies when triangle fills (hole disappears)

H₂ (2-dimensional homology):
  Counts voids/cavities
  Rare in practical datasets

Example:
  H₀ = {n - 5, n - 4, ..., 1}  (components merge)
  H₁ = {0, 0, 1, 1, 2, 1, 0, ...} (loops appear/disappear)
```

### 4. Persistent Homology

Standard homology asks: "Does this loop exist at this instant?"
Persistent homology asks: "At what scales does this loop matter?"

```
Key insight: Features that exist across many scales are robust/signal

Illustration:
       ε = 1.0     ε = 2.0     ε = 3.0     ε = 4.0
        •           •           •           •
       / \         /•\         /•\         /•\
      •   •       •   •       •   •       •   •
       \ /         \•/         \•/         \•/
        •           •           •           •

Loop appears at ε = 1.5 (birth)
Loop disappears at ε = 3.5 (death)
Persistence = 3.5 - 1.5 = 2.0

For trading: Long persistence = real market structure
             Short persistence = noise
```

### 5. Persistence Diagram

A visualization showing all birth-death pairs.

```
Death
  |
  |     *
  |    * *
  |   *   * *
  |  *       *
  | *         *
  |*___________*_____ Birth

Interpretation:
  - Points far from diagonal = long persistence (signal)
  - Points near diagonal = short persistence (noise)
  - Clustering = multiple features at similar scales
```

### 6. Persistence Landscape

Convert persistence diagram to a continuous function.

```
For each feature (bᵢ, dᵢ):
  λᵢ(t) = max(0, min(t - bᵢ, dᵢ - t))

λᵢ(t) visualized:
      /\
     /  \
    /    \
  b₁    d₁
  |      |
0 ─────────── time

For multiple features:
      /\  /\
     /  \/  \
    /        \

Key metric: C₁-norm = max value of all landscapes stacked
            L₁-norm = ∫ landscape(t) dt (area under curve)
```

### 7. Mapper Algorithm

A simplified representation of a dataset's topology.

```
Algorithm:
1. Choose a lens function: f(x) → ℝ (projects high-D → 1D or 2D)
2. Cover the range: Overlapping bins
3. Cluster within each bin: k-means → nodes
4. Connect bins: Edges between clusters in adjacent bins

Example (exchange networks):
   Lens = "exchange total volume"
   
   Volume 0-100K:   Cluster A, B, C
   Volume 100-200K: Cluster D, E (overlaps with previous)
   Volume 200K+:    Cluster F

   Mapper graph:
   A---D---F
    \ / \ /
     B   E
      \ /
       C
```

---

## Mathematics (Simplified)

### Rips Complex Filtration

```
Rips(X, ε) = simplicial complex where
  - 0-simplices = all points
  - 1-simplices = {(p,q) : d(p,q) ≤ ε}
  - 2-simplices = {(p,q,r) : all pairwise d ≤ ε}

Filtration: Rips(X, ε₀) ⊆ Rips(X, ε₁) ⊆ Rips(X, ε₂) ⊆ ...

Homology groups: H₀(Rips(ε)), H₁(Rips(ε)), H₂(Rips(ε))

Persistence: For each feature,
  - birth(f) = ε where f first appears
  - death(f) = ε where f disappears
  - pers(f) = death(f) - birth(f)
```

### Betti Numbers

Betti numbers are ranks of homology groups.

```
β₀ = # connected components
β₁ = # independent loops
β₂ = # independent voids

Example:
  Torus: β₀=1, β₁=2, β₂=1
  Circle: β₀=1, β₁=1, β₂=0
  Line: β₀=1, β₁=0, β₂=0

For persistent homology:
  β₀(t) = number of components at time t
  β₁(t) = number of loops at time t
```

---

## TDA for Time Series

### Takens' Embedding

Convert 1D time series → high-D point cloud while preserving topology.

```
Original series: {x₁, x₂, x₃, x₄, x₅, x₆, x₇, ...}

Embedding (dimension d=3, delay τ=1):
  Point 1: (x₁, x₂, x₃)
  Point 2: (x₂, x₃, x₄)
  Point 3: (x₃, x₄, x₅)
  ...

Why: Captures nonlinear dependencies
     Reconstructs attractor geometry
     Works better than direct feature engineering

Parameters:
  - Dimension d: 3-10 typical (try different values)
  - Delay τ: Usually 1 (time unit), adjust if autocorrelated
```

### Sliding Window + TDA

```
Time series: [—————————————————————————————————————————]

Window 1:    [——————]
Window 2:        [——————]
Window 3:            [——————]

For each window:
  Extract d=5 dimensional point cloud
  Compute persistent homology
  Extract features (C₁-norm, entropy, count)

Result: Time series of persistence features
        Track how topology evolves!
```

---

## TDA Metrics Explained

### C₁-Norm (Most Important)

```
Definition: C₁(t) = Σᵢ Σₜ max(0, min(t - bᵢ, dᵢ - t))

Interpretation: Sum of all persistence peaks
               Higher = more structure
               Spikes = new features appearing

For trading: When C₁-norm suddenly INCREASES → structure change
            When C₁-norm DECREASES → structure dissolving
            Peak before crash → early warning signal
```

### L₁-Norm

```
Definition: L₁ = ∫ landscape(t) dt

Interpretation: Total "area" of persistence
               Integrates across all features
               More robust than single max

Use: Compare periods; high L₁ = structured market
     low L₁ = chaos
```

### Entropy

```
Definition: Entropy = -Σᵢ pᵢ log(pᵢ)
where pᵢ = normalized persistence length of feature i

High entropy: Many features of varying persistence (complex)
Low entropy: Few dominant features (simple structure)

Use: Confidence scoring
     High entropy = uncertain (reduce position)
     Low entropy = clear structure (increase position)
```

### Feature Count

```
Definition: # of (b,d) pairs where (d - b) > threshold

Interpretation: # of "significant" topological features
               Many features = rich structure
               Few features = sparse/noisy

Use: Detection quality
     Sudden ↑ count = regime change detected
     Stable count = steady state
```

---

## Practical Example: Detecting Bull-to-Bear Transition

### Scenario

```
Date:  Jan  Feb  Mar  Apr  May  Jun
Price: 100  110  115  118  120  85     ← Crash in June
```

### Step 1: Create Point Clouds

Window size = 5 days, slide by 1 day

```
Days 1-5:   Points with (Open₁, High₁, Low₁, Close₁, Vol₁, RSI₁, MACD₁, ...)
Days 2-6:   Points with (Open₂, High₂, Low₂, Close₂, Vol₂, RSI₂, MACD₂, ...)
...
Days 150-154: Points with (Open₁₅₀, ..., MACD₁₅₀, ...)
```

### Step 2: Compute Persistent Homology

For each point cloud:
  - Build Rips complex
  - Compute homology across ε values
  - Extract persistence diagram

### Step 3: Extract Features

For each window's persistence diagram:
  - C₁-norm(t)
  - L₁-norm(t)
  - Entropy(t)
  - Feature count(t)

### Step 4: Track Changes

```
Time series of C₁-norm:
            ╱╲
           ╱  ╲
          ╱    ╲____
         ╱          ╲╲╲ ← Sharp drop = transition
        ╱              ╲
       ╱                ╲


Interpretation:
- Jan-May: C₁-norm stable (bull structure)
- Late May: C₁-norm starts dropping
- Early June: C₁-norm crashes (structure breaks)
- Crash happens → C₁-norm was leading indicator!
```

### Step 5: Trading Signal

```
ALERT: C₁-norm drop > 30% = SELL signal
       (or reduce position)

Confidence: Based on entropy
           Low entropy (clear structure) = high confidence
           High entropy (noisy) = low confidence
```

---

## Common Pitfalls

### 1. Overfitting to Historical Data

```
Problem: TDA finds patterns that won't repeat
Solution: Walk-forward validation (test on unseen future data)
         Parameter sensitivity analysis
         Ensemble multiple thresholds
```

### 2. Computational Cost

```
Problem: Computing persistent homology is expensive (O(n²) or O(n³))
Solution: Downsample point clouds
         Use streaming/incremental TDA
         Cache previous computations
         Use fast libraries (ripser, giotto)
```

### 3. Parameter Sensitivity

```
Problem: Results change with window size, embedding dim, etc.
Solution: Test range of parameters
         Report results with error bars
         Use robust metrics (rank rather than absolute)
```

### 4. False Signals

```
Problem: Noise creates fake topological features
Solution: Use persistence threshold (only long-lived features)
         Smooth landscapes before decision-making
         Combine with other indicators
```

---

## Tools & Libraries

### Python Libraries

```
giotto-tda
  - Full-featured TDA library
  - Persistent homology, Mapper, Ripser wrapper
  - pip install giotto-tda

ripser
  - Fast C++ implementation
  - Easiest to use for basic needs
  - pip install ripser

persim
  - Persistence metric computations
  - Compare diagrams (Wasserstein distance)
  - pip install persim

scikit-tda
  - Pure Python, educational
  - Slower but easy to understand

networkx
  - Graph/network analysis (for Mapper output)
  - pip install networkx
```

### Visualization

```
matplotlib
  - Persistence diagrams
  - Landscapes
  
plotly
  - Interactive 3D point clouds
  
gephi/cytoscape
  - Mapper network visualization
```

---

## Recommended Learning Path

1. **Week 1:** Understand point clouds, simplicial complexes, homology basics
2. **Week 2:** Work through Takens' embedding, sliding windows
3. **Week 3:** Compute persistence diagrams on toy datasets
4. **Week 4:** Extract C₁-norm, L₁-norm, entropy; understand interpretation
5. **Week 5:** Apply to real price data; compare features to price movements
6. **Week 6:** Build simple trading rule on TDA features
7. **Week 7:** Backtest and optimize parameters
8. **Week 8:** Forward test on new data (validation)

---

## Key Takeaways

1. **TDA captures shape:** Sees loops, holes, voids that statistics miss
2. **Persistence = robustness:** Long-lived features > short-lived noise
3. **Multi-scale:** Sees structure at all zoom levels simultaneously
4. **Interpretable:** C₁-norm peaks = market structure changes
5. **Applicable:** Works for price series AND network analysis
6. **Not a silver bullet:** Still needs risk management, validation, monitoring

---

## Further Reading

- Edelsbrunner, H., & Harer, J. (2010). "Computational Topology"
- Ghrist, R. (2008). "Barcodes: The persistent topology of data"
- Carlsson, G. (2009). "Topology and data" (foundational TDA paper)
- Pun, C. S., Xia, K., & Lee, S. X. (2018). "Persistent-homology-based machine learning"
