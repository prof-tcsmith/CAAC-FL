# Quick Reference Guide - CAAC-FL Literature Review

## Finding Specific Information

### Attack Implementations

| Attack | Paper | Section | Key Formula |
|--------|-------|---------|-------------|
| ALIE | Baruch et al. 2019 | Section 3 | z_max from cumulative normal, malicious = μ - z_max·σ |
| IPM (ε=0.5) | Li et al. 2024 | Section 8 | Small-scale stealthy attack |
| IPM (ε=100) | Li et al. 2024 | Section 8 | Large-scale gradient reversal |
| Label Flipping | Yin et al. 2018 | Section 1 | l → M-l-1 |
| Omniscient | Blanchard et al. 2017 | Section 2 | -λ·∇Q(xt) |
| Gaussian Noise | Multiple papers | Various | N(0, σ²I) |
| Adaptive | Cao et al. 2021 | Section 4 | Zeroth-order optimization |

### Baseline Implementations

| Method | Paper | Key Parameters | Complexity |
|--------|-------|----------------|------------|
| Krum | Blanchard et al. 2017 | f < (n-2)/2 | O(n²·d) |
| Trimmed Mean | Yin et al. 2018 | β = trim fraction | O(n·d·log n) |
| Geometric Median | Pillutla et al. 2022 | R=3 iterations, ν=10^-6 | O(R·n·d) |
| FLTrust | Cao et al. 2021 | 100-example root dataset | O(n·d) |
| LASA | Xu et al. 2024 | SL=0.3, λ_m=1.0, λ_d=1.0 | O(n·d·log d) |
| ClippedClustering | Li et al. 2024 | τ=50th percentile norm | O(n²·d) |
| BR-MTRL | Le et al. 2025 | τ_h=10, τ_φ=1 | O(n·d) |

### Dataset Configurations

| Dataset | Paper | Clients | Distribution | Byzantine % |
|---------|-------|---------|--------------|-------------|
| MNIST | Yin et al. 2018 | 10-40 | IID | 5-10% |
| CIFAR-10 | Li et al. 2024 | 20 | Dir(0.1) | 25% |
| CIFAR-10 | Cao et al. 2021 | 100 | q=0.5 | 20-95% |
| FEMNIST | Le et al. 2025 | 150 | Natural (writers) | 33% |
| MIMIC-III | CAAC-FL | 20 | Dir(0.5) | 20-40% |

### Performance Benchmarks

| Method | Dataset | Setting | Accuracy | Source |
|--------|---------|---------|----------|--------|
| Krum | CIFAR-10 | Non-IID, no attack | 10.00% | Li et al. 2024 |
| GeoMed | CIFAR-10 | Non-IID, no attack | 10.00% | Li et al. 2024 |
| TrimmedMean | CIFAR-10 | Non-IID, no attack | 80.98% | Li et al. 2024 |
| ClippedClustering | CIFAR-10 | Non-IID, no attack | 81.58% | Li et al. 2024 |
| FLTrust | MNIST | IID, 20% Byzantine | 96%+ | Cao et al. 2021 |
| LASA | CIFAR-10 | IID, 25% ByzMean | 89.08% | Xu et al. 2024 |
| BR-MTRL+Krum | CIFAR-10 | Non-IID, 20% Byzantine | 80.25% | Le et al. 2025 |

### Evaluation Metrics

| Metric | Papers Using | Definition | Target for CAAC-FL |
|--------|--------------|------------|-------------------|
| Test Accuracy | All | % correct predictions | >80% (non-IID, no attack) |
| TPR (Detection) | Xu et al. 2024 | % Byzantine correctly identified | >90% |
| FPR (Detection) | Xu et al. 2024 | % benign misclassified | <10% |
| Convergence Speed | Yin et al. 2018 | Rounds to target accuracy | Comparable to FedAvg |
| Attack Success Rate | Cao et al. 2021 | % backdoor triggers successful | <5% |
| Communication Cost | Pillutla et al. 2022 | Oracle calls per round | <3× baseline |

### Key Findings by Topic

#### Non-IID Challenge
- **Source:** Li et al. (2024) Section 8
- **Finding:** Distance-based schemes achieve 10% on non-IID without attacks
- **Implication:** Global thresholds fail to distinguish heterogeneity from attacks
- **CAAC-FL Solution:** Client-adaptive behavioral profiling

#### Variance Exploitation
- **Source:** Baruch et al. (2019) Section 3
- **Finding:** ALIE operates within σ bounds to evade detection
- **Implication:** Statistical defenses vulnerable when |μ| < z·σ
- **CAAC-FL Solution:** Temporal consistency tracking

#### Theoretical Foundation
- **Source:** Werner et al. (2023) Section 5
- **Finding:** Client-specific adaptation achieves O(1/√niT) convergence
- **Implication:** Personalization enhances robustness (provably)
- **CAAC-FL Application:** Theoretical justification for behavioral profiles

#### Privacy-Robustness Tradeoff
- **Source:** Pillutla et al. (2022) Section 6
- **Finding:** Cannot have privacy + communication + robustness simultaneously
- **Implication:** 3× communication cost for privacy-preserving robustness (RFA)
- **CAAC-FL Advantage:** Local behavioral profiling (no communication overhead)

## Code Snippets Location

### ALIE Attack Implementation
**Document:** reference-analysis-summary.md
**Section:** "2. ALIE Attack Implementation (Baruch et al.)"
**Lines:** ~320-340
```python
def alie_attack(benign_gradients, n_total, n_byzantine):
    s = math.floor(n_total / 2 + 1) - n_byzantine
    target_prob = (n_total - n_byzantine - s) / (n_total - n_byzantine)
    z_max = norm.ppf(target_prob)
    mu = benign_gradients.mean(dim=0)
    sigma = benign_gradients.std(dim=0)
    return mu - z_max * sigma
```

### Behavioral Profile Tracking
**Document:** reference-analysis-summary.md
**Section:** "5. Behavioral Profile Tracking (CAAC-FL Core)"
**Lines:** ~480-580
- EWMA mean and variance tracking
- Multi-dimensional anomaly score computation
- Magnitude, directional, temporal components

### Non-IID Partitioning
**Document:** reference-analysis-summary.md
**Section:** "1. Non-IID Partitioning"
**Lines:** ~280-310
- Dirichlet distribution implementation
- Class-based allocation

### Pairwise Cosine Similarity
**Document:** reference-analysis-summary.md
**Section:** "3. Pairwise Cosine Similarity Analysis"
**Lines:** ~360-380
- Explains differential robustness
- Used in Li et al. analysis

### Adaptive Attack Framework
**Document:** reference-analysis-summary.md
**Section:** "4. Adaptive Attack Framework"
**Lines:** ~400-430
- Zeroth-order optimization
- Defense-specific attack crafting

## Experimental Protocol Templates

### Standard Evaluation Pipeline
1. Baseline establishment (no attack, Mean aggregation)
2. Individual attack evaluation (5+ attack types)
3. Comparison across aggregation schemes (7+ schemes)
4. IID vs Non-IID comparison
5. FedSGD vs FedAvg comparison
6. Sensitivity analysis (Byzantine fraction, batch size)
7. Adaptive attack testing

### Minimum Experimental Suite (from Li et al.)
- **Datasets:** At least 3 (1 IID, 1 non-IID synthetic, 1 non-IID natural)
- **Attacks:** At least 5 (1 naive, 4 SOTA)
- **Baselines:** At least 4 (non-robust, spatial, adaptive, personalized)
- **Metrics:** Accuracy, TPR, FPR, convergence speed

### Ablation Study Template
1. Component isolation (magnitude only, direction only, temporal only)
2. Pairwise combinations
3. Full method
4. Hyperparameter sensitivity
5. Architecture variants

## Cross-Paper Comparisons

### Byzantine Tolerance Thresholds

| Method | Max Byzantine % | Source |
|--------|-----------------|--------|
| Krum | <50% (f < (n-2)/2) | Blanchard et al. |
| Trimmed Mean | <50% | Yin et al. |
| Geometric Median | <50% (breakdown point) | Pillutla et al. |
| FLTrust | Tested up to 95% | Cao et al. |
| LASA | <50% assumed | Xu et al. |
| ClippedClustering | Effective up to 25% | Li et al. |

### Actual Performance Under Attack

| Method | Setting | Byzantine % | Accuracy | Source |
|--------|---------|-------------|----------|--------|
| Krum | CIFAR-10, IPM ε=0.5 | 25% | ~15% | Li et al. |
| Median | CIFAR-10, IPM ε=0.5 | 25% | ~20% | Li et al. |
| ClippedClustering | CIFAR-10, IPM ε=0.5 | 25% | ~70% | Li et al. |
| FLTrust | MNIST, Adaptive | 45% | ~92% | Cao et al. |
| BR-MTRL+GM | CIFAR-10, Gaussian | 20% | 72.94% | Le et al. |

## CAAC-FL Success Criteria

Based on cross-paper analysis:

### Minimum Requirements
- [x] >80% accuracy on non-IID without attacks (vs. 10% for Krum/GeoMed)
- [x] >70% accuracy with 20% Byzantine clients (vs. <20% for most defenses)
- [x] TPR >90%, FPR <10% for Byzantine detection
- [x] Detection latency <10 rounds

### Competitive Targets
- [ ] Match or exceed ClippedClustering (81.58%) on non-IID baseline
- [ ] Outperform FLTrust (no centralized root dataset needed)
- [ ] Better than LASA on temporal attacks (slow-drift)
- [ ] Comparable communication cost to FedAvg

### Novel Contributions
- [ ] Temporal behavioral tracking (not in any baseline)
- [ ] Distinguishes heterogeneity from attacks (Li et al. gap)
- [ ] Proactive Byzantine exclusion (vs. reactive mitigation)
- [ ] Multi-dimensional consistency metrics

## Quick Lookup: Paper-Specific Strengths

| Paper | Best For | Key Takeaway |
|-------|----------|--------------|
| Yin et al. 2018 | Theoretical foundations | Coordinate-wise filtering, convergence rates |
| Blanchard et al. 2017 | Krum baseline | Distance-based selection, first provable defense |
| Baruch et al. 2019 | ALIE attack | Variance exploitation, defeats statistical defenses |
| Cao et al. 2021 | Adaptive attacks | Trust bootstrapping, zeroth-order optimization |
| Pillutla et al. 2022 | Geometric median | Privacy-robustness tradeoff, adaptive weighting |
| Werner et al. 2023 | Theory validation | Client-specific adaptation proof, gradient clustering |
| Xu et al. 2024 | Layer-wise defense | Sparsification, PDP metric, multi-attack suite |
| Li et al. 2024 | Non-IID problem | <10% failure, comprehensive baseline comparison |
| Le et al. 2025 | Multi-task learning | Shared + personalized architecture, AWS testbed |

---

**Usage:** Use Ctrl+F to search for specific topics, paper names, or metrics
**Related:** See reference-analysis-summary.md for full details
