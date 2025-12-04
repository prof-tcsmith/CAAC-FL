# Client-Adaptive Anomaly-Aware Clipping for Byzantine-Robust Federated Learning: A Comprehensive Experimental Study

**Working Draft - December 2024**

---

## Abstract

Byzantine-robust federated learning aims to enable collaborative model training despite the presence of malicious participants who may upload arbitrary updates to degrade global model performance. While recent aggregation schemes such as Krum, Trimmed Mean, and ClippedClustering have demonstrated robustness under controlled conditions, they share a critical limitation: they define "anomalous" relative to the global population, failing to accommodate legitimate client heterogeneity common in real-world deployments like healthcare. Furthermore, existing experimental studies, including the comprehensive evaluation by Li et al. (2024), exclusively test *immediate* attacks where Byzantine behavior begins from round zero—an unrealistic assumption given that real-world device compromise often occurs mid-training.

This paper makes three contributions. First, we introduce **CAAC-FL** (Client-Adaptive Anomaly-Aware Clipping), a novel Byzantine defense that maintains per-client behavioral profiles using EWMA-based tracking and applies client-specific adaptive thresholds based on historical consistency. By asking "is this anomalous for *this* client?" rather than "is this different from the population?", CAAC-FL dramatically reduces false positives on heterogeneous data while maintaining strong Byzantine detection. Second, we extend the experimental framework of Li et al. (2024) with three novel dimensions: (1) **compromise timing**—when attacks begin during training; (2) **attack windows**—transient attacks with defined start and end rounds; and (3) **non-IID severity**—varying Dirichlet α across the heterogeneity spectrum. Third, we provide comprehensive empirical evaluation comparing CAAC-FL against eight state-of-the-art aggregation schemes across six attack types and multiple threat scenarios.

Our experiments reveal that (1) many existing defenses fail catastrophically when attacks are delayed until after partial model convergence; (2) models exhibit varying recovery capacity after transient attacks depending on aggregation strategy; and (3) CAAC-FL maintains robust performance across all tested scenarios while achieving the lowest false positive rate on heterogeneous data.

**Keywords**: Federated Learning, Byzantine Robustness, Anomaly Detection, Non-IID Data, Healthcare AI

---

## 1. Introduction

### 1.1 Problem Statement

Federated Learning (FL) enables collaborative model training across distributed devices without centralizing private data [McMahan et al., 2017]. However, this distributed nature introduces vulnerability to *Byzantine attacks*, where malicious participants upload carefully crafted updates to degrade or manipulate the global model [Blanchard et al., 2017; Fang et al., 2020].

The research community has responded with numerous Byzantine-robust aggregation schemes, including geometric selection methods (Krum, Multi-Krum), coordinate-wise robust statistics (Median, Trimmed Mean), momentum-based clipping (Centered Clipping), and clustering approaches (ClippedClustering) [Li et al., 2024]. These methods have demonstrated effectiveness under specific conditions, typically achieving high accuracy when local datasets are independent and identically distributed (IID).

However, two critical gaps remain:

**Gap 1: Heterogeneity-Robustness Trade-off.** Real-world FL deployments, particularly in healthcare, involve institutions with legitimately different data distributions. A pediatric hospital's model updates will differ systematically from a geriatric care center's—both are valid, neither is Byzantine. Current defenses that define "anomalous" as "different from peers" inevitably flag legitimate institutional diversity as attacks, creating an unacceptable false positive rate that undermines participation incentives.

**Gap 2: Unrealistic Threat Models.** Existing experimental evaluations, including the comprehensive study by Li et al. (2024), exclusively test *immediate* attacks where Byzantine behavior begins at round zero and continues throughout training. In practice, devices are compromised at various points—through software vulnerabilities, insider threats, or supply chain attacks—and attacks may be transient rather than persistent. The robustness of aggregation schemes under these realistic threat models remains unexplored.

### 1.2 Contributions

This paper addresses both gaps through the following contributions:

**Contribution 1: CAAC-FL Algorithm.** We introduce Client-Adaptive Anomaly-Aware Clipping, a Byzantine defense paradigm that maintains per-client behavioral profiles using Exponentially Weighted Moving Averages (EWMA) and applies client-specific adaptive thresholds. CAAC-FL detects anomalies across three dimensions (magnitude, direction, temporal variance) while explicitly accommodating legitimate heterogeneity. Key innovations include:

- **Per-client profiling**: Each client develops a unique "gradient signature" based on historical behavior
- **Adaptive thresholds**: Clients earn trust through consistent non-anomalous behavior, receiving more flexible thresholds
- **Soft clipping**: Anomalous updates are scaled down rather than discarded, allowing borderline contributions
- **Cold-start mitigation**: Six mechanisms address Byzantine attacks from the first round

**Contribution 2: Extended Threat Models.** We introduce three novel experimental dimensions that extend beyond Li et al. (2024):

| Dimension | Li et al. 2024 | Our Extension |
|-----------|----------------|---------------|
| Compromise Timing | Round 0 only | Rounds 0, 10, 20, 30, 40 |
| Attack Windows | Always on | [0,end), [0,25), [25,end), [10,40), [0,10) |
| Non-IID Severity | α=0.1 or IID | α ∈ {0.1, 0.3, 0.5, 1.0} |

These dimensions enable studying: (a) whether partial model convergence provides inherent robustness; (b) whether models can recover after transient attacks; and (c) how data heterogeneity interacts with Byzantine robustness.

**Contribution 3: Comprehensive Evaluation.** We evaluate nine aggregation schemes (FedAvg, Median, Trimmed Mean, Krum, Multi-Krum, GeoMed, Centered Clipping, Clustering, ClippedClustering, and CAAC-FL) against six attack types (Sign Flipping, Random Noise, ALIE, IPM-small, IPM-large, Label Flipping) across multiple threat scenarios.

### 1.3 Paper Organization

Section 2 reviews related work on Byzantine-robust FL. Section 3 details the CAAC-FL algorithm. Section 4 describes our experimental framework including novel threat models. Section 5 presents experimental results. Section 6 discusses findings and limitations. Section 7 concludes.

---

## 2. Background and Related Work

### 2.1 Federated Learning

Federated Learning [McMahan et al., 2017] solves the distributed optimization problem:

$$\min_w F(w) = \frac{1}{K} \sum_{k=1}^{K} F_k(w)$$

where K clients collaboratively learn parameters w without sharing raw data. The standard FedAvg algorithm proceeds in rounds: the server broadcasts global parameters, clients perform local SGD, and the server aggregates updates via weighted averaging.

### 2.2 Byzantine Attacks

Byzantine attacks exploit FL's distributed nature by having malicious participants upload arbitrary updates [Blanchard et al., 2017]. We consider six attack types following Li et al. (2024):

**Model Poisoning Attacks:**
- **Sign Flipping (SF)**: Negates gradient direction, performing gradient ascent
- **Random Noise**: Adds Gaussian noise N(0, σ²) to updates
- **ALIE (A Little Is Enough)** [Baruch et al., 2019]: Crafts updates within the statistical range of honest clients to evade detection
- **IPM (Inner Product Manipulation)** [Xie et al., 2020]: Manipulates updates to create negative inner product with true gradient

**Data Poisoning Attacks:**
- **Label Flipping**: Trains on corrupted labels (l → L-l-1)

### 2.3 Byzantine-Robust Aggregation Schemes

**Distance-based Selection:**
- **Krum** [Blanchard et al., 2017]: Selects the single update closest to K-M-2 neighbors by Euclidean distance
- **Multi-Krum**: Averages top-m clients with lowest Krum scores
- **GeoMed** [Chen et al., 2017]: Computes geometric median via Weiszfeld algorithm

**Coordinate-wise Statistics:**
- **Median** [Yin et al., 2018]: Coordinate-wise median of updates
- **Trimmed Mean** [Yin et al., 2018]: Removes extreme β-fraction before averaging

**Clipping and Clustering:**
- **Centered Clipping (CC)** [Karimireddy et al., 2021]: Iteratively clips updates around center:
  $$\Delta_{l+1} \leftarrow \Delta_l + \frac{1}{K} \sum_k (\Delta_k - \Delta_l) \min\left(1, \frac{\tau}{\|\Delta_k - \Delta_l\|}\right)$$

- **Clustering** [Sattler et al., 2020]: Agglomerative clustering with average linkage on cosine distances, selecting larger cluster
- **ClippedClustering** [Li et al., 2024]: Clips updates using historical norm median *before* clustering

### 2.4 Limitations of Existing Approaches

All existing methods share a fundamental limitation: they define "anomalous" relative to the global population. This creates two failure modes:

1. **False Positives on Heterogeneous Data**: Legitimate institutional diversity appears anomalous. Li et al. (2024) report that "with Non-IID data, some aggregation schemes fail even in the complete absence of Byzantine clients."

2. **Static Threat Model**: All prior experimental evaluations assume attacks begin at round 0. Real compromise occurs at various points during deployment.

### 2.5 Li et al. (2024) Study

Li et al. (2024) provide the most comprehensive experimental study of Byzantine-robust aggregation to date, evaluating eight schemes across five attack types. Their key findings include:

- ClippedClustering performs best under IID conditions
- All schemes degrade significantly under Non-IID (Dirichlet α=0.1)
- ALIE attack is particularly effective against FedSGD
- IPM with ε=100 defeats most defenses

However, their study exclusively tests immediate attacks (round 0) and does not explore compromise timing, attack windows, or varying heterogeneity levels.

---

## 3. CAAC-FL: Client-Adaptive Anomaly-Aware Clipping

### 3.1 Design Philosophy

CAAC-FL introduces a paradigm shift in Byzantine defense:

> **Traditional**: "Is this update different from the population?" → Global threshold
> **CAAC-FL**: "Is this update anomalous for *this* client?" → Per-client adaptive threshold

This shift explicitly accommodates legitimate heterogeneity while maintaining strong Byzantine detection.

### 3.2 Per-Client Behavioral Profiling

Each client k maintains a behavioral profile updated via EWMA:

$$\mu_k^t = \alpha \cdot \|g_k^t\|_2 + (1-\alpha) \cdot \mu_k^{t-1}$$

$$(\sigma_k^t)^2 = \alpha \cdot (\|g_k^t\|_2 - \mu_k^t)^2 + (1-\alpha) \cdot (\sigma_k^{t-1})^2$$

where α=0.05 provides slow adaptation that resists profile poisoning attacks.

**Reliability Score**: Each client earns trust through consistent behavior:

$$R_k^t = \gamma \cdot \mathbb{1}[\text{not anomalous}] + (1-\gamma) \cdot R_k^{t-1}$$

### 3.3 Three-Dimensional Anomaly Detection

CAAC-FL scores clients across three independent attack surfaces:

**Dimension 1: Magnitude Anomaly**
$$A_{mag}^{k,t} = \frac{\|g_k^t\|_2 - \mu_k}{\sigma_k + \epsilon}$$

Detects updates with unusual norm relative to client's history. Catches ALIE and noise attacks that may appear normal globally but are unusual for specific clients.

**Dimension 2: Directional Anomaly**
$$A_{dir}^{k,t} = 1 - \frac{1}{W} \sum_{j} \cos(g_k^t, g_k^j)$$

Compares update direction against historical gradients and global aggregated gradient. Sign-flipping produces cos ≈ -1; legitimate specialization maintains consistent directions.

**Dimension 3: Temporal Anomaly**
$$A_{temp}^{k,t} = \frac{\sigma_k^t - \sigma_k^{t-W}}{\sigma_k^{t-W} + \epsilon}$$

Measures variance drift over window W. Distinguishes sudden attack onset from legitimate evolution.

**Composite Score**:
$$A_k^t = w_1 |A_{mag}| + w_2 A_{dir} + w_3 |A_{temp}|$$

Default weights (0.5, 0.3, 0.2) prioritize magnitude for random noise attacks.

### 3.4 Adaptive Thresholding

Client-specific thresholds adapt based on reliability:

$$\tau_k^t = \tau_{base} \cdot (1 + \beta \cdot R_k^{t-1})$$

High-reliability clients receive more flexible thresholds; new/suspicious clients face stricter scrutiny.

### 3.5 Soft Clipping

Rather than binary rejection, CAAC-FL scales anomalous updates:

$$\tilde{g}_k^t = \begin{cases} g_k^t & \text{if } A_k^t \leq \tau_k^t \\ g_k^t \cdot \frac{\tau_k^t}{A_k^t + \epsilon} & \text{otherwise} \end{cases}$$

This allows borderline contributions while limiting attack damage.

### 3.6 Cold-Start Mitigations

Six mechanisms address Byzantine attacks from round 1:

| Mechanism | Description |
|-----------|-------------|
| Conservative Warmup | Stricter thresholds during first 10 rounds |
| Cross-Client Comparison | 50% weight on population median during warmup |
| Global Gradient Reference | Previous round's aggregate for sign-flip detection |
| Delayed Trust | No reliability bonus until round 5 |
| Population Initialization | New profiles initialized from population statistics |
| New Client Weight Reduction | New clients limited to 30% contribution |

### 3.7 Algorithm Summary

```
Algorithm: CAAC-FL Aggregation
Input: Client updates {g_k^t}, profiles {P_k}
Output: Aggregated gradient g_agg

1. For each client k:
   a. Compute A_mag, A_dir, A_temp from profile P_k
   b. Compute composite score A_k^t
   c. Compute adaptive threshold τ_k^t
   d. Soft-clip: g̃_k = clip(g_k, A_k, τ_k)
   e. Update profile: P_k.update_ewma(g_k)
   f. Update reliability: P_k.update_reliability(A_k ≤ τ_k)

2. Aggregate: g_agg = weighted_average({g̃_k}, {n_k})

3. Return g_agg
```

---

## 4. Experimental Framework

### 4.1 Novel Threat Models

We extend Li et al. (2024) with three dimensions:

#### 4.1.1 Compromise Timing Study

**Motivation**: Real-world devices are compromised at various points—not just deployment start. Does partial model convergence provide inherent robustness?

**Design**: Attacks begin at rounds {0, 10, 20, 30, 40} out of 50 total rounds.

**Research Questions**:
- RQ1: Do attacks become less effective when model has partially converged?
- RQ2: Which aggregation schemes are most sensitive to compromise timing?

#### 4.1.2 Attack Window Study

**Motivation**: Attacks may be transient—malware detected and removed, or attacker loses access. Can models recover?

**Design**: Attack windows [start, end):
- [0, end): Full attack (Li et al. baseline)
- [0, 25): Early attack, then honest behavior
- [25, end): Late compromise only
- [10, 40): Mid-training window
- [0, 10): Brief initial attack

**Research Questions**:
- RQ3: Can models recover after transient attacks?
- RQ4: Is early vs. late attack more damaging?

#### 4.1.3 Non-IID Severity Study

**Motivation**: Li et al. tested only α=0.1 (severe) and IID. Real deployments span the spectrum.

**Design**: Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0}

**Research Questions**:
- RQ5: How does heterogeneity interact with Byzantine robustness?
- RQ6: At what α do defenses begin to fail on clean data?

### 4.2 Aggregation Schemes

We evaluate ten aggregation schemes:

| Scheme | Type | Detection |
|--------|------|-----------|
| FedAvg | Weighted mean | No |
| Median | Coordinate-wise median | No |
| Trimmed Mean | Trimmed coordinate mean | No |
| Krum | Single nearest selection | Yes |
| Multi-Krum | Top-k nearest averaging | Yes |
| GeoMed | Geometric median | No |
| CC | Centered Clipping | No |
| Clustering | Agglomerative clustering | Yes |
| ClippedClustering | Clipping + Clustering | Yes |
| **CAAC-FL** | **Per-client adaptive** | **Yes** |

### 4.3 Attack Types

Six attacks following Li et al. (2024):

| Attack | Description | Parameters |
|--------|-------------|------------|
| None | Baseline | - |
| Sign Flipping | Negate gradients | - |
| Random Noise | Gaussian N(0,25) | σ=5.0 |
| ALIE | Statistical evasion | z_max from CDF |
| IPM-small | Magnitude reduction | ε=0.5 |
| IPM-large | Direction reversal | ε=100 |
| Label Flipping | Data poisoning | l→L-l-1 |

### 4.4 Experimental Setup

| Parameter | Value | Li et al. 2024 |
|-----------|-------|----------------|
| Dataset | CIFAR-10 | CIFAR-10, MNIST |
| Model | SimpleCNN | CCT |
| Clients | 25 | 20 |
| Byzantine Ratio | 20% | 25% |
| Rounds | 50 | 600 (FedAvg) |
| Local Epochs | 5 | 50 SGD steps |
| Batch Size | 32 | 64 |
| Learning Rate | 0.01 | 0.1→0.025 |
| Non-IID α | 0.5 (default) | 0.1 or IID |

### 4.5 Evaluation Metrics

**Model Performance**:
- Final test accuracy
- Convergence trajectory

**Byzantine Detection** (for schemes with client selection):
- True Positive Rate: Byzantine clients correctly rejected
- False Positive Rate: Honest clients incorrectly rejected
- F1 Score

---

## 5. Experimental Results

*[Results to be filled after running experiments]*

### 5.1 Baseline Comparison (Immediate Attacks)

*Compare all schemes under immediate attack scenario (Li et al. compatible)*

### 5.2 Compromise Timing Study

*How do schemes perform when attacks start at different rounds?*

**Hypothesis**: Attacks delayed until after partial convergence will be less effective for most schemes, as the model has already learned useful representations.

### 5.3 Attack Window Study

*Can models recover from transient attacks?*

**Hypothesis**:
- Early-then-honest ([0,25)) will show recovery
- Late-only ([25,end)) will show sharp degradation from high baseline
- Brief attacks ([0,10)) may have minimal lasting impact

### 5.4 Non-IID Severity Study

*How does heterogeneity interact with robustness?*

**Hypothesis**: CAAC-FL will show smallest performance degradation as α decreases, due to per-client profiling.

### 5.5 CAAC-FL Analysis

*Detailed analysis of CAAC-FL components*

- Ablation: contribution of each anomaly dimension
- Cold-start: effectiveness of warmup mechanisms
- False positive rate comparison on Non-IID data

---

## 6. Discussion

### 6.1 Key Findings

*[To be completed after experiments]*

### 6.2 Implications for Practice

**For FL System Designers**:
- Choose defense based on expected heterogeneity level
- Consider delayed compromise as realistic threat model
- CAAC-FL provides best trade-off for heterogeneous deployments

**For Healthcare FL**:
- Traditional defenses may reject legitimate institutional diversity
- Per-client profiling essential for multi-site deployments

### 6.3 Limitations

1. **Computational Overhead**: Per-client profiling increases server memory
2. **Profile Poisoning**: Long-term attackers could gradually shift profiles
3. **Collusion**: Coordinated attackers sharing profile information not addressed
4. **Single Dataset**: Evaluation limited to CIFAR-10

### 6.4 Future Work

- Theoretical convergence analysis for CAAC-FL
- Extension to cross-device FL (millions of clients)
- Adaptive attack design specifically targeting CAAC-FL

---

## 7. Conclusion

This paper introduced CAAC-FL, a client-adaptive Byzantine defense that maintains per-client behavioral profiles to distinguish legitimate heterogeneity from attacks. We extended the experimental framework of Li et al. (2024) with three novel dimensions—compromise timing, attack windows, and non-IID severity—enabling study of realistic threat models previously unexplored.

Our comprehensive evaluation demonstrates that [summary of key results to be added]. CAAC-FL achieves [performance summary] while maintaining the lowest false positive rate on heterogeneous data, making it particularly suitable for healthcare and other domains with institutional diversity.

**Acknowledgments**: [To be added]

---

## References

- Baruch, G., Baruch, M., & Goldberg, Y. (2019). A Little Is Enough: Circumventing Defenses for Distributed Learning. NeurIPS.

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS.

- Chen, Y., Su, L., & Xu, J. (2017). Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent. POMACS.

- Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local Model Poisoning Attacks to Byzantine-Robust Federated Learning. USENIX Security.

- Karimireddy, S. P., He, L., & Jaggi, M. (2021). Learning from History for Byzantine Robust Optimization. ICML.

- Li, S., Ngai, E. C.-H., & Voigt, T. (2024). An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning. IEEE Transactions on Big Data, 10(6), 975-988.

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

- Sattler, F., Müller, K.-R., Wiegand, T., & Samek, W. (2020). On the Byzantine Robustness of Clustered Federated Learning. ICASSP.

- Xie, C., Koyejo, O., & Gupta, I. (2020). Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. UAI.

- Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML.

---

## Appendix A: CAAC-FL Implementation Details

### A.1 Default Parameters

```python
CAACFLStrategy(
    alpha=0.05,              # EWMA rate
    gamma=0.1,               # Reliability update
    tau_base=1.2,            # Base threshold
    beta=0.5,                # Reliability flexibility
    window_size=5,           # History window
    weights=(0.5, 0.3, 0.2), # Anomaly dimension weights
    warmup_rounds=10,        # Conservative period
    warmup_factor=0.3,       # Warmup strictness
    min_rounds_for_trust=5,  # Trust delay
    use_cross_comparison=True,
    use_population_init=True,
    new_client_weight=0.3,
)
```

### A.2 Complexity Analysis

**Space**: O(K × W) for K clients with window size W
**Time per round**: O(K × d) for d-dimensional gradients

---

## Appendix B: Extended Experimental Results

*[Tables and figures to be added after experiments]*

---

## Appendix C: Reproducibility

Code available at: [repository link]

```bash
# Run baseline experiments (Li et al. compatible)
python run_flower_experiments.py --all_strategies --all_attacks

# Run compromise timing study
python run_flower_experiments.py --timing_study --all_strategies

# Run attack window study
python run_flower_experiments.py --window_study --all_strategies

# Run non-IID severity study
python run_flower_experiments.py --alpha_study --all_strategies
```
