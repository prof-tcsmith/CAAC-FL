# Reference Paper Analysis for CAAC-FL Work in Progress

## Main Paper Overview: CAAC-FL

**Title:** Distinguishing Medical Diversity from Byzantine Attacks: Client-Adaptive Anomaly Detection for Healthcare Federated Learning

**Core Innovation:**
- Client-specific behavioral profiling instead of global thresholds
- Three-dimensional anomaly detection: magnitude, directional, temporal
- EWMA-based tracking of gradient norms and consistency patterns

**Experimental Setup:**
- **Datasets:** MIMIC-III (ICU mortality prediction, n=49,785), ChestX-ray8 (multi-label disease, 108,948 images), ISIC 2019 (melanoma, n=2,750)
- **Federation:** 20 clients with Dirichlet allocation (α=0.5) for label skew, power law dataset sizes
- **Attacks:** ALIE, Inner Product Manipulation, slow-drift poisoning, profile-aware adaptive attacks
- **Byzantine fractions:** 20-40%
- **Baselines:** FedAvg, Krum, Trimmed Mean, ARC, FLTrust, LASA
- **Metrics:** Model accuracy, false positive rate, attack impact, detection latency

**Research Hypotheses:**
- H1: Client-specific profiles reduce false positives in heterogeneous settings
- H2: Multi-dimensional defense outperforms single-metric approaches
- H3: Window-based profiling distinguishes abrupt attacks from gradual legitimate changes

---

## Reference Paper Analyses

### 1. Yin et al. (2018) - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates

**Key Contributions:**
- Coordinate-wise median and trimmed mean aggregation algorithms
- Theoretical optimality proofs for Byzantine-robust learning
- Two-phase experiments: gradient descent and one-round algorithms

**Attack Models:**
- Label flipping attack (y → 9-y systematic transformation)
- Random label noise (uniform random labels)
- Byzantine fraction: α=0.05 to α=0.1

**Experimental Settings:**
- Dataset: MNIST (60,000 samples)
- Models: Multi-class logistic regression, CNN
- Workers: m=10 to m=40 machines
- Data partition: Equal-sized IID partitions

**Evaluation Metrics:**
- Test accuracy and error rates
- Convergence speed (test error vs. communication rounds)
- Parameter error ||w^T - w*||_2
- Communication efficiency

**Repurposable Design Elements for CAAC-FL:**

1. **Label-flipping attack strategy** - Can adapt for healthcare:
   - MIMIC-III: flip alive↔dead labels
   - ChestX-ray8: systematic disease label flipping
   - ISIC 2019: benign↔malignant flipping

2. **Multi-fraction Byzantine testing** - Current CAAC-FL plans α=0.2-0.4, could add α ∈ {0.1, 0.15, 0.25, 0.3, 0.35} for finer granularity

3. **Convergence vs. communication rounds analysis** - Critical for federated learning, track rounds-to-accuracy metrics

4. **Multiple model complexity levels** - Test varying architectures per dataset:
   - MIMIC-III: Logistic regression, MLP, LSTM
   - ChestX-ray8: ResNet variants
   - ISIC 2019: EfficientNet/DenseNet variants

5. **Experimental comparison structure**: No Byzantine (baseline) → Vulnerable (standard aggregation) → Robust methods

6. **Stochastic vs. full-batch settings** - Test mini-batch local training (20%, 50%, 100% of local data)

7. **Statistical gradient characterization** - Report variance, skewness, tail behavior of gradients on healthcare datasets

**CAAC-FL Advantages:**
- Realistic non-IID settings (Dirichlet allocation vs. IID)
- Real healthcare datasets vs. MNIST
- Sophisticated attacks (ALIE, IPM, slow-drift) vs. simple label corruption
- Client-adaptive profiling vs. global thresholds
- Modern baselines (Krum, ARC, FLTrust, LASA)

---

### 2. Blanchard et al. (2017) - Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent

**Key Contributions:**
- Krum algorithm: First provably Byzantine-resilient aggregation for distributed SGD
- Selection based on sum of squared distances to n-f-2 closest neighbors
- Multi-Krum variant: Average top-m scoring updates (m=n-f)
- Byzantine tolerance: f < (n-2)/2 ≈ 50% asymptotically

**Attack Models:**
- **Omniscient Attack:** Byzantine workers compute accurate gradient, send opposite direction -λ·∇Q(xt)
- **Gaussian Attack:** Random vectors from N(0, σ²I) with σ=200
- Byzantine fractions: 33%, 45%

**Experimental Settings:**
- Datasets: Spambase, MNIST
- Models: MLP (2 hidden layers), ConvNet
- Workers: n=20, Byzantine: 0%, 33%, 45%
- Mini-batch sizes: 1, 3, 10, 20, 30, 40, 80, 120, 160
- Training: 500 rounds

**Evaluation Metrics:**
- Cross-validation error over rounds
- Convergence speed (rounds to target accuracy)
- Cost of resilience (performance gap at 0% Byzantine)
- Batch size sensitivity (error vs. batch size)

**Repurposable Design Elements for CAAC-FL:**

1. **Krum as strong baseline** - Include original Krum and Multi-Krum (m=n-f) in comparisons

2. **Omniscient attack model** - Implement as baseline attack:
   - Byzantine clients compute accurate gradient, send opposite direction
   - Tests worst-case informed adversary

3. **Batch size sensitivity analysis** - Critical finding: larger batches reduce cost of resilience
   - CAAC-FL application: Analyze performance across client dataset sizes (power-law distribution)
   - Expected: Behavioral profiling more valuable for small-dataset clients

4. **Multi-Krum concept** - Adaptively select top-m trusted clients based on behavioral scores
   - Compare: Static Multi-Krum vs. CAAC-FL adaptive selection

5. **Byzantine fraction evaluation** - Tests 0%, 33%, 45%; CAAC-FL plans 20%, 40%
   - Add 30% Byzantine for finer granularity and alignment with Blanchard

6. **Cost of resilience metric** - Performance under 0% Byzantine
   - Show CAAC-FL gracefully degrades to near-FedAvg when few Byzantine clients detected

7. **Convergence visualization** - Time-series plots: error vs. rounds
   - Side-by-side: 0% vs. high Byzantine fraction
   - Multiple curves for different methods

**Key Experiments for CAAC-FL:**

1. **Heterogeneity robustness** - Vary Dirichlet α ∈ {0.1, 0.5, 1.0, ∞ (IID)}
   - Hypothesis: Krum degrades with heterogeneity; CAAC-FL adapts

2. **Adaptive attack detection** - Slow-drift attack with gradual intensification
   - Track behavioral trust scores over time
   - Compare detection latency: CAAC-FL vs. Krum

3. **Behavioral transparency** - Visualize trust score evolution (heatmap: clients × rounds)
   - Validate: CAAC-FL correctly identifies Byzantine clients earlier

4. **Multi-dataset validation** - MIMIC-III (tabular), ChestX-ray8 (images), ISIC 2019 (images)
   - Show generalization across data modalities

**CAAC-FL Advantages over Krum:**
- Adapts to non-IID data (Dirichlet allocation) vs. IID assumption
- Detects adaptive attacks (slow-drift) vs. static attacks only
- Provides interpretable trust scores vs. opaque selection
- Lower cost of resilience under no attack (adaptively relaxes to FedAvg)

**Computational Complexity:**
- Krum: O(n²·d) - quadratic in workers, linear in dimension
- CAAC-FL: Similar complexity with profiling overhead (must justify benefit)

---

### 3. Baruch et al. (2019) - A Little Is Enough: Circumventing Defenses For Distributed Learning

**Key Contributions:**
- ALIE (A Little Is Enough) attack: Byzantine attack operating within population variance
- Circumvents major defenses (Krum, Trimmed Mean, Bulyan) by staying within statistical bounds
- Non-omniscient attack: only requires access to corrupted workers' data
- Backdoor attack variant maintaining benign accuracy while achieving >99% backdoor success

**ALIE Attack Mechanism:**
- Calculate z_max using cumulative normal distribution
- Position malicious gradients between mean and "supporters"
- For n workers, m Byzantine: s = ⌊n/2 + 1⌋ - m supporters needed
- Malicious gradient: μ - z_max·σ (coordinate-wise)

**Attack Models:**
- **Convergence Prevention**: Obstruct model from reaching good accuracy
- **Backdoor**: Manipulate model for specific inputs while maintaining benign accuracy
  - Pattern: Upper-left 5×5 pixels at max intensity
  - Loss: α·ℓ_backdoor + (1-α)·ℓ_Δ (α=0.2)

**Experimental Settings:**
- Datasets: MNIST, CIFAR-10, CIFAR-100
- Models: Simple architectures, WideResNet (CIFAR-100)
- Workers: n=51, m=12 (≈24% Byzantine)
- Attack parameters: z=1.5 (convergence), z=0.2 (backdoor)

**Evaluation Metrics:**
- Maximum test accuracy achieved
- Benign accuracy vs. backdoor success rate
- Gradient sign-flipping analysis: fraction where |μj| < z·σj
- Variance analysis over training rounds

**Repurposable Design Elements for CAAC-FL:**

1. **ALIE implementation formula** - Direct adaptation:
   - CAAC-FL with n=20, m=4 (20%): s=7, z_max≈0.76
   - CAAC-FL with n=20, m=8 (40%): s=3, z_max≈1.15

2. **Variance analysis methodology** - Track fraction of parameters where gradients can flip:
   - Evaluate for z ∈ {0.5, 1.0, 2.0}
   - Expected: Higher variance on medical data with Dirichlet α=0.5

3. **Non-IID adaptation** - Modify z_max for non-IID distribution:
   - Healthcare data likely has HIGHER variance than IID
   - ALIE potentially MORE effective in CAAC-FL setting

4. **Power-law dataset consideration** - Stratify analysis by client dataset size:
   - Small clients may have higher gradient variance
   - Distinguish variance from small data vs. malicious manipulation

5. **Medical backdoor patterns** - Healthcare-specific triggers:
   - MIMIC-III: Specific vital sign combinations
   - ChestX-ray8: Medical device artifacts → misdiagnosis
   - ISIC 2019: Ruler markers, hair patterns → wrong classification

6. **Behavioral profiling advantage** - CAAC-FL hypothesis:
   - ALIE produces consistent patterns across rounds
   - Behavioral profiling may detect temporal patterns statistical methods miss
   - Test: Static ALIE vs. adaptive ALIE (varying z_max per round)

**Key Experiments for CAAC-FL:**

1. **ALIE as primary attack** - Demonstrate ALIE circumvents statistical baselines (Krum, Trimmed Mean)

2. **CAAC-FL detection capability** - Show behavioral profiling detects ALIE's consistent manipulation

3. **Variance characterization** - Analyze gradient variance on medical datasets:
   - Compare IID vs. Dirichlet α=0.5
   - Stratify by client dataset size (power-law)

4. **Medical backdoor experiments**:
   - Design domain-appropriate backdoor patterns
   - Measure stealthiness and success rates
   - Evaluate CAAC-FL's backdoor detection

5. **Combined attack scenarios** - Multiple Byzantine attack types simultaneously:
   - Some use ALIE, others use IPM or slow-drift
   - Test CAAC-FL's robustness to heterogeneous attacks

**CAAC-FL Positioning:**
- Statistical defenses vulnerable to variance-based attacks
- Behavioral profiling adds temporal and client-specific detection
- Narrative: "ALIE circumvents statistical defenses by operating within variance bounds, but CAAC-FL's behavioral profiling tracks temporal patterns revealing coordinated Byzantine behavior"

**Implementation Priority:** High - ALIE is explicitly mentioned in CAAC-FL paper's attack set

---

### 4. Cao et al. (2021) - FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping

**Key Contributions:**
- Trust bootstrapping using small, clean server-side root dataset (100 examples)
- ReLU-clipped cosine similarity for trust scores: TS_i = ReLU(cos(g_i, g_0))
- Magnitude normalization: Projects all updates to same hyper-sphere
- Weighted aggregation: g = (1/Σ_j TS_j) · Σ_i (TS_i · ḡ_i)

**Attack Models:**
- **Label Flipping (LF)**: Data poisoning, flip labels l → M-l-1
- **Krum Attack**: Untargeted model poisoning optimized against Krum
- **Trim Attack**: Untargeted model poisoning optimized against Trimmed Mean
- **Scaling Attack**: Targeted backdoor with trigger patterns, scaled by λ=n
- **Adaptive Attack**: Zeroth-order gradient ascent optimized against FLTrust
  - Parameters: σ²=0.5, γ=0.005, η=0.01, V=Q=10

**Experimental Settings:**
- Datasets: MNIST-0.1 (IID), MNIST-0.5, Fashion-MNIST, CIFAR-10, HAR (30 real users), CH-MNIST (medical)
- Models: CNN, ResNet20, Logistic Regression
- Clients: 100 (30 for HAR, 40 for CH-MNIST)
- Byzantine fractions: 0%, 10%, 20%, 40%, 60%, 80%, 90%, 95%
- Root dataset: 100 examples (tested 50-500)
- Non-IID distribution: q parameter (0.1=IID to 0.5=highly non-IID)

**Evaluation Metrics:**
- Testing error rate (untargeted attacks)
- Attack success rate (targeted/backdoor attacks)
- Convergence speed (training error vs. iterations)
- Fidelity (accuracy under no attacks)
- Robustness (accuracy under attacks vs. FedAvg baseline)

**Repurposable Design Elements for CAAC-FL:**

1. **Root dataset concept** - CAAC-FL could collect 100 verified medical examples:
   - MIMIC-III: Expert-annotated ICU cases
   - ChestX-ray8: Radiologist-verified images
   - ISIC 2019: Dermatologist-confirmed diagnoses
   - **Privacy tradeoff**: Centralized data vs. distributed behavioral profiling

2. **Adaptive attack framework** - Apply to CAAC-FL:
   - Zeroth-order optimization with CAAC-FL's aggregation rule
   - Use FLTrust's parameters as starting point
   - Test if behavioral profiling resists optimization-based attacks

3. **Cosine similarity for directional analysis** - Can incorporate into CAAC-FL:
   - Already mentioned in CAAC-FL's directional anomaly detection
   - ReLU clipping prevents negative contributions

4. **Variable malicious fractions** - Extend CAAC-FL testing:
   - Current: 20-40%
   - Add: 60%+ to show robustness limits
   - Compare breakdown points across methods

5. **Targeted vs. untargeted attacks** - CAAC-FL should test both:
   - Untargeted: ALIE, Krum attack, Trim attack
   - Targeted: Scaling/backdoor with medical triggers
   - Slow-drift could be compared to Trim attack

6. **Medical dataset (CH-MNIST)** - Similar characteristics:
   - Small dataset, fewer clients (40 vs. CAAC-FL's 20)
   - Medical imaging domain
   - ResNet20 architecture applicable

7. **Ablation studies** - Test CAAC-FL components:
   - Behavioral profiling alone vs. with trust scores
   - With/without magnitude normalization
   - With/without temporal weighting

8. **Parameter sensitivity analysis**:
   - Root dataset size: 50-500 examples
   - CAAC-FL: Profile window size, anomaly thresholds, EWMA decay

9. **Scalability testing** - Clients: 50, 100, 200, 300, 400
   - CAAC-FL: Test beyond 20 clients for real deployment scenarios

10. **Evaluation table format** - Attacks × Methods matrix:
    - Clear, comprehensive comparison
    - Testing error / attack success rate

**CAAC-FL Advantages over FLTrust:**

1. **No centralized root dataset** - Privacy-preserving distributed approach
2. **Client-adaptive** - Personalized profiles vs. one-size-fits-all trust scores
3. **Temporal detection** - Tracks behavioral patterns over time for slow-drift
4. **Power-law data distribution** - More realistic than FLTrust's uniform distribution

**Critical for CAAC-FL:**

1. **Implement FLTrust exactly as baseline** - Fair comparison requires identical setup
2. **Privacy analysis** - Quantify privacy tradeoffs: root dataset vs. behavioral profiling
3. **Adaptive attack testing** - Must test optimization-based attacks against CAAC-FL
4. **Medical backdoor triggers** - Design healthcare-appropriate patterns for Scaling attack
5. **Robustness limits** - Test at 60%+ Byzantine to show where CAAC-FL breaks down

**Key Questions:**
- Can CAAC-FL achieve FLTrust-level robustness without centralized data?
- How does behavioral profiling cost compare to trust score computation?
- Does temporal profiling detect adaptive attacks faster than static trust scores?

---

### 5. Pillutla et al. (2022) - Robust Aggregation for Federated Learning (RFA)

**Key Contributions:**
- Geometric median (GM) aggregation instead of arithmetic mean
- Smoothed Weiszfeld algorithm with adaptive weighting: β_i ∝ 1/(ν ∨ ||v - w_i||)
- Converges in R=3 iterations empirically
- Breakdown point: 1/2 (optimal) - tolerates up to 50% corrupted devices

**Attack Models:**
- Non-adversarial corruption (hardware failures)
- Static data poisoning (fixed D_i → D̃_i)
- Adaptive data poisoning (depends on current model)
- Update poisoning: Gaussian noise, omniscient (negative aggregate)
- Corruption levels: ρ ∈ {0, 0.01, 0.1, 0.25}

**Experimental Settings:**
- Datasets: EMNIST, Shakespeare, Sent140
- Models: Linear, CNN, LSTM
- Devices: 628-1000 (naturally heterogeneous)
- Weiszfeld iterations: R=3
- Smoothing: ν=10^-6

**Evaluation Metrics:**
- Test accuracy
- Communication cost (calls to secure average oracle)
- Convergence stability (variance across 5 seeds)
- Heterogeneity metrics: Ω_X, Ω_{Y|X}, Ω overall

**Repurposable for CAAC-FL:**

1. **Adaptive weighting mechanism** - Global rule for comparison:
   - RFA: β_i ∝ 1/||v - w_i|| (global distance-based)
   - CAAC-FL: Client-specific thresholds (behavioral)
   - Research question: When does client-specific outperform global?

2. **Communication efficiency analysis** - Plot accuracy vs. oracle calls:
   - RFA: 3× overhead for iterative GM
   - One-step RFA: No overhead
   - CAAC-FL: Profile computation may be local (no overhead)

3. **Corruption protocol** - Diverse attack types:
   - Label flipping (Sent140)
   - Image negation (EMNIST)
   - Text reversal (Shakespeare)
   - Gaussian noise (σ² = variance of clean update)

4. **Heterogeneity-robustness tradeoff** - RFA shows 1-3% accuracy drop at ρ=0
   - Personalization mitigates (10.1% → 3.4% degradation)
   - CAAC-FL hypothesis: Client-adaptive better preserves heterogeneous updates

5. **Baseline suite** - Compare against coordinate-wise median, norm clipping, trimmed mean, Multi-Krum

**Key Finding:** Cannot have privacy + communication efficiency + robustness simultaneously
- RFA chooses privacy + robustness = 3× communication

---

### 6. Werner et al. (2023) - Provably Personalized and Robust Federated Learning

**Key Contributions:**
- First theoretical proof that client-specific adaptation enhances Byzantine robustness
- Gradient-based clustering (not loss-based)
- Threshold-Clustering algorithm with conservative radius τ ≈ √σΔ
- Near-optimal convergence: O(1/√niT) with smooth non-convex losses

**Theoretical Framework:**
- Intra-cluster similarity: ||∇fi(x) - ∇f̄i(x)||² ≤ A²||∇f̄i(x)||²
- Inter-cluster separation: ||∇fi(x) - ∇fj(x)||² ≥ Δ² - D²||∇fi(x)||²
- Byzantine robustness: Error scales as βiσΔ (bounded, not catastrophic)
- Near-optimality: Clustering error within σ/Δ factor of optimal

**Attack Models:**
- Label-flipping
- Large gradient attack
- Bit-flipping attack
- 50 Byzantine workers per cluster (50% tested)

**Experimental Settings:**
- Datasets: Synthetic (linear regression), MNIST (rotation + private label), CIFAR-10/100
- K=4 or 10 clusters, 50-75 clients per cluster
- Models: Linear, CNN, VGG-8/16
- Adaptive radius: 20th percentile of distances to cluster center

**Evaluation Metrics:**
- Test accuracy, test loss
- Clustering accuracy (vs. ground truth)
- Convergence rate theoretical bounds
- Byzantine robustness (performance under attacks)

**Direct Relevance to CAAC-FL:**

1. **Theoretical validation** - Werner et al. prove:
   - Client-specific models achieve near-optimal convergence
   - Personalization provides Byzantine robustness
   - Gradient-based clustering superior to loss-based
   - CAAC-FL cites this as theoretical foundation

2. **Gradient-based clustering** - CAAC-FL's behavioral profiling justified:
   - Werner: Cluster on gradient similarity
   - CAAC-FL: Profile gradient patterns over time
   - Both avoid loss-based methods (initialization sensitivity)

3. **Adaptive radius selection** - Percentile-based threshold:
   - Werner: 20th percentile
   - CAAC-FL: Can use similar approach for anomaly thresholds

4. **Assumptions to validate** - Empirically verify on medical data:
   - Intra-cluster gradient variance (A² parameter)
   - Inter-cluster separation (Δ parameter)
   - Plot over training (Figure 2 style)

5. **Baselines** - Local, Global (FedAvg), IFCA, HypCluster, Ditto, KNN-personalization, Ground Truth

**CAAC-FL as Practical Implementation:**
- Werner: Theory for K known clusters
- CAAC-FL: Extends to unknown K with hierarchical clustering
- Werner: Gradient similarity
- CAAC-FL: Behavioral profiles (temporal gradient patterns)
- Werner: Centralized server
- CAAC-FL: Could extend to decentralized (blockchain)

**Citation for CAAC-FL:**
> "Recent theoretical work [Werner et al., 2023] proves that gradient-based client clustering achieves near-optimal convergence rates while providing Byzantine robustness. CAAC-FL provides a practical implementation through client-adaptive behavioral profiling."

---

### 7. Xu et al. (2024) - LASA: Layer-Adaptive Sparsified Model Aggregation

**Key Contributions:**
- Layer-wise adaptive aggregation (L layers independently)
- Two-stage: (1) Top-k sparsification per client, (2) Layer-wise filtering
- Positive Direction Purity (PDP) metric: ρ = (1/2) × (1 + Σ sgn([x]ᵢ) / Σ |sgn([x]ᵢ)|)
- Median-based Z-score filtering: λᵢ = (xᵢ - Med(X)) / σ
- Uniform radius parameters (λₘ, λ_d) across layers

**Attack Models (8 total):**
- Naive: Random, Noise, Sign-Flip
- SOTA: Min-Max, Min-Sum, TailoredTrMean, Lie, ByzMean
- Attack ratio: 25% default (tested 5%-30%)

**Experimental Settings:**
- Datasets: MNIST, FMNIST, FEMNIST, CIFAR-10/100, Shakespeare
- Models: CNN, ResNet-18, RNN
- Clients: 100-6,000
- Non-IID: Dirichlet Dir(α=0.5), natural (FEMNIST, Shakespeare)
- Sparsification level: 0.3 default (tested 0.1-0.99)

**Evaluation Metrics:**
- Test accuracy (best over training)
- True Positive Rate (TPR) - malicious detection
- False Positive Rate (FPR) - benign misclassification
- κ-Robustness: E||x̂ - x̄_B||² ≤ κ = O(c_k(1 + f/(n-2f)))

**Repurposable for CAAC-FL:**

1. **Layer-wise analysis framework** - Apply to attention per layer:
   - Independent layer metrics (magnitude + direction)
   - MZ-score normalization for cross-layer comparison
   - CAAC-FL: Layer-specific behavioral profiles

2. **Sparsification protocol** - Top-k parameter selection:
   - Individual client sparsification (not uniform)
   - Optimal SL = 0.1-0.3
   - CAAC-FL: Attention weight pruning

3. **Multi-attack evaluation suite** - Comprehensive baseline:
   - 3 naive + 5 SOTA = 8 attack types
   - CAAC-FL must test against same suite

4. **Direction metrics** - PDP for stealthy attack detection:
   - Sign-based, robust to magnitude scaling
   - Enhanced by sparsification
   - CAAC-FL: Direction analysis of attention weights

5. **Ablation study design** - Component isolation:
   - Sparsification only, magnitude only, direction only, pairs, full
   - CAAC-FL: Isolate attention clustering, adaptive weighting, sparsification

6. **Hyperparameter sweeps**:
   - Sparsification: {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}
   - Filtering radius: λₘ, λ_d ∈ {1.0, 1.5, 2.0, 3.0, 4.0}
   - CAAC-FL: Cluster count, attention temperature, aggregation weights

**Key Findings:**
- Layer-level > coordinate-wise > model-wise granularity
- Sparsification more beneficial in non-IID
- Direction + magnitude complementary (robust to diverse attacks)
- Multi-metric filtering superior to single metric

**LASA Baselines:** FedAvg, TrMean, GeoMed, Multi-Krum, Bulyan, DnC, SignGuard, SparseFed

**Complexity:** O(nd log d) - comparable to Krum, TrMean

---

### 8. Li et al. (2024) - An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning

**Key Contributions:**
- Comprehensive experimental study of 8 Byzantine-robust aggregation schemes
- **Critical finding cited by CAAC-FL:** "Less than 10% accuracy in non-IID settings even without attacks"
- Identifies Non-IID as the fundamental challenge for Byzantine defenses
- Proposes ClippedClustering (best performer) with adaptive clipping threshold

**Attack Models (5 total):**
- **ALIE:** Exploits high variance (devastating on CIFAR-10, ineffective on MNIST)
- **IPM (ε=0.5):** Stealthy, doesn't reverse gradient but circumvents defenses
- **IPM (ε=100):** Aggressive, reverses gradient direction
- **Sign Flipping (SF):** Simple gradient reversal
- **Label Flipping (LF):** Data poisoning

**Experimental Settings:**
- Datasets: CIFAR-10 (CCT model), MNIST (2-layer perceptron)
- Clients: K=20, Byzantine: M=5 (25% default)
- Data distribution: IID vs. Non-IID (Dirichlet Dir(α=0.1))
- FL algorithms: FedSGD (6000 rounds), FedAvg (600 rounds, El=50)
- Batch sizes: 64 (CIFAR-10), 128 (MNIST)

**Aggregation Schemes Compared (8):**
1. **Krum:** 10% on Non-IID CIFAR-10 (failure)
2. **GeoMed:** 10% on Non-IID CIFAR-10 (failure)
3. **AutoGM:** 10% on Non-IID CIFAR-10 (failure)
4. **Median:** 10% on Non-IID CIFAR-10 (failure)
5. **TrimmedMean:** 80.98% (survives Non-IID)
6. **CC (Centered Clipping):** 80.01%
7. **Clustering:** 81.80%
8. **ClippedClustering:** 81.58% (best overall)

**Key Findings:**

1. **Non-IID performance degradation:**
   - Distance-based schemes (Krum, GeoMed, AutoGM): FAIL (10%)
   - Coordinate-wise schemes (Median): FAIL (10%)
   - Mean-based with mechanisms: SURVIVE (~80%)

2. **Pairwise cosine similarity analysis:**
   - FedAvg + IID: Highest similarity → best for clustering
   - FedSGD + Non-IID: Low similarity → hard to distinguish benign from malicious
   - Explains differential robustness across settings

3. **Attack effectiveness:**
   - ALIE: Dataset-dependent (high variance = more effective)
   - IPM (ε=0.5): Stealthy, circumvents most defenses
   - FedAvg more robust than FedSGD (lower gradient variance)

4. **Batch size impact:**
   - Larger batch → lower variance → better robustness
   - Trade-off: Computation vs. robustness

**Repurposable for CAAC-FL:**

1. **Non-IID partitioning:** Dirichlet Dir(α=0.1) for strong heterogeneity

2. **Baseline suite:** Compare against all 8 schemes (especially ClippedClustering)

3. **Attack suite:** IPM (ε=0.5 and ε=100), ALIE, LF, SF

4. **Pairwise cosine similarity analysis:** Explain why CAAC-FL works better

5. **Performance targets:**
   - Minimum: >80% on Non-IID without attacks
   - Goal: Maintain >70% under attacks (vs. ~10% for distance-based)

6. **Evaluation protocol:**
   - Test IID vs. Non-IID
   - Test FedSGD vs. FedAvg
   - Vary Byzantine fraction: 5-45%
   - Vary batch size

**CAAC-FL Motivation:**
> "Based on our experimental study, we conclude that the robustness of all the aggregation schemes is limited, highlighting the need for new defense strategies, in particular for Non-IID datasets." - Li et al.

**Critical Gap:** Existing defenses fail to distinguish legitimate heterogeneity from Byzantine behavior → CAAC-FL's client-adaptive profiling addresses this.

---

### 9. Le and Moothedath (2025) - Byzantine Resilient Federated Multi-Task Representation Learning

**Key Contributions:**
- BR-MTRL: Multi-task learning with shared representation (φ) + client-specific heads (h_i)
- Asymmetric update frequency: τ_h=10 epochs (heads), τ_φ=1 epoch (representation)
- Geometric Median and Krum aggregation for robust shared model learning
- Meta-test transfer learning to new clients

**Architecture:**
- **Shared global representation φ:** R^d → R^k (k ≪ d) - Common feature extractor
- **Client-specific local heads h_i:** R^k → Y - Personalized final layer
- **Privacy:** Local heads never leave device, only φ communicated
- **Model:** AlexNet-based (5 conv blocks + linear)

**Attack Model:**
- Gaussian noise injection: φ_i + σ·N(0,I)
- White-box (full knowledge)
- Byzantine proportions: 20% (CIFAR-10), 33% (FEMNIST)

**Experimental Settings:**
- Datasets: CIFAR-10 (60K, 10 classes), FEMNIST (145K, 62 classes)
- Clients: 100-1000 (CIFAR-10), 150 (FEMNIST)
- Non-IID: 2-5 classes per client (CIFAR-10), natural writer-based (FEMNIST)
- Training: 100-200 rounds, α=0.2 participation
- Optimizer: SGD with momentum (β=0.9), η=0.01

**Evaluation Metrics:**
- Average test accuracy
- Meta-test transferability (10 new clients, 10 epochs fine-tuning)
- Computational overhead (AWS testbed)

**Results:**
- BR-MTRL+GM: 72.94% (CIFAR-10, 20% Byzantine) vs. Byzantine FedPer: 52.72%
- BR-MTRL+Krum: 80.25% (best with 100 clients)
- Meta-test: 80% accuracy on new clients vs. 60% for Naive

**Repurposable for CAAC-FL:**

1. **Multi-task architecture** - Adopt shared (φ) + personalized (h_i) structure:
   - CAAC-FL: φ for behavioral consistency tracking, h_i for task-specific performance
   - Enables client-specific adaptation while maintaining global knowledge

2. **Alternating optimization** - Asymmetric update frequency (10:1):
   - Frequent: Client-specific models and local behavioral tracking
   - Infrequent: Global behavioral consistency model
   - Reduces Byzantine exposure on shared model

3. **Geometric Median aggregation** - Parameter-free baseline:
   - Apply to aggregate behavioral embeddings
   - Combine with CAAC-FL's temporal tracking

4. **AWS federated testbed** - Infrastructure:
   - EC2 t2.large instances per client
   - S3 cloud storage for model sharing
   - Flask HTTP communication
   - CAAC-FL: Add behavioral database (DynamoDB)

5. **Meta-test protocol** - Transfer learning evaluation:
   - Fix φ, train only h_i for new clients
   - CAAC-FL: Transfer learned behavioral baseline to new clients
   - Zero-shot Byzantine detection

6. **Non-IID data partitioning** - Class-based:
   - 2, 3, 5 classes per client
   - Natural partitioning (FEMNIST writers)

**Gaps CAAC-FL Addresses:**

1. **Temporal behavioral tracking** - BR-MTRL lacks:
   - Static representation φ (no behavioral history)
   - Single-snapshot detection (spatial outlier only)
   - Cannot identify behavioral drift over time
   - CAAC-FL: Multi-round pattern analysis, consistency scoring

2. **Proactive Byzantine identification** - BR-MTRL only mitigates:
   - Reactive robust aggregation (GM/Krum)
   - Byzantine clients continue participating
   - CAAC-FL: Early detection and exclusion

3. **Attack model diversity** - BR-MTRL tests only:
   - Gaussian noise injection
   - Static attack strategy
   - CAAC-FL: Multiple attack types, adaptive adversaries

4. **Behavioral consistency metrics** - BR-MTRL limited:
   - Accuracy-only evaluation
   - No consistency measures across rounds
   - CAAC-FL: Multi-dimensional consistency (magnitude, direction, temporal)

5. **Fine-grained aggregation control:**
   - BR-MTRL: Global GM/Krum (uniform)
   - CAAC-FL: Client-specific weights based on behavioral history

**CAAC-FL Positioning:**
> "Byzantine-resilient multi-task learning [Le & Moothedath 2025] explores client-specific model layers but lacks temporal behavioral tracking. CAAC-FL addresses this gap by integrating temporal consistency analysis with spatial robustness through client-adaptive behavioral profiling."

**Architectural Synergy:**
- BR-MTRL provides foundation: Shared + personalized architecture
- CAAC-FL extends: Temporal behavioral profiling on top of spatial defenses
- Combine: GM/Krum (spatial) + behavioral tracking (temporal) = comprehensive defense

---


---

## Comprehensive Summary and Experimental Protocol Recommendations for CAAC-FL

### Overview

This document analyzed 9 reference papers cited in the CAAC-FL work in progress paper. The analysis focused on experimental protocols, attack models, datasets, evaluation metrics, and repurposable design elements to inform CAAC-FL's implementation.

### Key Themes Across Papers

#### 1. The Non-IID Challenge

**Problem Statement:**
- Li et al. (2024): Existing defenses achieve <10% accuracy on Non-IID data **even without attacks**
- Distance-based schemes (Krum, GeoMed) fail completely
- Coordinate-wise schemes (Median, TrimmedMean) also fail
- Only mean-based with adaptive mechanisms survive (~80%)

**CAAC-FL's Unique Position:**
- Client-adaptive behavioral profiling specifically designed to distinguish:
  - Legitimate heterogeneity (data distribution differences)
  - Malicious behavior (Byzantine attacks)
- Temporal tracking adds dimension other methods lack

#### 2. Attack Landscape Evolution

**Historical Progression:**
1. **Simple attacks** (Yin et al. 2018): Label flipping, random noise
2. **Geometric attacks** (Blanchard et al. 2017): Omniscient attack
3. **Variance-based attacks** (Baruch et al. 2019): ALIE (operates within σ)
4. **Optimization-based attacks** (Cao et al. 2021): Adaptive to specific defenses
5. **Multi-attack scenarios** (Xu et al. 2024): 8 diverse attack types

**CAAC-FL Coverage Required:**
- Untargeted: ALIE, IPM, Noise, Sign-flip
- Targeted: Backdoor with medical triggers
- Adaptive: Slow-drift poisoning, profile-aware attacks
- Data-level: Label flipping

#### 3. Theoretical Foundations

**Werner et al. (2023) Provides:**
- First proof that client-specific adaptation enhances Byzantine robustness
- Gradient-based clustering achieves near-optimal convergence: O(1/√niT)
- Byzantine error bounded: βiσΔ (graceful degradation)
- Validation of personalization for security

**CAAC-FL Citation:**
> "Recent theoretical work [Werner et al., 2023] proves gradient-based client clustering achieves near-optimal convergence while providing Byzantine robustness. CAAC-FL provides a practical implementation through client-adaptive behavioral profiling."

#### 4. Aggregation Mechanism Spectrum

**Spatial Defenses (Single-Round):**
- **Krum** (Blanchard): Distance-based selection
- **Trimmed Mean** (Yin): Coordinate-wise filtering
- **Geometric Median** (Pillutla/Le): Robust averaging
- **FLTrust** (Cao): Trust bootstrapping with root dataset

**Limitations:**
- Fail on Non-IID data (Li et al.)
- Vulnerable to ALIE (Baruch et al.)
- Cannot handle adaptive attacks
- No temporal consistency

**Temporal Enhancements:**
- **LASA** (Xu): Layer-wise adaptive filtering
- **BR-MTRL** (Le): Multi-task personalization
- **Werner et al.**: Gradient-based clustering

**CAAC-FL Innovation:**
- Combines spatial (robust aggregation) + temporal (behavioral profiling)
- Client-adaptive thresholds vs. global rules
- Multi-dimensional anomaly detection (magnitude, direction, temporal)

---

### Consolidated Experimental Protocol for CAAC-FL

Based on analysis of all 9 papers, here is a comprehensive experimental design:

#### Datasets

**Primary (from CAAC-FL paper):**
1. **MIMIC-III:** ICU mortality prediction (n=49,785)
2. **ChestX-ray8:** Multi-label disease classification (108,948 images)
3. **ISIC 2019:** Melanoma detection (n=2,750)

**Benchmark Validation (from references):**
4. **CIFAR-10:** Standard FL benchmark (enables comparison to Li et al., Xu et al.)
5. **FEMNIST:** Natural non-IID (enables comparison to Le et al.)

#### Federation Setup

**Clients:**
- N = 20 (CAAC-FL planned)
- Validate scalability: Test N ∈ {50, 100} based on FLTrust, LASA

**Data Distribution:**
- **Dirichlet allocation:** α=0.5 (CAAC-FL), α=0.1 (high heterogeneity, Li et al.)
- **Power-law dataset sizes:** (CAAC-FL unique contribution)
- **IID baseline:** For comparison with Non-IID

**Byzantine Fractions:**
- Core: 20%, 40% (CAAC-FL planned)
- Extended: 10%, 30%, 60% (based on FLTrust, Li et al. sensitivity)

#### Attack Models

**Untargeted:**
1. **ALIE** (Baruch et al.): z_max calculated per N, m
2. **IPM** (Li et al.): ε=0.5 (stealthy), ε=100 (aggressive)
3. **Noise:** Gaussian N(0, σ²I)
4. **Sign-flip:** Simple gradient reversal
5. **Label-flipping** (Yin et al.): Systematic l → M-l-1

**Targeted:**
6. **Scaling/Backdoor** (Cao et al.): Medical triggers
   - MIMIC-III: Specific vital sign patterns
   - ChestX-ray8: Device artifacts
   - ISIC 2019: Ruler markers, hair patterns

**Adaptive:**
7. **Slow-drift poisoning** (CAAC-FL): Gradual intensification
8. **Profile-aware adaptive** (CAAC-FL): Optimized against behavioral profiling

#### Baseline Comparisons

**Non-Robust:**
- FedAvg (no defense)

**Statistical Defenses:**
- Krum (Blanchard et al.)
- Trimmed Mean (Yin et al.)
- Median (Yin et al.)

**Adaptive Defenses:**
- FLTrust (Cao et al.) - Requires root dataset
- RFA/Geometric Median (Pillutla et al.)
- LASA (Xu et al.) - Layer-wise

**Personalization:**
- BR-MTRL (Le et al.) - Multi-task

**Best Performers:**
- ClippedClustering (Li et al.) - Best on Non-IID

#### Evaluation Metrics

**Performance:**
- Test accuracy (primary)
- AUROC, AUPRC (medical datasets)
- Convergence speed (rounds to target accuracy)

**Detection:**
- True Positive Rate (Byzantine identification)
- False Positive Rate (benign misclassification)
- F1 score for detection
- Detection latency (rounds until identification)

**Robustness:**
- Accuracy degradation vs. no-attack baseline
- Attack impact (accuracy drop under attacks)
- Breakdown point (max Byzantine fraction tolerated)

**Efficiency:**
- Communication cost (rounds × messages)
- Computation time per round
- Storage overhead (behavioral profiles)

#### Training Configuration

**FL Algorithm:**
- **FedAvg** (more robust than FedSGD per Li et al.)
- Local epochs: El = 50
- Communication rounds: 500-1000 (dataset dependent)

**Optimizer:**
- SGD with momentum (β = 0.9)
- Learning rate: η = 0.01 with decay schedule (Li et al.)

**Batch Size:**
- Default: 64 (CIFAR-10), 32 (medical datasets)
- Sensitivity test: {32, 64, 128, 256}

**Client Sampling:**
- Full participation (α = 1.0) baseline
- Partial: α = 0.2 (Li et al., Le et al.)

---

### Repurposable Experimental Design Elements (Cross-Paper)

#### 1. Non-IID Partitioning (Li et al., Xu et al., Le et al.)

```python
# Dirichlet distribution
from numpy.random import dirichlet

def create_non_iid_partition(labels, n_clients, alpha=0.5):
    """
    Allocate data to clients using Dirichlet distribution
    alpha=0.1: high heterogeneity
    alpha=0.5: medium heterogeneity
    alpha=1.0: low heterogeneity
    """
    n_classes = len(set(labels))
    class_indices = [np.where(labels == i)[0] for i in range(n_classes)]
    
    client_data = [[] for _ in range(n_clients)]
    for class_idx in class_indices:
        proportions = dirichlet([alpha] * n_clients)
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)
        
        for i in range(n_clients):
            start = 0 if i == 0 else proportions[i-1]
            end = proportions[i]
            client_data[i].extend(class_idx[start:end])
    
    return client_data
```

#### 2. ALIE Attack Implementation (Baruch et al.)

```python
from scipy.stats import norm

def alie_attack(benign_gradients, n_total, n_byzantine):
    """
    A Little Is Enough attack
    """
    # Calculate minimum supporters needed
    s = math.floor(n_total / 2 + 1) - n_byzantine
    
    # Calculate z_max
    target_prob = (n_total - n_byzantine - s) / (n_total - n_byzantine)
    z_max = norm.ppf(target_prob)
    
    # Estimate mean and std from Byzantine clients (non-omniscient)
    mu = benign_gradients.mean(dim=0)
    sigma = benign_gradients.std(dim=0)
    
    # Generate malicious gradient
    malicious_gradient = mu - z_max * sigma
    
    return malicious_gradient
```

#### 3. Pairwise Cosine Similarity Analysis (Li et al.)

```python
def analyze_update_similarity(client_updates):
    """
    Compute pairwise cosine similarities
    Explains why defenses perform differently
    """
    n_clients = len(client_updates)
    similarity_matrix = np.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(n_clients):
            cos_sim = cosine_similarity(
                client_updates[i].flatten(),
                client_updates[j].flatten()
            )
            similarity_matrix[i, j] = cos_sim
    
    return similarity_matrix
```

#### 4. Adaptive Attack Framework (Cao et al. FLTrust)

```python
def adaptive_attack_to_defense(defense_func, benign_updates, n_byzantine):
    """
    Zeroth-order optimization against specific defense
    """
    sigma_sq = 0.5  # Gaussian smoothing variance
    gamma = 0.005   # Smoothing parameter
    eta = 0.01      # Learning rate
    V = Q = 10      # Query budget
    
    # Initialize with baseline attack (e.g., Trim attack)
    malicious_updates = initialize_trim_attack(benign_updates, n_byzantine)
    
    for iteration in range(V):
        # Zeroth-order gradient estimate
        gradient = estimate_gradient(
            defense_func, benign_updates, malicious_updates, 
            sigma_sq, Q
        )
        
        # Gradient ascent (maximize attack impact)
        malicious_updates += eta * gradient
    
    return malicious_updates
```

#### 5. Behavioral Profile Tracking (CAAC-FL Core)

```python
class BehavioralProfile:
    """
    Client-specific behavioral profile
    """
    def __init__(self, client_id, window_size=10, alpha=0.1):
        self.client_id = client_id
        self.window_size = window_size
        self.alpha = alpha  # EWMA decay
        
        # Historical gradient norms
        self.norm_history = []
        
        # EWMA statistics
        self.mu = None  # Mean norm
        self.sigma = None  # Std dev
        
        # Directional consistency
        self.direction_history = []
        
    def update(self, gradient):
        """Update profile with new gradient"""
        # Magnitude
        norm = torch.norm(gradient, p=2).item()
        self.norm_history.append(norm)
        
        # EWMA update
        if self.mu is None:
            self.mu = norm
            self.sigma = 0
        else:
            self.mu = self.alpha * norm + (1 - self.alpha) * self.mu
            self.sigma = math.sqrt(
                self.alpha * (norm - self.mu)**2 + 
                (1 - self.alpha) * self.sigma**2
            )
        
        # Direction
        if len(self.direction_history) > 0:
            cos_sim = torch.cosine_similarity(
                gradient.flatten(),
                self.direction_history[-1].flatten(),
                dim=0
            ).item()
            self.direction_history.append(gradient)
        else:
            self.direction_history.append(gradient)
        
        # Trim to window size
        if len(self.norm_history) > self.window_size:
            self.norm_history.pop(0)
            self.direction_history.pop(0)
    
    def compute_anomaly_score(self, gradient):
        """Compute multi-dimensional anomaly score"""
        if self.mu is None:
            return 0.0
        
        norm = torch.norm(gradient, p=2).item()
        
        # Magnitude anomaly
        A_mag = abs(norm - self.mu) / (self.sigma + 1e-8)
        
        # Directional anomaly
        if len(self.direction_history) > 0:
            cos_sims = [
                torch.cosine_similarity(
                    gradient.flatten(),
                    hist_grad.flatten(),
                    dim=0
                ).item()
                for hist_grad in self.direction_history
            ]
            A_dir = 1 - np.mean(cos_sims)
        else:
            A_dir = 0.0
        
        # Temporal consistency (variance drift)
        if len(self.norm_history) > self.window_size // 2:
            sigma_old = np.std(self.norm_history[:self.window_size//2])
            sigma_new = np.std(self.norm_history[self.window_size//2:])
            A_temp = abs(sigma_new - sigma_old) / (sigma_old + 1e-8)
        else:
            A_temp = 0.0
        
        # Composite score (weights from CAAC-FL paper)
        lambda_mag = lambda_dir = lambda_temp = 1.0
        A_composite = math.sqrt(
            lambda_mag * A_mag**2 + 
            lambda_dir * A_dir**2 + 
            lambda_temp * A_temp**2
        )
        
        return A_composite
```

#### 6. Evaluation Table Format (Multiple Papers)

```markdown
| Attack Type | FedAvg | Krum | TrMean | FLTrust | LASA | CAAC-FL |
|-------------|--------|------|--------|---------|------|---------|
| No Attack   | 80.0%  | 10.0%| 80.0%  | 79.5%   | 81.0%| **82.0%**|
| ALIE        | 15.0%  | 12.0%| 18.0%  | 70.0%   | 75.0%| **78.0%**|
| IPM (ε=0.5) | 20.0%  | 15.0%| 22.0%  | 72.0%   | 76.0%| **79.0%**|
| IPM (ε=100) | 12.0%  | 18.0%| 25.0%  | 68.0%   | 74.0%| **77.0%**|
| Slow-drift  | 18.0%  | 14.0%| 20.0%  | 65.0%   | 70.0%| **80.0%**|
```

---

### Critical Experimental Validations for CAAC-FL

#### Must Demonstrate:

1. **Non-IID Performance:**
   - >80% accuracy without attacks (vs. <10% for Krum/GeoMed)
   - Maintain >70% under 20-40% Byzantine clients

2. **Attack Resilience:**
   - Outperform FLTrust on ALIE (adaptive attack test)
   - Outperform LASA on slow-drift (temporal advantage)
   - Match or exceed ClippedClustering on IPM

3. **Detection Accuracy:**
   - TPR > 90% (Byzantine identification)
   - FPR < 10% (benign misclassification)
   - Detection latency < 10 rounds

4. **Heterogeneity Robustness:**
   - Performance stable across α ∈ {0.1, 0.5, 1.0}
   - Graceful degradation with increasing Byzantine fraction

5. **Efficiency:**
   - Communication cost comparable to FedAvg
   - Computation overhead <20% vs. baselines

#### Ablation Studies Required:

1. **Component isolation:**
   - Magnitude anomaly only
   - Directional anomaly only
   - Temporal anomaly only
   - Pairwise combinations
   - Full CAAC-FL

2. **Hyperparameter sensitivity:**
   - Profile window size W ∈ {5, 10, 20}
   - EWMA decay α ∈ {0.05, 0.1, 0.2}
   - Anomaly weights λ_mag, λ_dir, λ_temp

3. **Architecture variants:**
   - Client-adaptive vs. global thresholds
   - With/without trust score integration
   - Different aggregation rules (GM, Krum, TrMean)

---

### Final Recommendations

**High Priority (Essential for CAAC-FL):**

1. Implement ALIE attack exactly as Baruch et al. (primary threat)
2. Use Li et al.'s Non-IID Dirichlet partitioning (α=0.1, α=0.5)
3. Compare against ClippedClustering (best baseline per Li et al.)
4. Include FLTrust with root dataset (privacy tradeoff analysis)
5. Test on CIFAR-10 for direct comparison to all baselines
6. Report TPR, FPR, detection latency (not just accuracy)

**Medium Priority (Strengthens Contribution):**

1. Test on FEMNIST (natural non-IID, compare to BR-MTRL)
2. Include adaptive attack framework from Cao et al.
3. Implement pairwise cosine similarity analysis (Li et al.)
4. Test Byzantine fractions up to 60% (FLTrust benchmark)
5. AWS federated testbed deployment (Le et al. infrastructure)

**Lower Priority (Nice to Have):**

1. Theoretical convergence analysis (Werner et al. framework)
2. Privacy analysis (differential privacy bounds)
3. Scalability to 1000+ clients
4. Communication compression techniques

---

## Conclusion

The 9 reference papers provide comprehensive experimental foundations for CAAC-FL:

1. **Motivation:** Li et al. demonstrates <10% accuracy problem on Non-IID
2. **Theory:** Werner et al. proves client-specific adaptation enhances robustness
3. **Attacks:** Baruch et al. (ALIE), Li et al. (IPM), Cao et al. (adaptive)
4. **Baselines:** Krum, TrimmedMean, FLTrust, LASA, ClippedClustering, BR-MTRL
5. **Protocols:** Non-IID partitioning, attack implementation, evaluation metrics

**CAAC-FL's Unique Contribution:**
- Client-adaptive behavioral profiling (not global thresholds)
- Temporal tracking (not single-round spatial)
- Multi-dimensional anomaly detection (magnitude + direction + temporal)
- Distinguishes heterogeneity from malice (fundamental gap)

**Experimental Success Criteria:**
- Survive Non-IID without attacks (>80% vs. 10% for Krum)
- Resist ALIE (>70% vs. <20% for most defenses)
- Detect slow-drift (unique temporal advantage)
- Low false positives (<10% vs. heterogeneous clients)

This comprehensive analysis provides CAAC-FL with a solid foundation for implementation, evaluation, and positioning within the Byzantine-robust federated learning landscape.

