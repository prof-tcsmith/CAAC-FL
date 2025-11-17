# Experimental Protocol  
**Title:** Evaluating CAAC-FL: Client-Adaptive Anomaly-Aware Clipping for Byzantine-Robust Federated Learning under Heterogeneous Healthcare Data  

_Last updated: YYYY-MM-DD_

---

## 1. Objectives and Hypotheses

### 1.1 Objectives

1. **Primary Objective**  
   Quantitatively evaluate the performance and robustness of **CAAC-FL** (Client-Adaptive Anomaly-Aware Clipping) compared to existing Byzantine-robust federated learning (FL) aggregation methods under **non-IID, heterogeneous healthcare data**.

2. **Secondary Objectives**
   - Measure how well CAAC-FL **preserves contributions from benign but systematically different clients** (e.g., institutions with atypical label distributions).
   - Assess CAAC-FL’s ability to resist multiple **attack types**, including magnitude-based, direction-based, and temporally structured attacks.
   - Characterize the **temporal behavior** of anomaly scores and clipping thresholds, including detection latency and false alarms.
   - Quantify the **computational overhead** of CAAC-FL relative to competing methods.

### 1.2 Hypotheses

- **H1 – Heterogeneity Preservation**  
  Under high client heterogeneity and *no* attack, CAAC-FL:
  - Achieves equal or better global performance (accuracy / AUROC / AUPRC) than existing robust methods; and  
  - Exhibits a **lower false positive rate (FPR)** in flagging / clipping benign clients whose gradients systematically deviate from the majority.

- **H2 – Multi-Dimensional Robustness**  
  Against a range of Byzantine attacks (including ALIE, inner-product manipulation, and adaptive attacks), CAAC-FL’s multi-dimensional anomaly detection (magnitude + direction + temporal consistency) leads to:
  - Smaller degradation in global performance; and  
  - Lower false negative rate (FNR) in failing to down-weight malicious clients, compared to single-dimension or non-temporal baselines.

- **H3 – Temporal Discrimination**  
  CAAC-FL better distinguishes:
  - Abrupt, malicious changes in client behavior; from  
  - Slow, benign domain shift (e.g., changes in institutional case mix),  
  yielding faster detection of abrupt attacks and fewer false alarms under gradual benign changes.

---

## 2. Experimental Factors and Design

### 2.1 Independent Variables

1. **Aggregation Method (Factor A)**
   - A1: FedAvg (non-robust baseline)  
   - A2: Coordinate-wise Median  
   - A3: Trimmed Mean  
   - A4: Krum (or Multi-Krum)  
   - A5: RFA (Robust Federated Aggregation via geometric median)  
   - A6: FLTrust  
   - A7: LASA (Layer-Adaptive Sparsified Aggregation)  
   - A8: CAAC-FL (proposed method)  
   - (Optional A9: A representative personalized robust FL method, e.g., PRFL / BR-MTRL-style.)

2. **Data Heterogeneity Level (Factor B – via Dirichlet α)**
   - B1: Mild non-IID — \(\alpha = 1.0\)  
   - B2: Moderate non-IID — \(\alpha = 0.5\)  
   - B3: Extreme non-IID — \(\alpha = 0.1\)

3. **Byzantine Proportion (Factor C)**  
   - C1: 0% (no attack)  
   - C2: 10% of clients Byzantine  
   - C3: 20% of clients Byzantine  
   - C4: 40% of clients Byzantine  

4. **Attack Type (Factor D)**  
   - D1: None (clean training)  
   - D2: Random noise  
   - D3: Sign-flipping  
   - D4: ALIE (small but adversarial perturbations)  
   - D5: Inner-Product Manipulation (IPM)  
   - D6: Slow-drift poisoning (gradual shift)  
   - D7: Adaptive profile-aware attack (mimics benign profiles)

5. **Dataset / Task (Factor E)**
   - E1: MIMIC-III — ICU mortality (binary classification; tabular / temporal)  
   - E2: ChestX-ray8 — multi-label thoracic disease classification  
   - E3: ISIC 2019 — melanoma / skin-lesion classification (multi-class or binary)

6. **Client Count and Data Size**
   - Default: 20 clients per dataset, with non-uniform sample sizes per client (power-law distribution).
   - Optional scaling study: 50 and/or 100 clients.

### 2.2 Dependent Variables

1. **Global Performance**
   - Test accuracy, AUROC, AUPRC (task-appropriate).
   - Convergence speed (rounds to reach a predefined performance threshold).

2. **Robustness Metrics**
   - Drop in performance under each attack vs the corresponding clean baseline.
   - Targeted attack success rate (if applicable).

3. **Client-Level Detection Metrics**
   - **FPR (Benign):** fraction of benign clients whose updates are flagged as anomalous or heavily clipped.  
   - **FNR (Malicious):** fraction of Byzantine clients whose updates retain high influence.  
   - **Detection Latency:** number of rounds between attack onset and consistent down-weighting.

4. **Temporal Behavior**
   - Evolution of anomaly scores and clipping thresholds over rounds.
   - Stability of CAAC-FL under slow, benign domain shift.

5. **Overhead**
   - Additional per-round computation time at the server.
   - Additional server memory for per-client statistics.

---

## 3. Materials

### 3.1 Software

- Language: **Python 3.9+**  
- FL framework: **Flower** (preferred) or **FedML**  
- Deep learning library: **PyTorch** (preferred) or TensorFlow  
- Dependency management: `conda` or `pip` with `requirements.txt`  
- Experiment tracking: **Weights & Biases** or **MLflow** (recommended), plus local CSV/JSON logs  
- Version control: **Git**

### 3.2 Hardware

- GPU-equipped server (e.g., ≥1 NVIDIA GPU).  
- Sufficient CPU and RAM to simulate ≥20 clients (clients can be simulated sequentially or in parallel).

### 3.3 Data Preparation

For each dataset:

#### 3.3.1 MIMIC-III

- Task: ICU in-hospital mortality prediction (binary classification).
- Steps:
  - Extract relevant cohort and features (per chosen task definition).
  - Preprocess (e.g., imputation, normalization, time-window selection).
  - Split into **train / validation / test** (e.g., 70% / 10% / 20%).
- Save processed data in a reproducible format (e.g., HDF5 / Parquet).

#### 3.3.2 ChestX-ray8

- Task: multi-label thoracic disease classification.
- Steps:
  - Load images and disease labels.
  - Preprocess: resize to a fixed size (e.g., 224×224), normalize channels.
  - Create train / validation / test splits (stratified if possible).

#### 3.3.3 ISIC 2019

- Task: melanoma vs non-melanoma (or multi-class lesion classification).
- Steps:
  - Load dermatoscopic images and labels.
  - Preprocess: resize, normalize.
  - Create train / validation / test splits.

> **Note:** All preprocessing must be scripted and checked into the repo.

---

## 4. Model Architectures

Use standardized architectures per dataset:

### 4.1 MIMIC-III

- **Option A:** MLP on engineered features.  
- **Option B:** GRU or 1D-CNN on time-series inputs.

Output layer:
- Single logit + sigmoid for mortality probability.

### 4.2 ChestX-ray8

- Model: **ResNet-18** or **MobileNet-V2**.
- Output:
  - Multi-label head with one logit per disease; sigmoid activations.

### 4.3 ISIC 2019

- Model: **ResNet-50** or **EfficientNet-B0**.
- Output:
  - Softmax over lesion types or binary melanoma classification.

### 4.4 Common Training Hyperparameters (baseline)

- Optimizer: SGD or Adam  
- Learning rate:  
  - SGD: ~0.01  
  - Adam: ~1e-3  
- Batch size: ~32  
- Local epochs per round: 1–5  
- Number of federated rounds: 100–200 (dataset-dependent)

> **Constraint:** For a given dataset, all aggregation methods must use the **same architecture and hyperparameters**. Only the aggregation rule is varied.

---

## 5. Federated Learning Setup

### 5.1 Client Partitioning

- Number of clients: **20** per dataset.
- Data heterogeneity:
  - Use **Dirichlet sampling** over labels with parameter \(\alpha \in \{1.0, 0.5, 0.1\}\).
- Sample size imbalance:
  - Use a **power-law** distribution to assign different quantities of data per client (large vs small hospitals).
- Byzantine assignment:
  - For trials with attacks, randomly select \(f\) clients to be malicious, where \(f \in \{2, 4, 8\}\) for 10%, 20%, 40% in a 20-client setup.

### 5.2 Training Loop (Per Round)

For each round \(t = 1, \dots, T\):

1. **Server broadcast:**  
   - Send current global model \(w^t\) to all participating clients.

2. **Client-side local training (for each client \(i\)):**
   - Receive \(w^t\).
   - Perform \(E\) local epochs of SGD/Adam on local data.
   - Compute local update:
     \[
       g_i^t = w_i^t - w^t \quad \text{(or local gradient equivalent)}
     \]
   - If client \(i\) is Byzantine:
     - Apply designated **attack transformation** to \(g_i^t\).
   - Send \(g_i^t\) to server.

3. **Server aggregation:**
   - Apply chosen aggregation method (FedAvg, Median, Krum, RFA, FLTrust, LASA, CAAC-FL, etc.) to obtain global update \(\Delta w^t\).

4. **Server model update:**
   \[
     w^{t+1} = w^t + \eta \Delta w^t
   \]

5. **Logging:**
   - Compute validation metrics on a held-out validation set.
   - For CAAC-FL (and any method with per-client diagnostics), log per-client:
     - Gradient norm
     - Cosine similarity to global gradient (or previous gradients)
     - Anomaly score
     - Clipping threshold and clipping factor
     - “Benign vs anomalous” flag (if applicable)

---

## 6. Attack Definitions

Attacks are modular transformations applied to \(g_i^t\) on Byzantine clients.

### 6.1 Basic Attacks

1. **Random Noise (D2)**  
   - Replace gradient with Gaussian noise:
     \[
       g_i^t \leftarrow z^t, \quad z^t \sim \mathcal{N}(0, \sigma^2 I)
     \]
   - Choose \(\sigma\) so that \(\|z^t\|\) is similar to a typical benign gradient norm.

2. **Sign-Flipping (D3)**  
   - Flip and scale the gradient:
     \[
       g_i^t \leftarrow -\gamma g_i^t
     \]
   - Use \(\gamma > 1\) (e.g., 10) to increase adversarial impact.

### 6.2 Advanced Attacks

3. **ALIE (D4)**  
   - Coordinate-wise manipulation designed to:
     - Stay within typical coordinate ranges,
     - Still steer the global update in an adversarial direction.
   - Implementation: use the original ALIE algorithm (mean + scaled std per coordinate).

4. **Inner-Product Manipulation (IPM, D5)**  
   - Construct gradients that:
     - Match benign-like norm statistics,
     - But are oriented in a direction that misleads global training (e.g., roughly opposite to or orthogonal to benign gradients).

### 6.3 Temporal / Adaptive Attacks

5. **Slow-Drift Poisoning (D6)**  
   - Gradual transition from benign to adversarial:
     \[
       g_i^t \leftarrow (1 - \lambda_t) g_i^{\text{benign}, t} + \lambda_t g_i^{\text{adv}, t}
     \]
   - Schedule \(\lambda_t\) to increase slowly over ~20–30 rounds.

6. **Profile-Aware Attack (D7)**  
   - Attacker attempts to mimic properties that CAAC-FL monitors:
     - Keep norms close to historical mean.
     - Maintain cosine similarity within historical range.
     - Introduce a subtle, persistent bias in direction (e.g., small angle deviation each round).

> All attacks should be encapsulated as pluggable functions, with parameters and random seeds clearly documented.

---

## 7. CAAC-FL Implementation

CAAC-FL is implemented at the **server** as a custom aggregation rule.

### 7.1 Per-Client Statistics

For each client \(i\), maintain:

- EWMA of gradient norm: \(\mu_i^t\)  
- EWMA of norm variance: \((\sigma_i^t)^2\)  
- EWMA of directional consistency (e.g., cosine similarity with previous gradient or global gradient): \(\rho_i^t\)  
- Reliability score: \(R_i^t \in [0, 1]\), reflecting historical benign behavior.

EWMAs are updated as:
\[
\mu_i^t = \beta \mu_i^{t-1} + (1 - \beta) \|g_i^t\|
\]
(similarly for variance and directional consistency), where \(\beta \in [0, 1)\) is a decay factor (typically \(\beta = 0.9\)).

Update equations:
\[
\begin{align}
\mu_i^t &= \beta \mu_i^{t-1} + (1 - \beta) \|g_i^t\| \\
(\sigma_i^t)^2 &= \beta (\sigma_i^{t-1})^2 + (1 - \beta) (\|g_i^t\| - \mu_i^t)^2 \\
\rho_i^t &= \beta \rho_i^{t-1} + (1 - \beta) \cos(g_i^t, \bar{g}^{t-1})
\end{align}
\]

where \(\bar{g}^{t-1}\) is the aggregated global gradient from the previous round.

### 7.2 Anomaly Score Computation

For each client \(i\) at round \(t\), compute three anomaly components:

#### 7.2.1 Magnitude Anomaly
\[
A_{mag}^{i,t} = \frac{|\|g_i^t\| - \mu_i^{t-1}|}{\sigma_i^{t-1} + \epsilon}
\]
where \(\epsilon = 10^{-8}\) for numerical stability.

#### 7.2.2 Directional Anomaly
\[
A_{dir}^{i,t} = 1 - \frac{1}{W} \sum_{k=t-W}^{t-1} \cos(g_i^t, g_i^k)
\]
where \(W = 10\) is the window size for historical comparison.

#### 7.2.3 Temporal Consistency Anomaly
\[
A_{temp}^{i,t} = \frac{|\sigma_i^t - \sigma_i^{t-W}|}{\sigma_i^{t-W} + \epsilon}
\]

#### 7.2.4 Composite Anomaly Score
\[
A_i^t = \sqrt{\lambda_{mag}(A_{mag}^{i,t})^2 + \lambda_{dir}(A_{dir}^{i,t})^2 + \lambda_{temp}(A_{temp}^{i,t})^2}
\]

Default weights: \(\lambda_{mag} = 0.4\), \(\lambda_{dir} = 0.4\), \(\lambda_{temp} = 0.2\).

### 7.3 Reliability Score Update

The reliability score \(R_i^t\) tracks long-term trustworthiness:

\[
R_i^t = \gamma \cdot \mathbb{1}(A_i^t < \tau_{anomaly}) + (1 - \gamma) \cdot R_i^{t-1}
\]

where:
- \(\gamma = 0.1\) is the reliability smoothing parameter
- \(\tau_{anomaly} = 2.0\) is the anomaly threshold
- \(\mathbb{1}(\cdot)\) is the indicator function
- Initial reliability: \(R_i^0 = 0.5\) for all clients

### 7.4 Adaptive Clipping Threshold

The client-specific clipping threshold is:

\[
\tau_i^t = \mu_{global}^t \cdot f(A_i^t, R_i^t)
\]

where \(\mu_{global}^t\) is the median of all gradient norms at round \(t\), and:

\[
f(A_i^t, R_i^t) = \min\left(2.0, \max\left(0.1, \exp\left(-\frac{A_i^t}{2}\right) \cdot (1 + R_i^t)\right)\right)
\]

This function:
- Decreases exponentially with anomaly score
- Increases linearly with reliability
- Bounded between 0.1 and 2.0 times the global median

### 7.5 Bootstrap Phase

For rounds \(t \leq T_{bootstrap}\) where \(T_{bootstrap} = 20\):

1. Apply uniform clipping with conservative threshold:
   \[
   \tau_i^t = 0.5 \cdot \mu_{global}^t \quad \forall i
   \]

2. Collect statistics without applying client-specific adjustments
3. Initialize EWMA statistics:
   - \(\mu_i^{T_{bootstrap}} = \frac{1}{T_{bootstrap}} \sum_{t=1}^{T_{bootstrap}} \|g_i^t\|\)
   - \((\sigma_i^{T_{bootstrap}})^2 = \text{Var}(\{\|g_i^1\|, ..., \|g_i^{T_{bootstrap}}\|\})\)

4. Transition criteria (all must be met):
   - Minimum rounds completed: \(t > T_{bootstrap}\)
   - Stability check: \(\text{CV}(\mu_{global}^{t-5:t}) < 0.3\) (coefficient of variation over last 5 rounds)
   - Sufficient participation: Each client has participated in ≥50% of bootstrap rounds

### 7.6 Gradient Clipping and Aggregation

Given gradients \(\{g_i^t\}_{i=1}^N\) and thresholds \(\{\tau_i^t\}_{i=1}^N\):

1. **Clip each gradient:**
   \[
   \tilde{g}_i^t = g_i^t \cdot \min\left(1, \frac{\tau_i^t}{\|g_i^t\|}\right)
   \]

2. **Compute weights based on reliability and anomaly:**
   \[
   w_i^t = \frac{R_i^t \cdot \exp(-A_i^t/4)}{\sum_{j=1}^N R_j^t \cdot \exp(-A_j^t/4)}
   \]

3. **Weighted aggregation:**
   \[
   \Delta w^t = \sum_{i=1}^N w_i^t \cdot \tilde{g}_i^t
   \]

### 7.7 Handling New Clients

When client \(j\) joins at round \(t_{join}\):

1. Initialize with conservative defaults:
   - \(\mu_j^{t_{join}} = \mu_{global}^{t_{join}}\)
   - \(\sigma_j^{t_{join}} = \text{median}(\{\sigma_i^{t_{join}}\}_{i \neq j})\)
   - \(R_j^{t_{join}} = 0.3\) (lower than established clients)

2. Apply stricter clipping for probation period (10 rounds):
   \[
   \tau_j^t = 0.7 \cdot \tau_j^t \quad \text{for } t \in [t_{join}, t_{join} + 10]
   \]

3. Gradual trust building through reliability score updates

### 7.8 Algorithm Pseudocode

```python
def CAAC_FL_aggregate(gradients, round_t, client_stats, params):
    """
    CAAC-FL Aggregation Algorithm
    
    Args:
        gradients: Dict[client_id, gradient_tensor]
        round_t: Current round number
        client_stats: Dict[client_id, ClientProfile]
        params: Hyperparameters dictionary
    
    Returns:
        aggregated_gradient: Weighted aggregate of clipped gradients
        updated_stats: Updated client statistics
    """
    
    # Bootstrap phase
    if round_t <= params['T_bootstrap']:
        return bootstrap_aggregation(gradients, round_t, client_stats)
    
    # Compute global statistics
    grad_norms = {i: torch.norm(g).item() for i, g in gradients.items()}
    global_median = np.median(list(grad_norms.values()))
    
    # Initialize containers
    anomaly_scores = {}
    reliability_scores = {}
    clipping_thresholds = {}
    clipped_gradients = {}
    
    for client_id, gradient in gradients.items():
        stats = client_stats[client_id]
        
        # Update EWMA statistics
        stats.update_ewma(gradient, params['beta'])
        
        # Compute anomaly scores
        a_mag = stats.compute_magnitude_anomaly(gradient)
        a_dir = stats.compute_directional_anomaly(gradient, params['window_size'])
        a_temp = stats.compute_temporal_anomaly(params['window_size'])
        
        # Composite anomaly score
        anomaly = np.sqrt(
            params['lambda_mag'] * a_mag**2 +
            params['lambda_dir'] * a_dir**2 +
            params['lambda_temp'] * a_temp**2
        )
        anomaly_scores[client_id] = anomaly
        
        # Update reliability score
        stats.update_reliability(anomaly, params['gamma'], params['tau_anomaly'])
        reliability_scores[client_id] = stats.reliability
        
        # Compute adaptive threshold
        threshold = compute_threshold(
            anomaly, stats.reliability, global_median, params
        )
        clipping_thresholds[client_id] = threshold
        
        # Clip gradient
        clipped_gradients[client_id] = clip_gradient(gradient, threshold)
    
    # Compute weights and aggregate
    weights = compute_aggregation_weights(
        anomaly_scores, reliability_scores, params
    )
    
    aggregated = weighted_sum(clipped_gradients, weights)
    
    # Log diagnostics
    log_round_diagnostics(
        round_t, anomaly_scores, reliability_scores, 
        clipping_thresholds, weights
    )
    
    return aggregated, client_stats
```

---

## 8. Evaluation Metrics

### 8.1 Primary Performance Metrics

#### 8.1.1 Model Performance
- **Binary Classification (MIMIC-III, ISIC-melanoma)**:
  - AUROC (Area Under ROC Curve)
  - AUPRC (Area Under Precision-Recall Curve)
  - Balanced Accuracy
  - F1 Score
  - Sensitivity and Specificity at optimal threshold

- **Multi-label Classification (ChestX-ray8)**:
  - Macro-averaged AUROC
  - Micro-averaged AUROC
  - Per-disease AUROC
  - Hamming Loss
  - Subset Accuracy

- **Convergence Metrics**:
  - Rounds to reach 90% of best performance
  - Final performance at round 100 and 200
  - Performance stability (std dev over last 20 rounds)

### 8.2 Robustness Metrics

#### 8.2.1 Attack Impact
\[
\text{Performance Degradation} = \frac{\text{Performance}_{clean} - \text{Performance}_{attack}}{\text{Performance}_{clean}} \times 100\%
\]

#### 8.2.2 Byzantine Detection Metrics
- **True Positive Rate (TPR)**: Fraction of Byzantine clients correctly identified
  \[
  \text{TPR} = \frac{|\{i : i \in \mathcal{B} \land A_i^t > \tau_{anomaly}\}|}{|\mathcal{B}|}
  \]

- **False Positive Rate (FPR)**: Fraction of benign clients incorrectly flagged
  \[
  \text{FPR} = \frac{|\{i : i \notin \mathcal{B} \land A_i^t > \tau_{anomaly}\}|}{N - |\mathcal{B}|}
  \]

- **Detection Latency**: Rounds from attack onset to consistent detection
  \[
  L_{\text{detect}} = \min\{t : \text{TPR}_t > 0.8 \text{ for } 5 \text{ consecutive rounds}\} - t_{\text{attack}}
  \]

### 8.3 Heterogeneity Preservation Metrics

#### 8.3.1 Client Contribution Analysis
- **Effective Participation Rate**: 
  \[
  \text{EPR} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(w_i^t > \frac{1}{2N})
  \]

- **Contribution Entropy**:
  \[
  H(w^t) = -\sum_{i=1}^N w_i^t \log(w_i^t)
  \]

#### 8.3.2 Per-Client Performance
- Local test accuracy for each client's data distribution
- Fairness metrics: standard deviation and Gini coefficient of per-client accuracies

### 8.4 Temporal Behavior Metrics

- **Anomaly Score Stability**: Coefficient of variation over time for benign clients
- **Threshold Adaptation Rate**: Average change in thresholds per round
- **False Alarm Rate under Drift**: FPR during simulated benign domain shift

### 8.5 Computational Metrics

- **Server-side overhead per round** (seconds)
- **Additional memory usage** (MB) for client profiles
- **Communication overhead** if additional statistics are transmitted
- **Scalability**: Performance with 20, 50, 100 clients

---

## 9. Statistical Analysis Plan

### 9.1 Experimental Design

- **Design Type**: Full factorial design with selected interactions
- **Replication**: 5 independent runs per configuration with different random seeds
- **Total Experiments**: 
  - Factors: 8 methods × 3 heterogeneity levels × 4 Byzantine fractions × 7 attack types × 3 datasets
  - Subset for computational feasibility: ~500 core experiments

### 9.2 Hypothesis Testing

#### 9.2.1 H1 - Heterogeneity Preservation
- **Test**: Two-way ANOVA with post-hoc Tukey HSD
- **Factors**: Method (CAAC-FL vs baselines) × Heterogeneity level
- **Response**: FPR on benign clients
- **Null Hypothesis**: No difference in FPR between CAAC-FL and baselines
- **Significance Level**: α = 0.05 with Bonferroni correction

#### 9.2.2 H2 - Multi-Dimensional Robustness
- **Test**: Paired t-test for performance degradation
- **Comparison**: CAAC-FL vs each baseline under each attack
- **Null Hypothesis**: Equal performance degradation
- **Effect Size**: Cohen's d to quantify improvement magnitude

#### 9.2.3 H3 - Temporal Discrimination
- **Test**: Time-series analysis of detection latency
- **Methods**: 
  - Cox proportional hazards model for time-to-detection
  - Repeated measures ANOVA for anomaly score evolution
- **Covariates**: Attack type, heterogeneity level, Byzantine fraction

### 9.3 Statistical Power Analysis

Target power = 0.8 for detecting:
- 10% difference in AUROC
- 15% difference in FPR
- 5-round difference in detection latency

Required sample sizes (preliminary):
- 5 runs × 3 datasets = 15 observations per configuration
- Minimum detectable effect sizes computed via simulation

### 9.4 Missing Data and Outliers

- **Missing Rounds**: Clients may not participate every round
  - Handle via listwise deletion for per-round metrics
  - Use last observation carried forward for temporal analyses

- **Outlier Detection**: 
  - Identify via Grubbs' test at p < 0.001
  - Report results with and without outliers
  - Investigate outliers for implementation bugs

### 9.5 Reporting Standards

Follow CONSORT-AI reporting guidelines:
1. Report mean ± standard deviation for all metrics
2. Include 95% confidence intervals for key comparisons
3. Publish all raw experimental data
4. Provide effect sizes alongside p-values
5. Include convergence plots with error bands

---

## 10. Implementation Details

### 10.1 Software Architecture

```
caac-fl/
├── src/
│   ├── aggregators/
│   │   ├── base.py              # Abstract aggregator class
│   │   ├── fedavg.py
│   │   ├── median.py
│   │   ├── trimmed_mean.py
│   │   ├── krum.py
│   │   ├── rfa.py
│   │   ├── fltrust.py
│   │   ├── lasa.py
│   │   └── caac_fl.py           # Our method
│   ├── attacks/
│   │   ├── base.py              # Abstract attack class
│   │   ├── noise.py
│   │   ├── sign_flip.py
│   │   ├── alie.py
│   │   ├── ipm.py
│   │   ├── slow_drift.py
│   │   └── adaptive.py
│   ├── datasets/
│   │   ├── mimic.py
│   │   ├── chestxray.py
│   │   └── isic.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── resnet.py
│   │   └── mobilenet.py
│   ├── clients/
│   │   ├── client.py            # Client class
│   │   └── client_profile.py    # CAAC-FL profile tracking
│   ├── server/
│   │   ├── server.py            # Federated server
│   │   └── logger.py            # Experiment logging
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── config.py
├── configs/                      # Experiment configurations (YAML)
├── scripts/                      # Execution scripts
├── tests/                        # Unit and integration tests
├── results/                      # Experimental results
├── notebooks/                    # Analysis notebooks
└── requirements.txt
```

### 10.2 Key Implementation Classes

#### 10.2.1 ClientProfile Class
```python
@dataclass
class ClientProfile:
    """Maintains behavioral statistics for CAAC-FL"""
    client_id: int
    mu_norm: float = 0.0          # EWMA of gradient norm
    sigma_norm: float = 1.0       # EWMA of norm std dev
    rho_direction: float = 0.0    # EWMA of directional consistency
    reliability: float = 0.5      # Reliability score
    gradient_history: deque       # Last W gradients
    anomaly_history: list         # Historical anomaly scores
    rounds_participated: int = 0
    last_update_round: int = 0
    is_bootstrap: bool = True
```

#### 10.2.2 Attack Implementation
```python
class AdaptiveAttack(Attack):
    """Profile-aware adaptive attack"""
    
    def __init__(self, target_profile: ClientProfile, 
                 stealth_factor: float = 0.1):
        self.target_profile = target_profile
        self.stealth_factor = stealth_factor
        
    def apply(self, gradient: torch.Tensor, 
              round_t: int) -> torch.Tensor:
        # Compute target norm to match profile
        target_norm = self.target_profile.mu_norm
        
        # Add adversarial component while maintaining norm
        adv_direction = self.compute_adversarial_direction(gradient)
        scaling = target_norm / torch.norm(adv_direction)
        
        # Blend with original to maintain some consistency
        alpha = min(self.stealth_factor * round_t / 100, 0.5)
        return (1 - alpha) * gradient + alpha * scaling * adv_direction
```

### 10.3 Configuration Management

#### 10.3.1 Experiment Configuration (YAML)
```yaml
experiment:
  name: "caac_fl_mimic_extreme_heterogeneity"
  seed: 42
  num_rounds: 200
  num_clients: 20
  clients_per_round: 10
  
dataset:
  name: "mimic"
  split_strategy: "dirichlet"
  alpha: 0.1  # Extreme heterogeneity
  
model:
  architecture: "mlp"
  hidden_layers: [128, 64]
  dropout: 0.2
  
training:
  local_epochs: 5
  batch_size: 32
  optimizer: "sgd"
  learning_rate: 0.01
  
aggregator:
  method: "caac_fl"
  params:
    beta: 0.9
    gamma: 0.1
    window_size: 10
    lambda_mag: 0.4
    lambda_dir: 0.4
    lambda_temp: 0.2
    tau_anomaly: 2.0
    bootstrap_rounds: 20
    
attack:
  type: "adaptive"
  byzantine_fraction: 0.2
  start_round: 50
```

### 10.4 Reproducibility Measures

1. **Random Seed Management**:
   ```python
   def set_reproducible_seed(seed: int):
       torch.manual_seed(seed)
       np.random.seed(seed)
       random.seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```

2. **Dependency Pinning**: Exact versions in `requirements.txt`

3. **Data Versioning**: SHA-256 hashes of preprocessed datasets

4. **Code Versioning**: Git commit hash logged with each experiment

5. **Environment Capture**: 
   ```python
   pip freeze > experiment_environment.txt
   ```

### 10.5 Testing Strategy

#### 10.5.1 Unit Tests
- Each aggregator method tested independently
- Attack transformations verified for correctness
- Statistical computations validated against known values

#### 10.5.2 Integration Tests
- Full training loop with synthetic data
- Byzantine detection on controlled scenarios
- Convergence on toy problems

#### 10.5.3 Regression Tests
- Performance benchmarks must stay within 5% of baseline
- Detection metrics tracked across code changes

#### 10.5.4 Stress Tests
- Scalability up to 1000 clients (simulated)
- Memory usage under prolonged training
- Numerical stability with extreme parameters

---

## 11. Computational Requirements and Optimization

### 11.1 Hardware Requirements

#### 11.1.1 Minimum Configuration
- CPU: 8-core x86_64 processor
- RAM: 32 GB
- GPU: NVIDIA GPU with 8GB VRAM (GTX 1070 or better)
- Storage: 500 GB SSD

#### 11.1.2 Recommended Configuration
- CPU: 16-core x86_64 processor
- RAM: 64 GB
- GPU: NVIDIA GPU with 16GB+ VRAM (V100, A100, or RTX 3090)
- Storage: 1 TB NVMe SSD

### 11.2 Computational Complexity Analysis

#### 11.2.1 CAAC-FL Overhead
Per-round computational complexity:
- Profile updates: O(N) for N clients
- Anomaly computation: O(N × W) where W is window size
- Threshold calculation: O(N)
- Aggregation: O(N × D) where D is model dimension

Memory complexity:
- Client profiles: O(N × W × D) 
- Historical statistics: O(N × T) for T rounds

#### 11.2.2 Baseline Comparisons
| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| FedAvg | O(N × D) | O(D) |
| Median | O(N × D log N) | O(N × D) |
| Krum | O(N² × D) | O(N × D) |
| CAAC-FL | O(N × W × D) | O(N × W × D) |

### 11.3 Optimization Strategies

#### 11.3.1 Computational Optimizations
```python
# Vectorized anomaly computation
def compute_anomalies_vectorized(profiles, gradients):
    norms = torch.stack([torch.norm(g) for g in gradients])
    mus = torch.tensor([p.mu_norm for p in profiles])
    sigmas = torch.tensor([p.sigma_norm for p in profiles])
    
    # Vectorized magnitude anomaly
    a_mag = torch.abs(norms - mus) / (sigmas + 1e-8)
    
    # Parallel cosine similarity computation
    similarities = F.cosine_similarity(
        gradients.unsqueeze(1), 
        gradient_history.unsqueeze(0), 
        dim=2
    )
    a_dir = 1 - similarities.mean(dim=1)
    
    return a_mag, a_dir
```

#### 11.3.2 Memory Optimizations
- Gradient checkpointing for large models
- Lazy loading of client gradients
- Circular buffer for gradient history (fixed memory)
- Sparse gradient storage for communication efficiency

#### 11.3.3 Distributed Computing
```python
# Multi-GPU server aggregation
def distributed_aggregation(gradients, world_size):
    # Shard clients across GPUs
    clients_per_gpu = len(gradients) // world_size
    
    # Parallel anomaly computation
    with mp.Pool(world_size) as pool:
        anomaly_shards = pool.map(
            compute_anomalies,
            [gradients[i:i+clients_per_gpu] 
             for i in range(0, len(gradients), clients_per_gpu)]
        )
    
    return aggregate_shards(anomaly_shards)
```

### 11.4 Resource Monitoring

```python
class ResourceMonitor:
    """Track computational resources during experiments"""
    
    def log_round_resources(self, round_t):
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_gb': psutil.virtual_memory().used / 1e9,
            'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
            'aggregation_time_s': self.aggregation_timer.elapsed(),
            'communication_time_s': self.comm_timer.elapsed()
        }
        wandb.log(metrics, step=round_t)
```

---

## 12. Timeline and Milestones

### 12.1 Project Timeline (16 weeks)

#### Phase 1: Infrastructure Setup (Weeks 1-3)
- Week 1:
  - Set up development environment
  - Initialize repository structure
  - Implement base classes for clients, server, aggregators
  
- Week 2:
  - Implement FedAvg and basic baselines
  - Set up data loading pipelines
  - Create configuration management system
  
- Week 3:
  - Implement evaluation metrics
  - Set up logging and monitoring
  - Write unit tests for core components

#### Phase 2: CAAC-FL Implementation (Weeks 4-6)
- Week 4:
  - Implement client profile tracking
  - Develop EWMA statistics updates
  - Create anomaly score computations
  
- Week 5:
  - Implement adaptive thresholding
  - Develop bootstrap phase logic
  - Create weighted aggregation mechanism
  
- Week 6:
  - Integration testing of CAAC-FL
  - Performance optimization
  - Debug and refinement

#### Phase 3: Attack Implementation (Weeks 7-8)
- Week 7:
  - Implement basic attacks (noise, sign-flip)
  - Implement ALIE and IPM attacks
  - Test attack effectiveness
  
- Week 8:
  - Implement slow-drift attack
  - Implement adaptive profile-aware attack
  - Validate attack implementations

#### Phase 4: Experimental Execution (Weeks 9-12)
- Week 9:
  - Run experiments on MIMIC-III
  - Initial results analysis
  - Debug any issues
  
- Week 10:
  - Run experiments on ChestX-ray8
  - Continue MIMIC-III experiments
  
- Week 11:
  - Run experiments on ISIC 2019
  - Complete remaining configurations
  
- Week 12:
  - Run additional experiments for gaps
  - Collect all experimental data
  - Preliminary analysis

#### Phase 5: Analysis and Writing (Weeks 13-15)
- Week 13:
  - Statistical analysis of results
  - Generate figures and tables
  - Identify key findings
  
- Week 14:
  - Write results section
  - Write discussion section
  - Create presentation materials
  
- Week 15:
  - Finalize paper
  - Prepare code for release
  - Create documentation

#### Phase 6: Dissemination (Week 16)
- Final review and submission
- Prepare supplementary materials
- Archive experimental artifacts

### 12.2 Key Milestones

1. **M1 (Week 3)**: Basic FL infrastructure operational
2. **M2 (Week 6)**: CAAC-FL fully implemented and tested
3. **M3 (Week 8)**: All attacks implemented
4. **M4 (Week 10)**: 50% of experiments complete
5. **M5 (Week 12)**: All experiments complete
6. **M6 (Week 15)**: Paper draft complete
7. **M7 (Week 16)**: Submission ready

### 12.3 Risk Management

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Computational resources insufficient | High | Medium | Use cloud computing, reduce experiment scope |
| CAAC-FL underperforms | High | Low | Have backup algorithmic improvements ready |
| Data access issues | High | Low | Have backup datasets identified |
| Implementation bugs | Medium | Medium | Extensive testing, code review |
| Timeline delays | Medium | Medium | Build in 20% buffer time |

---

## 13. Code Organization and Documentation

### 13.1 Repository Structure

```
CAAC-FL/
├── README.md                    # Project overview and setup
├── LICENSE                      # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── .gitignore
├── .github/
│   ├── workflows/
│   │   ├── tests.yml          # CI/CD for testing
│   │   └── lint.yml           # Code quality checks
├── docs/
│   ├── installation.md         # Detailed setup instructions
│   ├── usage.md               # How to run experiments
│   ├── api.md                 # API documentation
│   └── results.md             # Results reproduction guide
├── src/                        # Source code (detailed above)
├── configs/
│   ├── default.yaml           # Default configuration
│   ├── experiments/           # Experiment-specific configs
│   └── sweeps/                # Hyperparameter sweep configs
├── scripts/
│   ├── setup_env.sh           # Environment setup script
│   ├── download_data.py       # Data download utility
│   ├── preprocess_data.py     # Data preprocessing
│   ├── run_experiment.py      # Main experiment runner
│   └── analyze_results.py     # Results analysis
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── notebooks/
│   ├── exploratory/           # Data exploration
│   ├── analysis/              # Results analysis
│   └── figures/               # Figure generation
├── results/
│   ├── logs/                  # Experiment logs
│   ├── checkpoints/           # Model checkpoints
│   └── figures/               # Generated figures
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── Makefile                   # Build automation
```

### 13.2 Documentation Standards

#### 13.2.1 Code Documentation
```python
def compute_anomaly_score(
    gradient: torch.Tensor,
    profile: ClientProfile,
    params: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute multi-dimensional anomaly score for a client gradient.
    
    This function implements the CAAC-FL anomaly detection mechanism
    combining magnitude, directional, and temporal components.
    
    Args:
        gradient: Client's gradient update for current round
        profile: Client's behavioral profile with historical statistics
        params: Hyperparameters including:
            - lambda_mag: Weight for magnitude anomaly (default: 0.4)
            - lambda_dir: Weight for directional anomaly (default: 0.4)
            - lambda_temp: Weight for temporal anomaly (default: 0.2)
            - window_size: Historical window for comparisons (default: 10)
    
    Returns:
        Tuple containing:
            - composite_score: Combined anomaly score (float)
            - components: Dictionary with individual anomaly components
    
    Raises:
        ValueError: If gradient dimension doesn't match profile history
        RuntimeError: If profile has insufficient history (< window_size)
    
    Example:
        >>> score, components = compute_anomaly_score(
        ...     gradient=torch.randn(100),
        ...     profile=client_profiles[0],
        ...     params={'lambda_mag': 0.4, 'lambda_dir': 0.4}
        ... )
        >>> print(f"Anomaly score: {score:.3f}")
    
    Note:
        Higher scores indicate more anomalous behavior. Scores > 2.0
        typically indicate Byzantine clients in our experiments.
    """
```

#### 13.2.2 Configuration Documentation
```yaml
# configs/default.yaml
# CAAC-FL Default Configuration
# Last Updated: 2024-01-15
# Authors: [Names]

# Experiment metadata
experiment:
  name: "default_experiment"
  description: |
    Default configuration for CAAC-FL experiments.
    Modify this file or create overrides for specific experiments.
  seed: 42  # Random seed for reproducibility
  
# Dataset configuration
dataset:
  name: "mimic"  # Options: mimic, chestxray, isic
  # Heterogeneity parameters
  heterogeneity:
    distribution: "dirichlet"  # How to split data
    alpha: 0.5  # Dirichlet parameter (lower = more heterogeneous)
    # 0.1: extreme heterogeneity
    # 0.5: moderate heterogeneity  
    # 1.0: mild heterogeneity
```

### 13.3 API Documentation

#### 13.3.1 Aggregator Interface
```python
class Aggregator(ABC):
    """Abstract base class for federated aggregation methods."""
    
    @abstractmethod
    def aggregate(
        self,
        gradients: Dict[int, torch.Tensor],
        round_t: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregate client gradients into global update.
        
        Args:
            gradients: Mapping from client_id to gradient tensor
            round_t: Current training round
            **kwargs: Method-specific parameters
            
        Returns:
            Aggregated gradient tensor
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state for new experiment."""
        pass
```

### 13.4 Testing Documentation

#### 13.4.1 Test Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for core CAAC-FL components
- All attacks must have validation tests

#### 13.4.2 Test Organization
```python
# tests/unit/test_caac_fl.py
class TestCAACFLAggregator:
    """Test suite for CAAC-FL aggregation method."""
    
    def test_anomaly_computation(self):
        """Test anomaly score computation correctness."""
        
    def test_threshold_adaptation(self):
        """Test adaptive threshold calculation."""
        
    def test_bootstrap_phase(self):
        """Test bootstrap phase behavior."""
        
    def test_new_client_handling(self):
        """Test handling of dynamically joining clients."""
```

### 13.5 Release Checklist

- [ ] All tests passing
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Results reproducible
- [ ] Data and models archived
- [ ] Version tagged
- [ ] CHANGELOG updated
- [ ] Citation information added

---

## 14. Limitations and Failure Modes

### 14.1 Known Limitations

#### 14.1.1 Algorithmic Limitations

1. **Slow Poisoning Vulnerability**
   - Attackers that gradually shift behavior may evade detection
   - Mitigation: Implement longer temporal windows and change-point detection

2. **Coordinated Attack Susceptibility**
   - Multiple Byzantine clients coordinating can manipulate statistics
   - Mitigation: Cross-validation of client behaviors, clustering analysis

3. **Cold Start Problem**
   - New clients lack behavioral history for profiling
   - Current solution (conservative defaults) may exclude legitimate outliers
   - Mitigation: Transfer learning from similar client profiles

4. **Hyperparameter Sensitivity**
   - Performance depends on proper tuning of λ weights and thresholds
   - Mitigation: Automated hyperparameter optimization, adaptive tuning

#### 14.1.2 Practical Limitations

1. **Scalability Constraints**
   - Memory grows linearly with clients and window size
   - May become prohibitive for thousands of clients
   - Mitigation: Hierarchical aggregation, profile compression

2. **Communication Overhead**
   - Additional statistics may increase bandwidth requirements
   - Mitigation: Quantization, periodic profile updates

3. **Computational Cost**
   - 2-3x overhead compared to FedAvg
   - May be significant for resource-constrained scenarios
   - Mitigation: Selective profiling, approximate computations

### 14.2 Failure Modes

#### 14.2.1 Detection Failures

1. **False Negatives**
   - **Scenario**: Adaptive attacker successfully mimics benign profile
   - **Impact**: Malicious updates corrupt global model
   - **Detection**: Monitor global model performance degradation
   - **Recovery**: Rollback to previous checkpoint, adjust sensitivity

2. **False Positives**
   - **Scenario**: Legitimate domain shift triggers anomaly detection
   - **Impact**: Valid updates excluded, slower convergence
   - **Detection**: Track participation rates and convergence speed
   - **Recovery**: Relax thresholds, extend adaptation period

#### 14.2.2 System Failures

1. **Profile Corruption**
   - **Scenario**: Attacker contaminates profile during bootstrap
   - **Impact**: Permanently skewed statistics
   - **Detection**: Sanity checks on profile parameters
   - **Recovery**: Profile reset with extended bootstrap

2. **Numerical Instability**
   - **Scenario**: Division by near-zero variance
   - **Impact**: NaN/Inf in computations
   - **Detection**: Numerical checks after each operation
   - **Recovery**: Fallback to default values, add stability epsilon

### 14.3 Edge Cases

1. **Single Client Remaining**
   - Cannot compute relative statistics
   - Fallback: Use absolute thresholds

2. **All Clients Byzantine**
   - No reliable baseline for comparison
   - Fallback: Terminate training, alert operator

3. **Extreme Heterogeneity**
   - All clients appear anomalous
   - Fallback: Increase bootstrap period, relax thresholds

4. **Model Divergence**
   - Aggregated updates cause model explosion
   - Fallback: Gradient clipping, learning rate reduction

### 14.4 Assumptions and Prerequisites

#### 14.4.1 Assumptions
1. Byzantine fraction < 50%
2. Gradients are informative (not random)
3. Client participation is semi-regular
4. Network is synchronous or semi-synchronous

#### 14.4.2 Prerequisites
1. Clients can compute and transmit gradients
2. Server has sufficient memory for profiles
3. Initial model is reasonable (not random)
4. Data has some common structure across clients

### 14.5 Ethical Considerations

1. **Privacy Risks**
   - Behavioral profiles could reveal information about client data
   - Mitigation: Differential privacy on profiles

2. **Fairness Concerns**
   - May systematically disadvantage minority data distributions
   - Mitigation: Fairness-aware thresholding

3. **Transparency**
   - Clients cannot inspect why they're flagged as anomalous
   - Mitigation: Provide anomaly reports to clients

---

## Appendix A: Mathematical Formulations

### A.1 Complete CAAC-FL Algorithm

```
Algorithm 1: CAAC-FL Aggregation
─────────────────────────────────────────────────────
Input: Gradients {g_i^t}, round t, profiles {P_i}
Output: Aggregated gradient Δw^t
Parameters: β, γ, λ_mag, λ_dir, λ_temp, τ_anomaly, W

1: if t ≤ T_bootstrap then
2:    return BootstrapAggregation({g_i^t})
3: end if

4: // Update profiles
5: for each client i with gradient g_i^t do
6:    μ_i^t ← β·μ_i^(t-1) + (1-β)·‖g_i^t‖
7:    σ_i^t ← √(β·(σ_i^(t-1))² + (1-β)·(‖g_i^t‖ - μ_i^t)²)
8:    ρ_i^t ← β·ρ_i^(t-1) + (1-β)·cos(g_i^t, ḡ^(t-1))
9: end for

10: // Compute anomaly scores
11: for each client i do
12:    A_mag^i ← |‖g_i^t‖ - μ_i^(t-1)| / (σ_i^(t-1) + ε)
13:    A_dir^i ← 1 - (1/W)·Σ_k cos(g_i^t, g_i^k)
14:    A_temp^i ← |σ_i^t - σ_i^(t-W)| / (σ_i^(t-W) + ε)
15:    A_i^t ← √(λ_mag·(A_mag^i)² + λ_dir·(A_dir^i)² + λ_temp·(A_temp^i)²)
16: end for

17: // Update reliability scores  
18: for each client i do
19:    R_i^t ← γ·𝟙(A_i^t < τ_anomaly) + (1-γ)·R_i^(t-1)
20: end for

21: // Compute adaptive thresholds
22: μ_global ← median({‖g_i^t‖})
23: for each client i do
24:    f_i ← min(2, max(0.1, exp(-A_i^t/2)·(1 + R_i^t)))
25:    τ_i^t ← μ_global · f_i
26: end for

27: // Clip gradients
28: for each client i do
29:    g̃_i^t ← g_i^t · min(1, τ_i^t/‖g_i^t‖)
30: end for

31: // Compute weights
32: for each client i do
33:    w_i^t ← (R_i^t · exp(-A_i^t/4)) / Σ_j(R_j^t · exp(-A_j^t/4))
34: end for

35: // Aggregate
36: Δw^t ← Σ_i w_i^t · g̃_i^t

37: return Δw^t
─────────────────────────────────────────────────────
```

### A.2 Attack Formulations

#### A.2.1 ALIE Attack
```
z_j = μ_j + σ_j · Φ^(-1)((n-f-s)/(n-f))
```
where:
- μ_j, σ_j: mean and std of coordinate j across benign clients
- Φ^(-1): inverse CDF of standard normal
- n: total clients, f: Byzantine clients, s: tolerance parameter

#### A.2.2 Inner Product Manipulation
```
g_mal = -ε · (Σ_i∈B g_i) / ‖Σ_i∈B g_i‖
```
where:
- B: set of benign clients
- ε: perturbation magnitude

#### A.2.3 Adaptive Profile-Aware Attack
```
g_adv^t = argmin_g L_task(g) + λ_stealth · D_profile(g, P_target)
```
where:
- L_task: adversarial objective
- D_profile: distance to target profile statistics
- λ_stealth: stealth regularization weight

### A.3 Statistical Formulas

#### A.3.1 Dirichlet Data Distribution
```
p_i,k ~ Dir(α·1_K)
n_i,k ~ Multinomial(n_i, p_i,k)
```
where:
- p_i,k: probability of class k at client i
- α: concentration parameter
- n_i,k: number of samples of class k at client i

#### A.3.2 Power-Law Client Sizes
```
n_i = n_min · (i/N)^(-α_size)
```
where:
- n_i: samples at client i
- α_size: power law exponent (typically 0.5-1.5)

---

## Appendix B: Hyperparameter Settings

### B.1 CAAC-FL Default Parameters

| Parameter | Symbol | Default Value | Range | Description |
|-----------|--------|---------------|-------|-------------|
| EWMA decay | β | 0.9 | [0.8, 0.95] | Profile smoothing factor |
| Reliability decay | γ | 0.1 | [0.05, 0.2] | Trust update rate |
| Magnitude weight | λ_mag | 0.4 | [0.2, 0.6] | Norm anomaly weight |
| Direction weight | λ_dir | 0.4 | [0.2, 0.6] | Angle anomaly weight |
| Temporal weight | λ_temp | 0.2 | [0.1, 0.3] | Variance anomaly weight |
| Anomaly threshold | τ_anomaly | 2.0 | [1.5, 3.0] | Anomaly cutoff |
| Window size | W | 10 | [5, 20] | History window |
| Bootstrap rounds | T_bootstrap | 20 | [10, 30] | Initial phase length |
| Stability epsilon | ε | 1e-8 | - | Numerical stability |

### B.2 Baseline Method Parameters

#### B.2.1 Trimmed Mean
- Trim fraction: 0.1 (trim 10% from each tail)

#### B.2.2 Krum
- Selection count: n - f - 2 (n=clients, f=Byzantine)

#### B.2.3 FLTrust
- Root dataset size: 100 samples
- Trust score threshold: 0.1

#### B.2.4 LASA
- Sparsification ratio: 0.1
- Layer-wise adaptive: True

### B.3 Attack Parameters

| Attack | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| Random Noise | σ | 1.0 | Noise standard deviation |
| Sign-Flip | γ | 10 | Amplification factor |
| ALIE | s | 1 | Tolerance parameter |
| IPM | ε | 0.1 | Perturbation magnitude |
| Slow-Drift | T_drift | 30 | Drift duration (rounds) |
| Adaptive | λ_stealth | 0.5 | Stealth regularization |

---

## Appendix C: Dataset Specifications

### C.1 MIMIC-III Processing

```python
def preprocess_mimic(raw_data_path):
    """
    MIMIC-III preprocessing for mortality prediction.
    
    Cohort definition:
    - Adults (age ≥ 18) 
    - First ICU stay
    - Minimum 24 hours of data
    
    Features (48 total):
    - Demographics: age, gender, ethnicity (3)
    - Vital signs: HR, BP, RR, SpO2, Temp (5)
    - Lab values: 25 common tests
    - Clinical scores: SOFA, SAPS-II, GCS (3)
    - Medications: 12 indicator variables
    
    Target:
    - In-hospital mortality (binary)
    """
```

### C.2 ChestX-ray8 Configuration

```python
CHEST_XRAY_DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 
    'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly',
    'Nodule', 'Mass', 'Hernia'
]

IMAGE_SIZE = (224, 224)
NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
```

### C.3 ISIC 2019 Labels

```python
ISIC_CATEGORIES = {
    0: 'MEL',    # Melanoma
    1: 'NV',     # Melanocytic nevus
    2: 'BCC',    # Basal cell carcinoma
    3: 'AK',     # Actinic keratosis
    4: 'BKL',    # Benign keratosis
    5: 'DF',     # Dermatofibroma
    6: 'VASC',   # Vascular lesion
    7: 'SCC',    # Squamous cell carcinoma
    8: 'UNK'     # Unknown
}

# Binary task
BINARY_MAPPING = {
    'MEL': 1,    # Malignant
    'Others': 0  # Benign
}
```

---

## Appendix D: Reproducibility Checklist

### D.1 Environment Setup

```bash
# Create conda environment
conda create -n caac_fl python=3.9
conda activate caac_fl

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Download datasets
python scripts/download_data.py --dataset all

# Preprocess data
python scripts/preprocess_data.py --config configs/preprocessing.yaml

# Run tests
pytest tests/ -v --cov=src --cov-report=html
```

### D.2 Running Experiments

```bash
# Single experiment
python scripts/run_experiment.py \
    --config configs/experiments/caac_fl_mimic.yaml \
    --seed 42 \
    --gpu 0

# Hyperparameter sweep
python scripts/run_sweep.py \
    --config configs/sweeps/caac_fl_hyperparam.yaml \
    --num_workers 4

# Reproduce all paper results
bash scripts/reproduce_all.sh
```

### D.3 Result Validation

```python
# Verify results match paper
python scripts/validate_results.py \
    --results_dir results/paper_results/ \
    --tolerance 0.01

# Generate paper figures
python notebooks/figures/generate_all.py \
    --results results/paper_results/ \
    --output figures/
```

---

## Appendix E: Troubleshooting Guide

### E.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| OOM Error | CUDA out of memory | Reduce batch size, use gradient checkpointing |
| Slow convergence | Loss plateaus early | Increase learning rate, check data loading |
| NaN in gradients | Training crashes | Add gradient clipping, check for division by zero |
| Profile corruption | Anomaly scores explode | Reset profiles, extend bootstrap phase |
| Poor Byzantine detection | High FNR | Tune anomaly threshold, adjust weights |

### E.2 Debugging Commands

```python
# Enable debug logging
export CAAC_FL_DEBUG=1

# Profile performance
python -m cProfile -o profile.stats scripts/run_experiment.py

# Memory profiling
python -m memory_profiler scripts/run_experiment.py

# Check gradient statistics
python scripts/debug/check_gradients.py --checkpoint path/to/ckpt
```

### E.3 Contact and Support

- GitHub Issues: https://github.com/[username]/caac-fl/issues
- Email: [research team emails]
- Documentation: https://[username].github.io/caac-fl/

---

**End of Protocol**

_This protocol provides comprehensive implementation details for the CAAC-FL research project. All sections should be reviewed and updated as the implementation progresses._
