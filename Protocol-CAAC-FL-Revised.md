# Experimental Protocol  
**Title:** Evaluating CAAC-FL: Client-Adaptive Anomaly-Aware Clipping for Byzantine-Robust Federated Learning under Heterogeneous Healthcare Data  

_Last updated: YYYY-MM-DD_  

---

## 0. High-Level Overview

This protocol specifies the experimental design, implementation details, and analysis plan for evaluating **CAAC-FL** against existing Byzantine-robust federated learning (FL) methods on heterogeneous healthcare data.

Key goals:

1. Provide a **feasible** experimental matrix (Core, Extended, Ablation tiers).  
2. Define a **single, canonical CAAC-FL algorithm** without conflicting variants.  
3. Make the **threat model** and attacker capabilities explicit.  
4. Specify **bootstrap logic** and temporal behavior assumptions.  
5. Give a **clear implementation roadmap** for a PhD student or research assistant.

---

## 1. Objectives and Hypotheses

### 1.1 Objectives

1. **Primary Objective**  
   Quantitatively evaluate the performance and robustness of **CAAC-FL** compared to existing Byzantine-robust FL aggregation methods under **non-IID, heterogeneous healthcare data**.

2. **Secondary Objectives**
   - Measure how well CAAC-FL **preserves contributions from benign but systematically different clients** (e.g., hospitals with atypical label distributions).
   - Assess CAAC-FL’s resilience under **multiple attack types** (magnitude-based, direction-based, temporal/adaptive).
   - Characterize **temporal behavior** of anomaly scores and clipping thresholds (detection latency, false alarms).
   - Quantify **computational overhead** of CAAC-FL relative to strong baselines.

### 1.2 Hypotheses

- **H1 – Heterogeneity Preservation**  
  Under high client heterogeneity and *no attack*, CAAC-FL:
  - Achieves equal or better global performance (accuracy / AUROC / AUPRC) than existing robust methods; and  
  - Exhibits **lower benign false positive rate (FPR)** (fewer benign clients heavily clipped or effectively ignored).

- **H2 – Multi-Dimensional Robustness**  
  Against a range of Byzantine attacks (e.g., sign-flipping, ALIE-style, inner-product–style), CAAC-FL’s **multi-dimensional anomaly detection** (norm, direction, temporal drift) yields:
  - Smaller degradation in global metrics, and  
  - Lower false negative rate (FNR) for malicious clients, compared with single-dimension or non-temporal baselines.

- **H3 – Temporal Discrimination**  
  CAAC-FL better distinguishes:
  - Abrupt, malicious changes in client behavior, from  
  - Slow benign domain drift (e.g., evolving institutional case mix),  
  yielding **faster detection of abrupt attacks** and **fewer false alarms** under benign drift.

---

## 2. Threat Model

### 2.1 System Model

- Central FL **server** coordinates training and aggregates updates.
- Multiple **clients** (e.g., hospitals) each hold private datasets and train locally.
- Training proceeds in **synchronous rounds**: server broadcasts global model, clients return updates.

### 2.2 Adversaries

- A fraction of clients are **Byzantine** (malicious or compromised).
- Byzantine clients can arbitrarily manipulate the update they send.

### 2.3 Attacker Knowledge and Capabilities

- **Attackers know**:
  - The global model parameters they receive each round.
  - Their own local data, gradients, and update history.
- **Attackers do not know**:
  - Other clients’ raw gradients or local data.
  - The server’s internal per-client profiles (they may approximate from their own history, but have no direct access).
- **Capabilities**:
  - Replace their gradient with any tensor (full control over their update).
  - Coordinate with other Byzantine clients (correlated attacks).

### 2.4 Attack Types (Used in Experiments)

Core attacks:

1. **Sign-Flipping** – invert and scale gradients to destabilize training.  
2. **ALIE-style** – coordinate-wise manipulations to evade coordinate-based defenses.  
3. **Slow-Drift Poisoning** – gradually shift update direction over many rounds.

Optional extended attacks:

4. **Random Noise** – replace gradient with Gaussian noise.  
5. **Inner-Product–style attack** – maintain norm but steer direction adversarially.  
6. **Profile-aware heuristic attack** – approximate CAAC-FL’s behavior by keeping norm and direction within typical ranges while introducing subtle bias.

---

## 3. Experimental Factors and Design

### 3.1 Independent Variables

1. **Aggregation Method (Factor A)**  

   **Core Methods**
   - A1: FedAvg (non-robust baseline)  
   - A2: Krum (distance-based robust aggregation)  
   - A3: FLTrust (trust bootstrapping)  
   - A4: CAAC-FL (proposed method)  

   **Extended Methods (Optional)**
   - A5: Coordinate-wise Median  
   - A6: Trimmed Mean  
   - A7: RFA (Geometric median)  
   - A8: LASA  

2. **Data Heterogeneity (Factor B – Dirichlet α)**  
   - B1: Mild non-IID — α = 1.0  
   - B2: Extreme non-IID — α = 0.1  

3. **Byzantine Proportion (Factor C)**  
   - C1: 0% (clean training)  
   - C2: 20% Byzantine clients  

4. **Attack Type (Factor D)**  
   - D1: None (clean)  
   - D2: Sign-Flipping  
   - D3: ALIE-style  
   - D4: Slow-Drift (for temporal discrimination)  

5. **Dataset / Task (Factor E)**  
   - E1: MIMIC-III – ICU mortality (binary classification)  
   - E2: ISIC 2019 – melanoma vs non-melanoma (binary classification)  
   - **Extended:** ChestX-ray8 – multi-label thoracic disease (E3).

6. **Number of Clients (Factor F)**  
   - F1: 20 clients (core)  
   - **Extended:** 50 clients (scaling).

### 3.2 Dependent Variables

- **Global performance**:
  - AUROC, AUPRC, accuracy, F1 (dataset-dependent).  
- **Robustness**:
  - Performance degradation under attack vs clean baseline (ΔAUROC, ΔAUPRC).  
- **Client-level detection (for CAAC-FL)**:
  - Benign FPR: fraction of benign clients flagged as anomalous.  
  - Malicious TPR/FNR: fraction of malicious clients correctly/incorrectly flagged.  
  - Detection latency: rounds from attack onset to reliable detection.  
- **Temporal behavior**:
  - Stability of anomaly scores and thresholds for benign clients.  
- **Overhead**:
  - Per-round server time and memory overhead vs FedAvg.

---

## 4. Feasible Experimental Matrix

To keep the project tractable, we define **Core**, **Extended**, and **Ablation** sets.

### 4.1 Core Experiments (Minimum for Paper)

For each dataset \(E \in \{	ext{MIMIC-III}, 	ext{ISIC 2019}\}\):

1. **Heterogeneity Preservation (H1, Clean)**  
   - Factors:
     - A ∈ {FedAvg, Krum, FLTrust, CAAC-FL}  
     - B ∈ {1.0, 0.1}  
     - C = 0%  
     - D = None  
   - Seeds: 3 per combination.

2. **Robustness (H2) under Extreme Non-IID**  
   - A ∈ {FedAvg, Krum, FLTrust, CAAC-FL}  
   - B = 0.1  
   - C = 20%  
   - D ∈ {Sign-Flipping, ALIE}  
   - Seeds: 3 per combination.

3. **Temporal Discrimination (H3)**  
   - A ∈ {FedAvg, FLTrust, CAAC-FL}  
   - B = 0.1  
   - C = 20%  
   - D = Slow-Drift poisoning  
   - Seeds: 3 per combination.

This defines a compact yet hypothesis-aligned core matrix.

### 4.2 Extended Experiments (Optional)

- Add ChestX-ray8 (E3).  
- Add additional robust aggregators (Median, Trimmed Mean, RFA, LASA).  
- Add profile-aware heuristic attack.  
- Increase seeds to 5 for key configurations.

### 4.3 Ablation Experiments (CAAC-FL Components)

For MIMIC-III and ISIC:

- Methods:
  - CAAC-MAG: magnitude anomaly only.  
  - CAAC-DIR: direction anomaly only.  
  - CAAC-TEMP: temporal anomaly only.  
  - CAAC-MAG+DIR: magnitude + direction.  
  - CAAC-FULL: full CAAC-FL (norm + direction + temporal).  

- B = 0.1, C = 20%, D ∈ {Sign-Flipping, ALIE}.  
- Seeds: 3 per combination.

---

## 5. Materials and Data Preparation

### 5.1 Software

- Python 3.9+  
- FL framework: **Flower** (recommended)  
- Deep learning: **PyTorch**  
- Experiment tracking: Weights & Biases or MLflow  
- Environment and dependencies: `requirements.txt` (PyTorch, torchvision, numpy, scipy, etc.)  
- Version control: Git

### 5.2 Hardware

- 1 GPU with 8–16 GB VRAM  
- 32–64 GB RAM  
- SSD storage (≥500 GB)

### 5.3 Data Preprocessing

#### 5.3.1 MIMIC-III

- Task: in-hospital mortality (binary).  
- Steps:
  - Define cohort (adult ICU, first ICU stay).  
  - Extract features (demographics, vitals, labs, scores).  
  - Impute missing values; standardize/normalize.  
  - Split into train/val/test (e.g., 70/10/20) at patient level.

#### 5.3.2 ISIC 2019

- Task: melanoma vs non-melanoma (binary).  
- Steps:
  - Map lesion labels to malignant vs benign.  
  - Resize images (e.g., 224×224), normalize using ImageNet statistics.  
  - Train/val/test split (stratified if possible).

#### 5.3.3 ChestX-ray8 (Extended)

- Task: multi-label disease classification.  
- Steps:
  - Load CXR images and multi-label targets.  
  - Resize, normalize as above.  
  - Train/val/test split.

> All preprocessing pipelines must be scripted and checked into the repository.

---

## 6. Model Architectures

For each dataset, **fix a single architecture** used across all methods; only aggregation changes.

### 6.1 MIMIC-III

- **Option A (tabular)**: MLP over engineered features.  
- **Option B (temporal)**: GRU or 1D CNN over time series.  

Output: single logit + sigmoid.

### 6.2 ISIC 2019

- CNN backbone: ResNet-18 or EfficientNet-B0.  
- Output: single logit + sigmoid.

### 6.3 Training Hyperparameters (Baseline)

- Optimizer: Adam  
- Learning rate: 1e-3  
- Batch size: 32  
- Local epochs per round: 1–5 (fix per dataset)  
- Total FL rounds: 100–150  
- Learning rate schedule: optional cosine decay or step decay.

---

## 7. Federated Learning Setup

### 7.1 Client Partitioning

- Number of clients: 20 (core).  
- Partition training data using Dirichlet over labels:  
  - α ∈ {1.0, 0.1}.  
- Introduce client data size imbalance via a power-law distribution (few large clients, many small).

### 7.2 Training Loop (Per Round)

For each round \(t = 1, \dots, T\):

1. **Server broadcast**: send global model \(w^t\) to all clients.  
2. **Client update** (for each client \(i\)):
   - Receive \(w^t\).  
   - Train locally for E epochs.  
   - Compute update: \(g_i^t = w_i^t - w^t\).  
   - If client is Byzantine, transform \(g_i^t\) using its attack strategy.  
   - Send \(g_i^t\) to server.  
3. **Server aggregation**:
   - Apply chosen aggregation rule (FedAvg, Krum, FLTrust, CAAC-FL).  
4. **Global update**:
   - \(w^{t+1} = w^t + \eta \Delta w^t\).  
5. **Evaluation** (every k rounds):
   - Evaluate \(w^{t+1}\) on central validation set.  
   - Log metrics and CAAC-FL diagnostics.

Core experiments assume **full participation** (all clients each round) for simplicity.

---

## 8. Canonical CAAC-FL Algorithm

This section defines a **single, non-conflicting CAAC-FL specification**.

### 8.1 Notation

- \(g_i^t\): update from client \(i\) at round \(t\).  
- \(v_i^t = 	ext{flatten}(g_i^t)\).  
- \(n_i^t = \|v_i^t\|_2\): gradient norm.  
- \(ar{g}^{t-1}\): previous aggregated update.  
- \(ar{v}^{t-1} = 	ext{flatten}(ar{g}^{t-1})\).

### 8.2 Per-Client Profile \(P_i\)

Each client \(i\) has:

- \(\mu_i\): EWMA of gradient norm.  
- \(\sigma_i\): EWMA of norm standard deviation.  
- \(ho_i\): EWMA of cosine similarity vs previous global gradient.  
- \(R_i\): reliability score in [0, 1].  
- `rounds_seen_i`: number of rounds participated.

Initialization:

- \(\mu_i = \mu_{	ext{init}}\) (e.g., median norm after bootstrap).  
- \(\sigma_i = \sigma_{	ext{init}} > 0\).  
- \(ho_i = 0\).  
- \(R_i = 0.5\).

Hyperparameters (canonical defaults):

- EWMA decay: \(eta = 0.9\).  
- Reliability smoothing: \(\gamma = 0.1\).  
- Anomaly weights: \(\lambda_{mag} = 0.4\), \(\lambda_{dir} = 0.4\), \(\lambda_{temp} = 0.2\).  
- Anomaly threshold: \(	au_{anom} = 2.0\).  
- Threshold shaping: \(f_{min} = 0.25\), \(f_{max} = 2.0\), α = 0.5, δ = 0.5.  
- Weight scaling: \(eta_w = 0.5\).

### 8.3 Bootstrap Phase

Goal: avoid early profile contamination.

- Bootstrap rounds: \(1 \le t \le T_{boot}\), with \(T_{boot} = 10\).  
- Aggregation: global norm clipping + FedAvg:
  - Use a fixed global clipping threshold for all clients.  
- Profiles:
  - Update \(\mu_i, \sigma_i\) from observed norms.  
  - Optionally update \(ho_i\) from cosine similarity.  
  - Keep \(R_i\) near 0.5 (or update very weakly).

Termination:

- Must have \(t \ge T_{boot}\).  
- Preferably, each client has participated in ≥50% of rounds.  
- If not satisfied by \(2T_{boot}\), proceed anyway.

From round \(t > T_{boot}\), use full CAAC-FL.

### 8.4 Anomaly Computation (for \(t > T_{boot}\))

Given update \(g_i^t\):

1. **Compute norm and cosine**:

   - Flatten \(g_i^t\): \(v_i^t = 	ext{flatten}(g_i^t)\).  
   - \(n_i^t = \|v_i^t\|_2\).  
   - If \(\|ar{v}^{t-1}\|_2 > 0\):

     \[
     c_i^t = rac{\langle v_i^t, ar{v}^{t-1} angle}{\|v_i^t\|_2 \|ar{v}^{t-1}\|_2}
     \]

     else set \(c_i^t = 1.0\).

2. **Update statistics (compute new values)**:

   Using previous \(\mu_i, \sigma_i, ho_i\):

   - \(\mu_i^{new} = eta \mu_i + (1 - eta) n_i^t\).  
   - \(\sigma_i^{new} = \sqrt{eta \sigma_i^2 + (1 - eta) (n_i^t - \mu_i^{new})^2}\).  
   - \(ho_i^{new} = eta ho_i + (1 - eta) c_i^t\).

3. **Anomaly components (using old stats)**:

   - **Magnitude anomaly**:

     \[
     A_{mag}^i = rac{|n_i^t - \mu_i|}{\sigma_i + \epsilon}
     \]

   - **Direction anomaly** (drop in cosine):

     \[
     A_{dir}^i = \max(0, ho_i - c_i^t)
     \]

   - **Temporal anomaly** (change in EWMA norm scale):

     \[
     A_{temp}^i = rac{|\mu_i^{new} - \mu_i|}{\mu_i + \epsilon}
     \]
     (Set \(A_{temp}^i = 0\) if \(\mu_i = 0\).)

4. **Composite anomaly**:

   \[
   A_i^t = \sqrt{
      \lambda_{mag} (A_{mag}^i)^2 +
      \lambda_{dir} (A_{dir}^i)^2 +
      \lambda_{temp} (A_{temp}^i)^2
   }
   \]

5. **Update profile**:

   - \(\mu_i \leftarrow \mu_i^{new}\), \(\sigma_i \leftarrow \sigma_i^{new}\), \(ho_i \leftarrow ho_i^{new}\).  
   - `rounds_seen_i += 1`.

### 8.5 Reliability Update

Reliability is an EWMA of “benign behavior” indicators:

\[
R_i^{new} = (1 - \gamma) R_i + \gamma \cdot \mathbf{1}(A_i^t < 	au_{anom})
\]

- If anomaly is below threshold, event = 1 (evidence of benign behavior).  
- Else, event = 0.

Clip \(R_i^{new}\) into [0, 1] and set \(R_i \leftarrow R_i^{new}\).

### 8.6 Threshold and Clipping

Compute global median norm:

\[
\mu_{global}^t = 	ext{median}_j (n_j^t)
\]

Define scale factor:

\[
s_i^t = \exp(-lpha A_i^t) \cdot (1 + \delta R_i)
\]

Clamp:

\[
s_i^t \leftarrow \min(f_{max}, \max(f_{min}, s_i^t))
\]

Client-specific threshold:

\[
	au_i^t = \mu_{global}^t \cdot s_i^t
\]

Clip gradient:

\[
	ilde{g}_i^t =
egin{cases}
g_i^t \cdot rac{	au_i^t}{n_i^t + \epsilon}, & 	ext{if } n_i^t > 	au_i^t \
g_i^t, & 	ext{otherwise}
\end{cases}
\]

### 8.7 Aggregation Weights and Update

Define raw weights:

\[
\omega_i^t = R_i \cdot \exp(-eta_w A_i^t)
\]

Normalize:

\[
w_i^t = rac{\omega_i^t}{\sum_j \omega_j^t + \epsilon}
\]

Aggregate:

\[
\Delta w^t = \sum_i w_i^t 	ilde{g}_i^t
\]

Update global model:

\[
w^{t+1} = w^t + \eta \Delta w^t
\]

---

## 9. Attack Implementations (Concrete)

### 9.1 Sign-Flipping

```python
def sign_flip_attack(gradient, scale=10.0):
    return -scale * gradient
```

### 9.2 ALIE-Style Attack (Simplified)

Within a round, using all submitted gradients \(\{g_j^t\}\):

1. Compute coordinate-wise mean and std:

```python
stack = torch.stack([g.view(-1) for g in gradients])  # shape: [N, D]
mu = stack.mean(dim=0)
sigma = stack.std(dim=0) + 1e-8
```

2. Byzantine client sets:

```python
def alie_attack(mu, sigma, k=2.5):
    return (mu + k * sigma).view_as_original_shape()
```

(Optionally project to match typical norm range.)

### 9.3 Slow-Drift Poisoning

```python
def slow_drift_attack(benign_grad, adv_direction, t, t_start, t_end):
    '''
    adv_direction: normalized tensor indicating adversarial direction.
    t: current round
    t_start: round where drift begins
    t_end: round where drift reaches full strength
    '''
    if t < t_start:
        return benign_grad
    alpha = min(1.0, max(0.0, (t - t_start) / (t_end - t_start)))
    n = benign_grad.norm(p=2)
    adv = adv_direction * n
    return (1 - alpha) * benign_grad + alpha * adv
```

Attackers approximate `adv_direction` from prior gradients or domain-specific objectives.

---

## 10. Evaluation Metrics and Analysis

### 10.1 Global Performance

For each configuration (dataset × method × condition):

- **Binary tasks** (MIMIC-III, ISIC):
  - AUROC, AUPRC, accuracy, F1 on held-out test set.  
- **Convergence**:
  - Metric vs round curves (AUROC vs round).  
  - Rounds to reach 90% of best clean AUROC.

### 10.2 Robustness Metrics

For each method:

\[
\Delta 	ext{AUROC} = 	ext{AUROC}_{	ext{clean}} - 	ext{AUROC}_{	ext{attack}}
\]

- Compute for each attack type and heterogeneity level.  
- Compare across methods for H2.

### 10.3 Client-Level Detection Metrics (CAAC-FL)

Given ground-truth benign/Byzantine labels:

- **Benign FPR**:

  \[
  	ext{FPR} = rac{\#\{i 	ext{ benign}: A_i^t \ge 	au_{anom}\}}{\#	ext{benign}}
  \]

- **Malicious TPR**:

  \[
  	ext{TPR} = rac{\#\{i 	ext{ Byzantine}: A_i^t \ge 	au_{anom}\}}{\#	ext{Byzantine}}
  \]

- **Detection latency**:
  - Earliest round after attack onset where TPR ≥ 0.8 for 3 consecutive rounds.

Compute these per round and summarize (mean/median over a window).

### 10.4 H1-Specific Analyses

Under **no attack**:

- Compare AUROC and AUPRC across methods as α varies (1.0 vs 0.1).  
- For CAAC-FL:
  - Measure benign FPR under high heterogeneity.  
  - Optionally, measure per-client performance (test metrics per client’s domain) to ensure no systematic suppression of minority clients.

### 10.5 Statistical Analysis (Simplified)

- For each configuration:
  - Run 3 independent seeds.  
  - Report mean ± standard deviation for metrics.  
- For key pairwise comparisons (e.g., CAAC-FL vs FLTrust, CAAC-FL vs Krum):
  - Use non-parametric **Wilcoxon signed-rank test** on AUROC across seeds.  
  - Report p-values and effect sizes (difference of means).

This avoids stronger parametric assumptions and overly complex models (e.g., ANOVA, Cox models).

---

## 11. Ethical and Privacy Considerations

- **Data**:  
  - Use publicly released, de-identified datasets (MIMIC-III, ISIC, ChestX-ray8).  
  - No real federated deployment or protected health information (PHI) is involved.

- **Profiles and Leakage**:  
  - CAAC-FL maintains per-client statistics derived from gradients (norms, cosines).  
  - In real deployments, such statistics may leak information about client populations or behavior patterns.  
  - Proper deployments should consider:
    - Secure aggregation or encryption of updates.  
    - Differential privacy on profile statistics.  
    - Access control and logging for profile inspection.

- **Fairness**:  
  - Robust aggregation methods must not systematically penalize unusual yet benign hospitals (e.g., pediatric vs geriatric centers).  
  - Per-client outcome analysis should verify that CAAC-FL does not disproportionately degrade performance for minority distributions.

---

## 12. Implementation and Repository Structure

### 12.1 Suggested Repository Layout

```text
caac-fl/
├── README.md
├── Protocol.md
├── requirements.txt
├── src/
│   ├── aggregators/
│   │   ├── fedavg.py
│   │   ├── krum.py
│   │   ├── fltrust.py
│   │   └── caac_fl.py
│   ├── attacks/
│   │   ├── sign_flip.py
│   │   ├── alie.py
│   │   └── slow_drift.py
│   ├── datasets/
│   │   ├── mimic.py
│   │   └── isic.py
│   ├── models/
│   │   ├── mlp.py
│   │   └── resnet18.py
│   ├── fl_core/
│   │   ├── client.py
│   │   └── server.py
│   └── utils/
│       ├── metrics.py
│       └── logging.py
├── configs/
│   ├── core_mimic.yaml
│   ├── core_isic.yaml
│   └── ablation_mimic.yaml
├── scripts/
│   ├── run_experiment.py
│   └── analyze_results.py
└── notebooks/
    └── analysis_core.ipynb
```

This structure is **recommended, not mandatory**; it can be adapted.

### 12.2 Minimal Testing

- **Unit tests**:
  - CAAC-FL aggregator (anomaly computation, thresholds, clipping, weights).  
  - Attack modules (sign-flip, ALIE, slow-drift).  

- **Integration test**:
  - Tiny synthetic dataset with 5 clients, 10 rounds:
    - Confirm training runs end-to-end.  
    - Inspect anomaly scores and ensure sensible behavior.

---

## 13. Suggested Timeline (Semester-Scale)

- **Weeks 1–2**:  
  - Implement FL core, FedAvg, and MIMIC-III pipeline.  
  - Run basic sanity checks.

- **Weeks 3–4**:  
  - Implement Krum, FLTrust, canonical CAAC-FL.  
  - Add unit tests.

- **Weeks 5–6**:  
  - Implement attacks (sign-flip, ALIE, slow-drift).  
  - Validate on synthetic data.

- **Weeks 7–9**:  
  - Run **Core experiments** on MIMIC-III and ISIC.  
  - Log all metrics and diagnostics.

- **Weeks 10–12**:  
  - Run ablation experiments.  
  - Run selected Extended experiments if time/resources allow.

- **Weeks 13–15**:  
  - Perform analysis, generate plots/tables, and draft paper sections.  

---

## 14. Canonical CAAC-FL Summary (Quick Reference)

1. Maintain **per-client EWMA profiles** of norm, norm variance, cosine similarity, and reliability.  
2. Use a **bootstrap phase** (10–20 rounds) with uniform clipping and FedAvg to initialize profiles.  
3. From round \(t > T_{boot}\):  
   - Compute magnitude, direction, and temporal anomalies.  
   - Combine into a single anomaly score \(A_i^t\).  
   - Update reliability as an EWMA of low-anomaly events.  
   - Set per-client clipping thresholds using global median norm scaled by anomaly and reliability.  
   - Clip gradients accordingly.  
   - Weight clients by reliability × exp(−β_w × anomaly).  
   - Aggregate clipped updates via weighted average.

This canonical definition should be treated as the **ground truth** for all implementations and experiments.

---

**End of Protocol**
