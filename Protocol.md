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
(similarly for variance and directional consistency), where \(\beta \in [0, 1)\) is a decay factor (]()
