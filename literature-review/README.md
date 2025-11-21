# Literature Review for CAAC-FL

This directory contains comprehensive analysis of reference papers cited in the CAAC-FL (Client-Adaptive Anomaly-Aware Clipping for Federated Learning) work in progress paper.

## Contents

### Main Document
- **reference-analysis-summary.md** (47KB, 1,237 lines) - Comprehensive analysis of all 9 reference papers

## Papers Analyzed

1. **Yin et al. (2018)** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates
   - Coordinate-wise median and trimmed mean algorithms
   - Theoretical optimality proofs
   - Label flipping attacks

2. **Blanchard et al. (2017)** - Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
   - Krum algorithm (provably Byzantine-resilient)
   - Multi-Krum variant
   - Omniscient and Gaussian attacks

3. **Baruch et al. (2019)** - A Little Is Enough: Circumventing Defenses For Distributed Learning
   - ALIE attack (operates within variance bounds)
   - Circumvents major defenses
   - Backdoor attack variant

4. **Cao et al. (2021)** - FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping
   - Trust bootstrapping with root dataset
   - ReLU-clipped cosine similarity
   - Adaptive attack framework

5. **Pillutla et al. (2022)** - Robust Aggregation for Federated Learning (RFA)
   - Geometric median aggregation
   - Smoothed Weiszfeld algorithm
   - Privacy-communication-robustness tradeoff

6. **Werner et al. (2023)** - Provably Personalized and Robust Federated Learning
   - Theoretical proof: client-specific adaptation enhances robustness
   - Gradient-based clustering
   - Near-optimal convergence rates

7. **Xu et al. (2024)** - LASA: Layer-Adaptive Sparsified Model Aggregation
   - Layer-wise adaptive aggregation
   - Top-k sparsification
   - Positive Direction Purity (PDP) metric

8. **Li et al. (2024)** - An Experimental Study of Byzantine-Robust Aggregation Schemes
   - **Key finding:** <10% accuracy on non-IID data even without attacks
   - Comprehensive comparison of 8 aggregation schemes
   - ClippedClustering (best performer)

9. **Le and Moothedath (2025)** - Byzantine Resilient Federated Multi-Task Representation Learning
   - Multi-task learning with shared representation + client-specific heads
   - Asymmetric update frequency
   - Geometric Median and Krum aggregation

## Key Insights

### The Non-IID Challenge
- Li et al. demonstrates existing defenses fail completely on non-IID data
- Distance-based schemes (Krum, GeoMed): 10% accuracy
- Only mean-based with adaptive mechanisms survive (~80%)

### Attack Landscape
- **ALIE** (Baruch et al.): Operates within variance, defeats statistical defenses
- **IPM** (Li et al.): Stealthy vs. aggressive variants
- **Adaptive attacks** (Cao et al.): Optimized against specific defenses

### Theoretical Foundation
- Werner et al. proves client-specific adaptation enhances Byzantine robustness
- Gradient-based clustering achieves near-optimal convergence: O(1/√niT)
- Byzantine error bounded: βiσΔ (graceful degradation)

## CAAC-FL Experimental Protocol Recommendations

### Datasets
- **Primary:** MIMIC-III, ChestX-ray8, ISIC 2019 (healthcare)
- **Benchmark:** CIFAR-10, FEMNIST (comparison to baselines)

### Federation Setup
- **Clients:** 20 (planned)
- **Byzantine fractions:** 20%, 40% (core), 10%, 30%, 60% (extended)
- **Data distribution:** Dirichlet α=0.5 (planned), α=0.1 (high heterogeneity)
- **Power-law dataset sizes** (CAAC-FL unique contribution)

### Attack Models
1. ALIE (primary threat)
2. IPM (ε=0.5 stealthy, ε=100 aggressive)
3. Label flipping
4. Slow-drift poisoning
5. Backdoor with medical triggers

### Baselines
- FedAvg, Krum, Trimmed Mean, FLTrust, LASA, ClippedClustering, BR-MTRL

### Critical Metrics
- **Performance:** Test accuracy (target: >80% on non-IID without attacks)
- **Detection:** TPR>90%, FPR<10%, latency<10 rounds
- **Robustness:** Maintain >70% under 20-40% Byzantine clients

## Implementation Priorities

### High Priority
1. ALIE attack implementation (formula in summary)
2. Non-IID Dirichlet partitioning (α ∈ {0.1, 0.5})
3. Comparison against ClippedClustering
4. Behavioral profile tracking (code templates provided)
5. FLTrust baseline with root dataset

### Medium Priority
1. Test on FEMNIST (natural non-IID)
2. Adaptive attack framework from Cao et al.
3. Pairwise cosine similarity analysis
4. Byzantine fraction sensitivity up to 60%
5. AWS federated testbed deployment

### Lower Priority
1. Theoretical convergence analysis
2. Privacy analysis (differential privacy bounds)
3. Scalability to 1000+ clients
4. Communication compression techniques

## Document Structure

The main analysis document contains:

1. **Main Paper Overview** - CAAC-FL summary
2. **Individual Paper Analyses (9)** - Each with:
   - Key contributions
   - Attack models
   - Experimental settings
   - Evaluation metrics
   - Repurposable design elements
3. **Comprehensive Summary** - Cross-paper insights including:
   - Consolidated experimental protocol
   - Implementation code snippets
   - Comparison tables
   - Critical validations needed
4. **Final Recommendations** - Prioritized action items

## Usage

This literature review provides:
- Experimental protocol templates
- Attack implementation formulas
- Baseline comparison frameworks
- Evaluation metric definitions
- Code snippets for common components

All recommendations are evidence-based, citing specific papers and experimental results.

---

**Last Updated:** November 19, 2025
**Total Analysis:** 9 papers, 1,237 lines
**Coverage:** Attack models, experimental protocols, theoretical foundations, implementation guidelines
