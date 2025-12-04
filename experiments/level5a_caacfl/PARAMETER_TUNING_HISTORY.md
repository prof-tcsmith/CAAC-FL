# CAAC-FL Parameter Tuning History

## Version 3 (Current) - Delayed Compromise Threat Model - 2025-12-04

### Threat Model Change

**Problem Identified with v2:**
- Even with tuned parameters, random noise attack still evades detection
- Root cause: Byzantine clients attack from round 0, establishing malicious behavior as baseline
- Profile-based detection fails because "anomaly" is the client's normal pattern

**Solution: Delayed Compromise Threat Model**
- Byzantine clients behave honestly during warmup (rounds 0 to compromise_round-1)
- CAAC-FL builds profiles based on legitimate behavior
- At compromise_round, Byzantine clients begin attacking
- Behavioral deviation from honest profile should trigger detection

**Configuration:**
- `compromise_round = 15` (default, attacks begin at round 15)
- This is after `warmup_rounds = 10`, so profiles are well-established
- Results saved in: `results/flower/` (with `_comp15` suffix in filenames)

### Parameters (Same as v2)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.05 | Slower EWMA updates |
| `tau_base` | 1.2 | Lower threshold |
| `weights` | (0.5, 0.3, 0.2) | Prioritize magnitude |
| `warmup_rounds` | 10 | Longer conservative period |
| `warmup_factor` | 0.3 | Stricter during warmup |
| `min_rounds_for_trust` | 5 | Longer trust building |
| `new_client_weight` | 0.3 | Less influence for new clients |

---

## Version 2 - Tuned Parameters (Immediate Attack) - 2025-12-03

**Rationale:** Improve detection of random noise attacks while maintaining performance against sign flipping and ALIE.

### Results with Immediate Attack (compromise_round=0)
**Saved in:** `v2_immediate_attack/`

| Attack | Byzantine % | Final Accuracy | Degradation |
|--------|-------------|----------------|-------------|
| None | 0% | **77.98%** | - |
| Sign Flipping | 10% | **75.98%** | 2.00pp |
| Sign Flipping | 20% | **72.05%** | 5.93pp |
| Sign Flipping | 30% | **68.89%** | 9.09pp |
| Random Noise | 10% | **35.22%** | 42.76pp ❌ |
| Random Noise | 20% | **10.00%** | 67.98pp ❌ |
| Random Noise | 30% | **10.01%** | 67.97pp ❌ |
| ALIE | 10% | **73.43%** | 4.55pp |
| ALIE | 20% | **68.33%** | 9.65pp |
| ALIE | 30% | **63.34%** | 14.64pp |

### Analysis
- **Sign Flipping:** Good defense (2-9pp degradation)
- **ALIE:** Moderate defense (4.5-14.6pp degradation)
- **Random Noise:** Complete failure (42-68pp degradation)

### Why Random Noise Still Failed
Byzantine clients attacking from round 0 establish their malicious behavior as the baseline profile. When their anomaly scores are computed, they're compared against their own history, which is already malicious.

| Attack | Mean Anomaly Score | Threshold (τ) | Detection |
|--------|-------------------|---------------|-----------|
| Baseline (honest) | 0.9357 | 1.2 | - |
| Random Noise 10% | 0.7573 | 1.2 | LOWER than honest! |
| Random Noise 30% | 0.9134 | 1.2 | Similar to honest |

The anomaly scores for random noise are actually LOWER than honest clients because the noise smooths out the heterogeneity from non-IID data.

### Parameter Changes from v1
| Parameter | v1 (Old) | v2 (New) | Rationale |
|-----------|----------|----------|-----------|
| `alpha` | 0.1 | **0.05** | Slower EWMA updates to resist profile poisoning |
| `tau_base` | 2.0 | **1.2** | Lower threshold to catch more anomalies |
| `weights` | (1/3, 1/3, 1/3) | **(0.5, 0.3, 0.2)** | Prioritize magnitude for random noise attacks |
| `warmup_rounds` | 5 | **10** | Longer conservative period |
| `warmup_factor` | 0.5 | **0.3** | Stricter during warmup |
| `min_rounds_for_trust` | 3 | **5** | Longer trust building period |
| `new_client_weight` | 0.5 | **0.3** | Less influence for new/untrusted clients |

---

## Version 1 (Original) - Initial Implementation

**Saved in:** `v1_original_params/`

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.1 | EWMA smoothing factor |
| `gamma` | 0.1 | Reliability update rate |
| `tau_base` | 2.0 | Base anomaly threshold |
| `beta` | 0.5 | Threshold flexibility factor |
| `window_size` | 5 | History window size |
| `weights` | (1/3, 1/3, 1/3) | Equal weighting for magnitude, directional, temporal |
| `warmup_rounds` | 5 | Warmup period |
| `warmup_factor` | 0.5 | Warmup threshold multiplier |
| `min_rounds_for_trust` | 3 | Rounds before reliability bonus |
| `use_cross_comparison` | True | Cross-client comparison enabled |
| `use_population_init` | True | Population initialization enabled |
| `new_client_weight` | 0.5 | Weight for new clients |

### Results (CIFAR-10, 25 clients, 50 rounds, Dirichlet α=0.5)

| Attack | Byzantine % | Final Accuracy | Degradation |
|--------|-------------|----------------|-------------|
| None | 0% | 77.85% | - |
| Sign Flipping | 10% | 75.99% | 1.86pp |
| Sign Flipping | 20% | 72.75% | 5.10pp |
| Sign Flipping | 30% | 69.08% | 8.77pp |
| Random Noise | 10% | 37.92% | **39.93pp** |
| Random Noise | 20% | 10.01% | **67.84pp** |
| Random Noise | 30% | 14.97% | **62.88pp** |
| ALIE | 10% | 72.59% | 5.26pp |

---

## Directory Structure

```
level5a_caacfl/
├── v1_original_params/         # Results with v1 parameters (immediate attack)
├── v2_immediate_attack/        # Results with v2 parameters (immediate attack)
├── results/flower/             # Current experiment results
├── PARAMETER_TUNING_HISTORY.md # This file
├── caacfl.py                   # CAAC-FL algorithm implementation
├── caacfl_strategy.py          # Flower strategy wrapper
└── run_flower_experiments.py   # Experiment runner with delayed compromise support
```

## Usage

```bash
# Run with delayed compromise (recommended for profile-based detection)
python run_flower_experiments.py --strategy caacfl --all_attacks \
    --compromise_round 15 --dataset cifar10 --output_dir ./results/flower

# Run with immediate attack (traditional threat model, for comparison)
python run_flower_experiments.py --strategy caacfl --all_attacks \
    --compromise_round 0 --dataset cifar10 --output_dir ./results/immediate
```
