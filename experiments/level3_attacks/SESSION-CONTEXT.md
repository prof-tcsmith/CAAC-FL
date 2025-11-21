# Level 3 Experiments - Session Context

**Last Updated:** 2025-11-21
**Status:** Experiments interrupted by reboot, need to restart on faster machine

## Current State

### Completed Work

1. **Code Updates (all committed to main branch)**
   - Updated all 4 run scripts to use **50 clients** (default parameter changed from 15 to 50)
   - Modified all scripts to save both CSV and JSON formats
   - Files: `run_fedavg.py`, `run_fedmedian.py`, `run_krum.py`, `run_trimmed_mean.py`

2. **Backup Created**
   - Original 15-client results backed up to `results_15clients/` directory
   - Includes all 12 experiment results plus analysis

3. **Analysis Documents Created**
   - `ISSUES_FOUND.md` - Technical diagnostic of Krum/Trimmed Mean failures
   - `LEVEL3_ANALYSIS.md` - Complete analysis with 3 options
   - User selected **Option 2: Scale to 50 clients**

### Experiments Progress (50-client configuration)

**Completed (3 of 12):**
- ✅ FedAvg + no attack: 74.40% accuracy (completed)
- ✅ FedAvg + random_noise: 22.63% accuracy (completed)
- ✅ FedAvg + sign_flipping: results in `results/level3_fedavg_sign_flipping_metrics.json`

**Interrupted:**
- ❌ FedMedian + no attack: Was on Round 43/50 when reboot occurred

**Not Started (9 remaining):**
- FedMedian + random_noise
- FedMedian + sign_flipping
- Krum + no attack
- Krum + random_noise
- Krum + sign_flipping
- Trimmed Mean + no attack
- Trimmed Mean + random_noise
- Trimmed Mean + sign_flipping

## Key Findings So Far

### 15-Client Results (Baseline - in results_15clients/)
| Method | No Attack | Random Noise | Sign Flipping |
|--------|-----------|--------------|---------------|
| FedAvg | 76.77% | **10.01%** ❌ | 70.31% |
| FedMedian | 75.17% | 75.07% ✅ | 69.39% |
| Krum | **8.13%** ❌ | 8.86% ❌ | 10.76% ❌ |
| Trimmed Mean | **10.18%** ❌ | 10.77% ❌ | 11.89% ❌ |

**Problem:** Krum and Trimmed Mean completely failed with 15 clients (random guess accuracy ~8-10%)

**Root Cause:**
- Krum selects ONE client model → fails with high heterogeneity (α=0.5)
- Trimmed Mean needs many clients → only 15 clients, trimming 20% = only 9 values used
- FedAvg extremely vulnerable to Random Noise attack (87% degradation)

### 50-Client Results (Partial - in results/)
| Method | No Attack | Random Noise | Sign Flipping |
|--------|-----------|--------------|---------------|
| FedAvg | 74.40% (-2.37%) | **22.63% (+12.62%)** 🎯 | Check results/ |
| FedMedian | Running... | Not started | Not started |
| Krum | Not started | Not started | Not started |
| Trimmed Mean | Not started | Not started | Not started |

**Key Improvement:** FedAvg under Random Noise improved from 10.01% to 22.63% with 50 clients!
This suggests more honest clients (40 vs 12) dilute Byzantine noise better.

## Configuration Details

### Current Parameters (50-client setup)
- **Total clients:** 50 (changed from 15)
- **Byzantine clients:** 10 at 20% ratio (changed from 3)
- **Honest clients:** 40 at 80% ratio (changed from 12)
- **Rounds:** 50
- **Data heterogeneity:** α=0.5 (high non-IID)
- **Attack parameters:**
  - Random Noise: σ=1.0 Gaussian noise
  - Sign Flipping: Reverse gradient direction
- **Model:** CNN on CIFAR-10
- **Krum:** f=10 (Byzantine tolerance parameter)
- **Trimmed Mean:** β=0.2 (trim ratio, removes 10 from each end)

### Timing Estimates (50 clients)
- Each experiment: ~56 minutes (3,350-3,380 seconds)
- Remaining 9 experiments: ~8.4 hours
- Total for all 12: ~11.2 hours

### Files Modified (all committed)
```
experiments/level3_attacks/run_fedavg.py:36          default=50
experiments/level3_attacks/run_fedavg.py:199         logger.save_json()
experiments/level3_attacks/run_fedmedian.py:36       default=50
experiments/level3_attacks/run_fedmedian.py:199      logger.save_json()
experiments/level3_attacks/run_krum.py:40            default=50
experiments/level3_attacks/run_krum.py:205           logger.save_json()
experiments/level3_attacks/run_trimmed_mean.py:37    default=50
experiments/level3_attacks/run_trimmed_mean.py:203   logger.save_json()
```

## Next Steps on New Machine

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prof-tcsmith/CAAC-FL.git
   cd CAAC-FL/experiments/level3_attacks
   ```

2. **Setup environment (if needed):**
   ```bash
   conda env create -f ../environment.yml
   conda activate caac-fl
   ```

3. **Verify CIFAR-10 dataset exists:**
   ```bash
   ls -lh data/cifar-10-batches-py/
   # Should show ~170MB of data files
   # If missing: python ../download_dataset.py
   ```

4. **Clear partial results:**
   ```bash
   # IMPORTANT: Back up the 3 completed FedAvg results first if needed
   # Or just let run_all.sh overwrite them
   rm -f results/*.csv results/*.json results/*.png
   ```

5. **Run all experiments:**
   ```bash
   bash run_all.sh 2>&1 | tee experiment_run_50clients_v2.log
   ```
   - Estimated time: ~11-12 hours on faster machine (may be faster)
   - Creates 12 experiment files in `results/`
   - Automatically runs `analyze_results.py` at the end
   - Generates `level3_summary.csv` and `level3_attack_impact.png`

6. **Monitor progress (in another terminal):**
   ```bash
   tail -f experiment_run_50clients_v2.log | grep "ROUND\|Running\|Completed"
   ```

## Expected Outcomes

With 50 clients, we expect:

1. **Krum:** Should improve from 8.13% to 60-70% baseline
   - More clients → better selection diversity
   - 40 honest clients give better model options

2. **Trimmed Mean:** Should improve from 10.18% to 65-72% baseline
   - Trimming 10/50 = 40 values used (vs 9/15 previously)
   - Better statistical robustness with more samples

3. **FedAvg Random Noise:** Already showing improvement (22.63% vs 10.01%)
   - More honest clients dilute Byzantine noise

4. **FedMedian:** Should maintain ~75% (already robust)

## Files to Review After Completion

1. `results/level3_summary.csv` - Aggregated final accuracies
2. `results/level3_attack_impact.png` - Visualization
3. Compare with `results_15clients/` to quantify improvement

## Research Questions to Answer

1. Did Krum recover from failure with 50 clients?
2. Did Trimmed Mean recover from failure with 50 clients?
3. How much did FedAvg's Random Noise vulnerability improve?
4. Is 50 clients sufficient, or do we need even more (Option 2b)?
5. Should we explore Option 1 (increase α to 2.0) or Option 3 (tune attacks)?

## Important Notes

- All code changes are committed and pushed to main branch
- Dataset files excluded from git (in .gitignore)
- Results ARE included in git (CSV, JSON, PNG files)
- Log files excluded from git (*.log in .gitignore)
- Original 15-client results preserved in `results_15clients/`

## Context for AI Assistant

When picking up this work:
1. Read `LEVEL3_ANALYSIS.md` for background on why we're using 50 clients
2. Read `ISSUES_FOUND.md` for technical details on method failures
3. The user chose Option 2 (scale to 50 clients) from three options
4. We need to complete all 12 experiments and compare with 15-client baseline
5. Focus is on Byzantine-robust aggregation methods under attacks

## Future Work (Level 4 & 5)

After completing Level 3, see:
- `../LEVEL4-LEVEL5-IMPLEMENTATION-GUIDE.md` - Complete implementation guide
- `../EXPERIMENT-PLAN.md` - Full 5-level experimental progression plan

Level 4 will add:
- ALIE attacks (variance-based stealthy)
- Simple behavioral tracking (prototype)
- Detection metrics (TPR, FPR)

Level 5 will implement:
- Full CAAC-FL with multi-dimensional anomaly detection
- Client-specific adaptive thresholds
- Trust score system with reliability tracking
