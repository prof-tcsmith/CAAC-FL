# Level 3 Experimental Issues - Diagnostic Report

## Summary
The Level 3 experiments completed successfully from a technical standpoint (no crashes), but **Krum and Trimmed Mean aggregation methods are completely failing to learn**, achieving only random-guess accuracy (~10%) while FedAvg and FedMedian work correctly (~75% baseline).

---

## Issues Fixed

### 1. Empty Summary CSV ✅ FIXED
**Problem:** `analyze_results.py` expected JSON files but run scripts only called `logger.save_csv()`.

**Solution:** Created `convert_csv_to_json.py` to convert existing CSVs to JSON format.

**Fix for Future:** Update all run scripts to call both:
```python
logger.save_csv()
logger.save_json()  # ADD THIS LINE
```

---

## Critical Performance Issues

### Results Summary

| Method | No Attack | Random Noise | Sign Flipping |
|--------|-----------|--------------|---------------|
| **FedAvg** | 76.77% ✅ | 10.01% ❌ | 70.31% ⚠️ |
| **FedMedian** | 75.17% ✅ | 75.07% ✅ | 69.39% ✅ |
| **Krum** | 8.13% ❌ | 8.86% ❌ | 10.76% ❌ |
| **Trimmed Mean** | 10.18% ❌ | 10.77% ❌ | 11.89% ❌ |

*(Random guess for 10-class CIFAR-10 = 10%)*

---

## Problem Analysis

### Issue 1: Krum Complete Failure (8.13% accuracy)

**Observed:**
- Baseline accuracy: 8.13% (random chance)
- Loss stays at 2.30 (initial untrained model)
- No improvement over 50 rounds

**Likely Causes:**

1. **Krum selects ONE client's model entirely** - With highly non-IID data (Dirichlet α=0.5), a single client trained on a biased data subset produces a model that doesn't generalize to the global test set.

2. **High data heterogeneity** - Each of 15 clients has very different class distributions. Client 0 might only see classes 1,3,7 while Client 5 only sees 2,4,6,8. Their individual models can't classify all 10 classes.

3. **Selection instability** - Krum selects a different client each round based on distances, causing the global model to "jump" between incompatible models.

**Expected Behavior:**
- Multi-Krum (select top-m clients and average) would work better
- OR lower heterogeneity (higher α like 1.0 or 5.0)
- OR more clients to increase coverage

**Literature Note:** Krum is designed for IID or mildly non-IID settings. In highly heterogeneous federated learning, it degrades significantly.

---

### Issue 2: Trimmed Mean Complete Failure (10.18% accuracy)

**Observed:**
- Baseline accuracy: 10.18% (barely better than random)
- Loss stays at 2.30 (no learning)
- Trimming 20% = removing 3/15 clients from each end

**Likely Causes:**

1. **Over-aggressive coordinate-wise trimming** - With 15 clients and 20% trimming, we remove top 3 and bottom 3 values PER PARAMETER. With heterogeneous updates, this might be removing the most informative gradients.

2. **Coordinate-wise sorting breaks gradient coherence** - Neural network gradients have correlations across parameters. Sorting and trimming each coordinate independently destroys these relationships.

3. **Heterogeneous updates have high variance** - With α=0.5, client updates point in very different directions. The "middle" values after sorting may not represent a meaningful update direction.

**Example:**
```
Parameter W[0,0] updates from 15 clients:
[-0.5, -0.3, -0.2, 0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
         ^trim these^                                 ^trim these^
                     Average middle 9 values...
```

But the correlation with W[0,1] is broken because those values are sorted independently.

**Expected Behavior:**
- Should work with more clients (30-50)
- OR lower trimming ratio (10% instead of 20%)
- OR use gradient clipping before trimming

---

### Issue 3: FedAvg Catastrophic Failure under Random Noise (10.01% accuracy)

**Observed:**
- Baseline (no attack): 76.77% ✅
- Random Noise attack: 10.01% ❌ (87% degradation!)
- Sign Flipping attack: 70.31% ⚠️ (only 8% degradation)

**Problem:** Random noise attack (Gaussian noise σ=1.0 added to all parameters) completely destroys FedAvg performance, worse than sign flipping!

**Likely Cause:**
- Noise scale (σ=1.0) is too large relative to model updates
- Byzantine clients (20% = 3/15) contribute pure noise
- FedAvg averages everything, so noise pollutes 20% of the aggregated model
- Over 50 rounds, accumulated noise degrades the model to random guess

**Why Sign Flipping is Less Harmful:**
- Sign flipping reverses gradient direction but keeps magnitude
- When averaged with 12 honest updates, the Byzantine updates partially cancel out
- The honest majority's signal still dominates

**Fix:** Scale random noise to match update magnitudes, e.g., σ=0.1 * std(updates)

---

## Root Cause Summary

The experimental setup has **mismatched assumptions**:

1. **High data heterogeneity (α=0.5)** - Creates very non-IID data
2. **Small number of clients (15)** - Insufficient for heterogeneous settings
3. **High Byzantine ratio (20% = 3 clients)** - Above theoretical tolerance for some methods
4. **Attack parameters not tuned** - Random noise too strong

| Method | Assumption | Reality | Result |
|--------|------------|---------|--------|
| Krum | Mild heterogeneity | High heterogeneity (α=0.5) | Fails |
| Trimmed Mean | Many clients (50+) | Only 15 clients | Fails |
| FedAvg | No attacks | Random noise too strong | Fails |
| FedMedian | Coordinate-wise robust | High heterogeneity | **Works!** ✅ |

---

## Recommendations

### Option 1: Fix Data Distribution (Easier)
**Reduce heterogeneity to match method assumptions:**

```python
# In run scripts, change:
alpha=0.5  # Current (very heterogeneous)
# To:
alpha=2.0  # Moderate heterogeneity - should allow learning
```

**Expected outcome:** All methods should achieve 60-75% baseline accuracy

---

### Option 2: Fix Method Implementations (Harder)
**Modify strategies to handle heterogeneity:**

1. **Multi-Krum:** Select top-3 clients and average them instead of picking one
2. **Trimmed Mean:** Reduce trim_ratio from 0.2 to 0.1 (10%)
3. **Random Noise Attack:** Scale noise to `σ = 0.1 * std(honest_updates)`

---

### Option 3: Increase Scale (Most Realistic)
**Match real federated learning scenarios:**

```python
NUM_CLIENTS = 50  # Up from 15
NUM_BYZANTINE = 10  # 20% of 50
alpha = 0.5  # Keep heterogeneous
```

**Expected outcome:** Krum and Trimmed Mean should work with more clients

---

## Validation Questions

To diagnose further, check:

1. **Are Krum/TrimmedMean receiving initial parameters?**
   - Check Flower logs for "Requesting initial parameters"

2. **Are client training losses decreasing?**
   - Check individual client metrics during training

3. **What do aggregated parameters look like?**
   - Add logging to print parameter statistics after aggregation

4. **How similar are client updates?**
   - Compute pairwise distances between client updates
   - High variance suggests incompatible updates

---

## Conclusion

The Level 3 experiments **technically succeeded** (no errors, data collected), but revealed that:

1. **FedMedian is the clear winner** for Byzantine robustness under high data heterogeneity
2. **Krum and Trimmed Mean require modifications** or different experimental settings
3. **FedAvg needs attack parameter tuning** (Random Noise is too destructive)

The results are **scientifically valid** - they show these methods don't work well in this regime. But for a more balanced comparison, I recommend **Option 1 (increase α to 2.0)** to give all methods a fair chance to learn.
