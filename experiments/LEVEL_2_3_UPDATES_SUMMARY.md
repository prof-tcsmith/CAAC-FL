# Level 2 & 3 Updates Summary

## Changes Applied Based on Level 1 Learnings

### Level 2 (Heterogeneous Data)
**Files Updated:** `run_fedavg.py`, `run_fedmedian.py`, `run_krum.py`

1. ✅ **LOCAL_EPOCHS: 1 → 5**
   - Now consistent with Levels 1 & 3
   - Provides better convergence with heterogeneous data

2. ✅ **GPU allocation: 0.15 → 0.04**
   - Already had from initial fix
   - Allows all 50 clients to run in parallel

3. ✅ **num_workers: 4**
   - Already had from initial optimizations
   - Better data loading performance

4. ✅ **Timing tracking**
   - Already had from initial optimizations
   - Shows elapsed/remaining time estimates

5. ✅ **Warning suppression**
   - Already had in run scripts and client.py
   - Clean experiment logs

---

### Level 3 (Byzantine Attacks)
**Files Updated:** `run_fedavg.py`, `run_fedmedian.py`, `run_krum.py`, `run_trimmed_mean.py`

1. ✅ **LOCAL_EPOCHS: 5**
   - Already correct
   - Consistent with other levels

2. ✅ **GPU allocation: 0.15 → 0.04**
   - **CRITICAL FIX** - Was missed in initial pass due to different quote style
   - Changed from `"num_gpus": 0.15` to `"num_gpus": 0.04`
   - Prevents deadlock with 50 clients

3. ✅ **num_workers: 4 (NEWLY ADDED)**
   - **NEW** - Added to both test_loader and train_loader
   - Level 3 creates DataLoaders inline, not via shared function
   - Improves data loading performance
   ```python
   # Test loader:
   test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

   # Train loader:
   train_loader = DataLoader(
       train_dataset,
       batch_size=BATCH_SIZE,
       sampler=torch.utils.data.SubsetRandomSampler(client_indices),
       num_workers=4
   )
   ```

4. ✅ **Timing tracking**
   - Already had from initial optimizations
   - Shows elapsed/remaining time estimates

5. ✅ **Warning suppression**
   - Already had in run scripts and client.py
   - Clean experiment logs

---

## Shared Optimizations (All Levels)

Via `shared/data_utils.py`:
- ✅ `pin_memory=False` - Eliminates PyTorch warnings
- ✅ `persistent_workers=True` - Keeps DataLoader workers alive
- ✅ `prefetch_factor=2` - Prefetches batches for efficiency

Via `client.py` files:
- ✅ Warning suppression filters in all 3 client implementations

---

## Final Configuration (All Levels)

```python
NUM_CLIENTS = 50          # All experiments
LOCAL_EPOCHS = 5          # All levels (was 1 for Levels 1 & 2)
BATCH_SIZE = 32           # Unchanged (maintains validity)
LEARNING_RATE = 0.01      # Unchanged
GPU_ALLOCATION = 0.04     # All levels (allows 50 clients in parallel)
num_workers = 4           # All levels (efficient data loading)
```

---

## Key Fixes

### Critical Fix: Level 3 GPU Allocation
**Why it was missed:** Level 3 uses different code style:
- Levels 1 & 2: `'num_gpus': 0.04` (single quotes)
- Level 3: `"num_gpus": 0.04` (double quotes)

Initial regex pattern only matched single quotes. Fixed with comprehensive pattern matching.

### New Addition: Level 3 num_workers
**Why it was different:** Level 3 creates DataLoaders inline in run scripts (for Byzantine attack flexibility), while Levels 1 & 2 use `shared/create_dataloaders()`.

Added `num_workers=4` directly to DataLoader calls in all 4 Level 3 scripts.

---

## Verification

All 15 files now have consistent optimizations:
- ✅ 10 run scripts (LOCAL_EPOCHS=5, GPU=0.04, num_workers=4, timing, warnings)
- ✅ 3 client.py files (warning suppression)
- ✅ 2 shared modules (DataLoader optimizations)

**Result:** All experiments will run efficiently with 50 clients, no warnings, and consistent parameters across all levels.
