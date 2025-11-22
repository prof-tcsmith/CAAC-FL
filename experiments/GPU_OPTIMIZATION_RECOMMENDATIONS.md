# GPU Optimization Recommendations for CAAC-FL Experiments

## Current Performance Issues
- GPU utilization: ~2% per process
- Sequential/bursty GPU usage instead of parallel
- High data loading overhead relative to compute

## Optimization Strategies

### 1. Increase Batch Size (Highest Impact)
**Current:** `BATCH_SIZE = 32`
**Recommended:** `BATCH_SIZE = 128` or `256`

GPUs achieve peak efficiency with larger batches. This change alone can increase GPU utilization by 3-5x.

```python
BATCH_SIZE = 128  # or 256 if GPU memory allows
```

**Impact:** Reduces number of batches per epoch, increases GPU occupancy

---

### 2. Increase Local Epochs
**Current:** `LOCAL_EPOCHS = 1`
**Recommended:** `LOCAL_EPOCHS = 3` to `5`

More training per round amortizes the overhead of model serialization and data loading.

```python
LOCAL_EPOCHS = 5  # Level 1
LOCAL_EPOCHS = 5  # Level 2 & 3 (already set correctly)
```

**Impact:** Reduces overhead-to-computation ratio from ~90% to ~50%

---

### 3. Optimize DataLoader Configuration

**Current Issues:**
- `pin_memory=False` (we disabled it to fix warnings)
- `num_workers=2` (too low)
- No persistent workers (recreated each epoch)

**Recommended Changes:**

```python
# In shared/data_utils.py - create_dataloaders()
DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,              # Increase from 2
    pin_memory=True,            # Re-enable for GPU transfer efficiency
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2,          # Prefetch 2 batches per worker
)
```

**Impact:** Reduces data loading bottleneck by 2-3x

---

### 4. Increase GPU Allocation Per Client

**Current:** `'num_gpus': 0.1` per client
**Recommended:** `'num_gpus': 0.2` or `0.25`

Larger GPU slices give Ray's scheduler better parallelization opportunities.

```python
# In all run_*.py files
client_resources={
    'num_cpus': 1,
    'num_gpus': 0.2 if DEVICE.type == 'cuda' else 0
}
```

With 30 clients × 0.2 GPU = 6.0 GPU total needed:
- First batch: 10 clients in parallel (2.0 GPU)
- Second batch: 10 clients in parallel (2.0 GPU)
- Third batch: 10 clients in parallel (2.0 GPU)

**Impact:** Better GPU utilization through larger work chunks

---

### 5. Enable Mixed Precision Training

Add automatic mixed precision (AMP) for 2x speedup on modern GPUs:

```python
# In level*_*/client.py - FlowerClient.fit()
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(self.local_epochs):
    for batch in self.trainloader:
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

        self.optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
```

**Impact:** 1.5-2x speedup on RTX 4090 with no accuracy loss

---

### 6. Disable Pin Memory Warnings (Not Performance)

The pin_memory deprecation warnings don't affect performance, but to fix them properly:

```python
# In shared/data_utils.py
DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if num_workers > 0 else False,
)
```

The warnings come from PyTorch internals and are safe to ignore.

---

### 7. Reduce Ray Actor Overhead

**Option A: Increase clients per actor**
```python
# In ray_init_args
ray_init_args = {
    "include_dashboard": False,
    "num_cpus": 128,
    "num_gpus": 2,
    "_memory": 50 * 1024 * 1024 * 1024,
    "object_store_memory": 100 * 1024 * 1024 * 1024,
    "_system_config": {
        "max_direct_call_object_size": 100 * 1024 * 1024,  # 100MB
    }
}
```

**Option B: Reduce number of clients** (if scientifically valid)
- Fewer clients = less overhead
- But may affect heterogeneity simulation

---

## Expected Performance Improvements

### Conservative Estimate (Batch=128, Epochs=5, Workers=4, GPU=0.2)
- **Current:** ~35 seconds/round
- **Optimized:** ~15-20 seconds/round
- **Speedup:** 1.75-2.3x faster
- **GPU Utilization:** 15-30% per process

### Aggressive Estimate (Batch=256, Epochs=5, Workers=4, GPU=0.25, AMP)
- **Current:** ~35 seconds/round
- **Optimized:** ~10-12 seconds/round
- **Speedup:** 2.9-3.5x faster
- **GPU Utilization:** 30-50% per process

---

## Implementation Priority

**High Impact (Do First):**
1. Increase batch size to 128-256
2. Set local epochs to 5 for Level 1
3. Increase num_workers to 4

**Medium Impact:**
4. Increase GPU allocation to 0.2
5. Enable persistent_workers
6. Add mixed precision training

**Low Impact (Optional):**
7. Tune Ray system config
8. Profile with PyTorch profiler to identify remaining bottlenecks

---

## Testing Plan

1. **Baseline Test:** Run current config with timing
2. **Test 1:** Change only batch_size to 128, compare
3. **Test 2:** Add local_epochs=5, compare
4. **Test 3:** Add num_workers=4 + persistent_workers=True, compare
5. **Test 4:** Increase GPU allocation to 0.2, compare
6. **Test 5:** Enable mixed precision, compare

---

## Notes on Federated Learning Simulation Limitations

Even with optimizations, FL simulations have inherent overhead:
- **Ray serialization:** Each client's model must be serialized/deserialized
- **Python GIL:** Limits CPU parallelism
- **Small model:** SimpleCNN trains in milliseconds, overhead is significant

For maximum GPU utilization (>80%), you would need:
- Much larger models (ResNet50, BERT, etc.)
- Longer training per round (10+ epochs)
- Batch sizes of 256-512

But for research purposes, 2-3x speedup from basic optimizations is worthwhile.
