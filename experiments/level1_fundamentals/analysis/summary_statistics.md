# Baseline Aggregation Comparison: Summary Statistics

## MNIST

### Iid Equal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 98.98 | 0.03 | [98.94, 99.02] | 5 |
| FEDMEAN | 98.94 | 0.01 | [98.92, 98.96] | 5 |
| FEDMEDIAN | 98.89 | 0.03 | [98.84, 98.93] | 5 |

### Iid Unequal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 99.12 | 0.02 | [99.09, 99.15] | 5 |
| FEDMEAN | 98.97 | 0.04 | [98.92, 99.02] | 5 |
| FEDMEDIAN | 98.59 | 0.09 | [98.46, 98.71] | 5 |

## FASHION-MNIST

### Iid Equal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 89.09 | 0.12 | [88.92, 89.26] | 5 |
| FEDMEAN | 89.07 | 0.05 | [89.01, 89.14] | 5 |
| FEDMEDIAN | 89.05 | 0.14 | [88.86, 89.24] | 5 |

### Iid Unequal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 90.10 | 0.25 | [89.76, 90.45] | 5 |
| FEDMEAN | 89.18 | 0.13 | [88.99, 89.36] | 5 |
| FEDMEDIAN | 88.09 | 0.23 | [87.76, 88.41] | 5 |

## CIFAR10

### Iid Equal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDMEAN | 62.73 | 0.44 | [62.12, 63.35] | 5 |
| FEDAVG | 62.66 | 0.25 | [62.31, 63.00] | 5 |
| FEDMEDIAN | 61.27 | 0.34 | [60.80, 61.74] | 5 |

### Iid Unequal

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 67.24 | 0.85 | [66.06, 68.43] | 5 |
| FEDMEAN | 62.94 | 0.25 | [62.59, 63.29] | 5 |
| FEDMEDIAN | 57.03 | 0.83 | [55.89, 58.18] | 5 |

---
*95% CI = 95% Confidence Interval*
