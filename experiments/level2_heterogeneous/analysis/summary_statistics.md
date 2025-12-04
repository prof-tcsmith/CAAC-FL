# Level 2 Non-IID Aggregation Comparison: Summary Statistics

## CIFAR10

### Alpha = 0.5

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 62.06 | 0.51 | [61.61, 62.51] | 5 |
| FEDMEAN | 59.38 | 0.53 | [58.92, 59.84] | 5 |
| FEDMEDIAN | 52.91 | 1.07 | [51.97, 53.85] | 5 |


## MNIST

### Alpha = 0.5

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 98.90 | 0.08 | [98.83, 98.97] | 5 |
| FEDMEAN | 98.82 | 0.05 | [98.78, 98.86] | 5 |
| FEDMEDIAN | 98.56 | 0.09 | [98.48, 98.64] | 5 |


## FASHION-MNIST

### Alpha = 0.5

| Strategy | Mean Acc (%) | Std | 95% CI | N |
|----------|-------------|-----|--------|---|
| FEDAVG | 88.64 | 0.26 | [88.41, 88.87] | 5 |
| FEDMEAN | 88.17 | 0.20 | [87.99, 88.35] | 5 |
| FEDMEDIAN | 87.25 | 0.28 | [87.00, 87.50] | 5 |


---
*95% CI = 95% Confidence Interval*

*Generated from 45 experiments*
