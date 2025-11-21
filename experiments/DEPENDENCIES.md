# CAAC-FL Experiments - Dependencies Reference

## Overview

This document provides a complete reference of all dependencies used in the CAAC-FL experimental framework.

## Requirements Files Summary

| File | Purpose | Size | Use Case |
|------|---------|------|----------|
| `requirements.txt` | Base dependencies only | ~10 packages | Reference or custom PyTorch installation |
| `requirements-cpu.txt` | Complete CPU setup | ~10 packages | **Recommended for most users** |
| `requirements-gpu.txt` | Complete GPU setup | ~10 packages | NVIDIA CUDA GPU systems |
| `requirements-dev.txt` | Development tools | ~20 packages | Contributors and developers |
| `environment.yml` | Conda environment | Full spec | Conda users |

## Core Dependencies

### Deep Learning Frameworks

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `torch` | >=2.0.0 | PyTorch deep learning framework | BSD-3-Clause |
| `torchvision` | >=0.15.0 | Computer vision datasets and models | BSD-3-Clause |

**Installation Notes:**
- CPU version: ~200MB download
- GPU version: ~2GB download (includes CUDA libraries)
- CPU-only is sufficient for Level 1-3 experiments
- GPU recommended for Level 4-5 (faster training)

### Federated Learning

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `flwr` (Flower) | >=1.5.0, <2.0.0 | Federated learning framework | Apache 2.0 |

**Why Flower?**
- Built-in aggregation strategies (FedAvg, FedMedian, Krum, etc.)
- Easy simulation mode for development
- Production-ready for distributed deployment
- Active development and community

### Numerical Computing

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `numpy` | >=1.24.0, <2.0.0 | Array operations, linear algebra | BSD-3-Clause |
| `scipy` | >=1.10.0 | Scientific computing, optimization | BSD-3-Clause |

**Note**: NumPy 2.0+ introduces breaking changes, so we cap at <2.0.0 for stability.

### Data Processing

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `pandas` | >=2.0.0 | Data manipulation and analysis | BSD-3-Clause |

**Used for:**
- Metrics logging and aggregation
- Results comparison across experiments
- CSV export for analysis

### Visualization

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `matplotlib` | >=3.7.0 | Plotting and visualization | PSF-based |
| `seaborn` | >=0.12.0 | Statistical visualization | BSD-3-Clause |

**Used for:**
- Training curves (accuracy, loss)
- Comparison plots across methods
- Detection metrics visualization
- Distribution analysis

### Machine Learning Utilities

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `scikit-learn` | >=1.3.0 | ML metrics and utilities | BSD-3-Clause |

**Used for:**
- Confusion matrix computation
- Classification metrics (precision, recall, F1)
- Byzantine detection metrics (TPR, FPR)

### General Utilities

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `tqdm` | >=4.65.0 | Progress bars | MIT/MPL |
| `pillow` | >=9.0.0 | Image processing | HPND |

## Development Dependencies

Only included in `requirements-dev.txt`.

### Interactive Development

| Package | Version | Purpose |
|---------|---------|---------|
| `jupyter` | >=1.0.0 | Jupyter metapackage |
| `ipykernel` | >=6.0.0 | IPython kernel |
| `jupyterlab` | >=4.0.0 | JupyterLab interface |
| `notebook` | >=7.0.0 | Classic notebook interface |

### Testing

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.0.0 | Testing framework |
| `pytest-cov` | >=4.0.0 | Coverage reporting |

### Code Quality

| Package | Version | Purpose |
|---------|---------|---------|
| `black` | >=23.0.0 | Code formatter |
| `flake8` | >=6.0.0 | Linter |
| `mypy` | >=1.0.0 | Type checker |
| `isort` | >=5.12.0 | Import sorter |

### Documentation

| Package | Version | Purpose |
|---------|---------|---------|
| `sphinx` | >=7.0.0 | Documentation generator |
| `sphinx-rtd-theme` | >=1.3.0 | Read the Docs theme |

### Profiling

| Package | Version | Purpose |
|---------|---------|---------|
| `memory-profiler` | >=0.61.0 | Memory profiling |
| `line-profiler` | >=4.0.0 | Line-by-line profiling |

### Advanced Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| `plotly` | >=5.14.0 | Interactive plots |

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Disk**: 2GB free space
- **CPU**: Multi-core recommended

### Recommended Requirements

- **Python**: 3.11
- **RAM**: 16GB
- **Disk**: 5GB free space
- **CPU**: 4+ cores
- **GPU** (optional): NVIDIA CUDA-capable GPU with 4GB+ VRAM

### For GPU Support

- **CUDA**: 11.8 or 12.1
- **cuDNN**: Compatible version (installed with PyTorch)
- **NVIDIA Driver**: Recent version (check with `nvidia-smi`)

## Installation Size Estimates

| Configuration | Download Size | Installed Size |
|---------------|---------------|----------------|
| CPU-only | ~300 MB | ~1.5 GB |
| GPU (CUDA 11.8) | ~2.5 GB | ~8 GB |
| GPU (CUDA 12.1) | ~2.8 GB | ~9 GB |
| + Development | +50 MB | +300 MB |

## Dependency Tree

```
CAAC-FL Experiments
├── PyTorch (torch + torchvision)
│   └── Used for: Model definition, training, evaluation
├── Flower (flwr)
│   ├── Depends on: grpcio, protobuf, numpy
│   └── Used for: Federated learning simulation
├── Data Processing
│   ├── NumPy (numpy)
│   ├── SciPy (scipy)
│   └── Pandas (pandas)
├── Visualization
│   ├── Matplotlib (matplotlib)
│   └── Seaborn (seaborn)
├── ML Utilities
│   └── scikit-learn
└── Utilities
    ├── tqdm (progress bars)
    └── Pillow (image processing)
```

## Version Compatibility

### Tested Configurations

| Python | PyTorch | Flower | Status |
|--------|---------|--------|--------|
| 3.11 | 2.9.1 | 1.5+ | ✅ Tested |
| 3.10 | 2.0+ | 1.5+ | ✅ Expected to work |
| 3.9 | 2.0+ | 1.5+ | ✅ Expected to work |
| 3.8 | 2.0+ | 1.5+ | ⚠️ Minimum supported |

### Known Issues

1. **NumPy 2.0+**: Breaking changes, use <2.0.0
2. **Flower 2.0+**: API changes, use <2.0.0
3. **Python 3.7**: Not supported (EOL)
4. **Windows**: May need Visual C++ Build Tools for some packages

## Package Sources

All packages are available from:
- **PyPI** (Python Package Index): pip install
- **Conda-forge**: conda install (most packages)
- **PyTorch channel**: conda install pytorch (recommended for conda users)

## License Compliance

All dependencies use permissive licenses compatible with academic and commercial use:
- BSD-3-Clause: PyTorch, NumPy, scikit-learn, pandas, seaborn
- Apache 2.0: Flower
- MIT: tqdm
- PSF-based: matplotlib

## Updating Dependencies

### Check for Updates

```bash
pip list --outdated
```

### Update All

```bash
pip install --upgrade -r requirements-cpu.txt
```

### Update Individual Package

```bash
pip install --upgrade torch torchvision
```

**Note**: Test experiments after updates to ensure compatibility.

## Security

### Vulnerability Scanning

```bash
pip install pip-audit
pip-audit
```

### Keeping Dependencies Secure

- Regularly update to latest versions
- Monitor security advisories
- Use virtual environments
- Pin versions in production

## Additional Resources

- **PyTorch**: https://pytorch.org/
- **Flower**: https://flower.dev/
- **NumPy**: https://numpy.org/
- **scikit-learn**: https://scikit-learn.org/
- **Pandas**: https://pandas.pydata.org/
- **Matplotlib**: https://matplotlib.org/

---

**Last Updated**: 2025-11-20
**Maintained by**: CAAC-FL Research Team
