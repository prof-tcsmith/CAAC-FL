# Quick Installation Guide

**TL;DR**: Get up and running in 3 steps.

## Prerequisites

- Python 3.8+ installed
- Git (to clone repository)
- ~2GB free disk space

---

## Installation (Choose One)

### Option A: CPU Installation (Fastest)

```bash
cd experiments
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-cpu.txt
```

### Option B: GPU Installation (NVIDIA CUDA)

```bash
cd experiments
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-gpu.txt
```

### Option C: Conda Installation

```bash
cd experiments
conda env create -f environment.yml
conda activate caac-fl
```

---

## Verify Installation

```bash
python -c "import torch; import flwr; print('✓ Installation successful!')"
```

---

## Download Dataset

```bash
python download_dataset.py
```

---

## Run First Experiment

```bash
cd level1_fundamentals
bash run_all.sh
```

---

## What's Installed?

- **PyTorch 2.0+**: Deep learning framework
- **Flower 1.5+**: Federated learning
- **NumPy, Pandas**: Data processing
- **Matplotlib, Seaborn**: Visualization
- **scikit-learn**: ML utilities

---

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade pip
pip install -r requirements-cpu.txt --force-reinstall
```

**GPU not detected?**
```bash
nvidia-smi  # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Need help?** See [SETUP.md](SETUP.md) for detailed instructions.

---

## Installation Files

- `requirements.txt` - Base dependencies
- `requirements-cpu.txt` - CPU-only (recommended)
- `requirements-gpu.txt` - GPU with CUDA
- `requirements-dev.txt` - Development tools
- `environment.yml` - Conda environment

---

**Ready to go!** See [README.md](README.md) for experiment details.
