# CAAC-FL Experiments Setup Guide

## Prerequisites

- **Python**: 3.8 or higher (3.11 recommended)
- **OS**: Linux, macOS, or Windows
- **GPU** (optional): NVIDIA CUDA-capable GPU for faster training
- **RAM**: Minimum 8GB, 16GB recommended
- **Disk Space**: ~2GB for dependencies + dataset

## Quick Start

Choose one of the installation methods below based on your preference and system.

---

## Method 1: Pip Installation (Recommended)

### For CPU-only Systems

```bash
# Navigate to experiments directory
cd experiments

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all dependencies (CPU version)
pip install -r requirements-cpu.txt
```

### For GPU Systems (NVIDIA CUDA)

```bash
# Navigate to experiments directory
cd experiments

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (GPU version with CUDA 11.8)
pip install -r requirements-gpu.txt

# For CUDA 12.1, edit requirements-gpu.txt first to use cu121 index
```

### Manual Installation (if requirements files don't work)

```bash
# Activate your virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install PyTorch (choose one):
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install flwr numpy pandas matplotlib seaborn scikit-learn scipy tqdm pillow
```

---

## Method 2: Conda Installation

### Create Conda Environment

```bash
# Navigate to experiments directory
cd experiments

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate caac-fl
```

### For GPU Support with Conda

Edit `environment.yml` and replace:
```yaml
- cpuonly
```

With one of:
```yaml
- pytorch-cuda=11.8  # for CUDA 11.8
# or
- pytorch-cuda=12.1  # for CUDA 12.1
```

Then create the environment:
```bash
conda env create -f environment.yml
conda activate caac-fl
```

### Manual Conda Installation

```bash
# Create new environment
conda create -n caac-fl python=3.11

# Activate environment
conda activate caac-fl

# Install PyTorch (CPU)
conda install pytorch torchvision cpuonly -c pytorch

# For GPU with CUDA 11.8:
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy pandas matplotlib seaborn scikit-learn scipy tqdm pillow -c conda-forge

# Install Flower (not available in conda, use pip)
pip install flwr
```

---

## Method 3: Development Installation

For development with additional tools (testing, linting, Jupyter):

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base + development dependencies
pip install -r requirements-cpu.txt  # or requirements-gpu.txt
pip install -r requirements-dev.txt
```

---

## Verification

After installation, verify everything is working:

### Quick Verification

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA availability (if GPU installed)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Flower installation
python -c "import flwr; print(f'Flower {flwr.__version__}')"
```

### Comprehensive Testing

```bash
cd level1_fundamentals
python test_setup.py
```

Expected output:
```
============================================================
Level 1 Setup Verification
============================================================
Testing imports...
  ✓ Flower imported successfully
  ✓ torchvision imported successfully
  ✓ SimpleCNN imported successfully
  ✓ Data utilities imported successfully
  ✓ Metrics utilities imported successfully
...
✓ All tests passed! Ready to run experiments.
```

---

## Download Dataset

After installation, download the CIFAR-10 dataset:

```bash
# From experiments directory
python download_dataset.py

# Verify dataset
python verify_dataset.py
```

This will download ~170MB and extract the CIFAR-10 dataset to `./data/`

## Running Experiments

### Level 1: Fundamentals

```bash
cd level1_fundamentals
bash run_all.sh
```

This will:
1. Run FedAvg experiment
2. Run FedMedian experiment
3. Generate comparison plots and analysis

Results will be saved in `level1_fundamentals/results/`

## Directory Structure

```
experiments/
├── shared/              # Shared utilities across all levels
│   ├── models.py        # Model architectures
│   ├── data_utils.py    # Data loading and partitioning
│   └── metrics.py       # Evaluation and logging
├── level1_fundamentals/ # Level 1: Basic FL
├── level2_heterogeneous/# Level 2: Non-IID data
├── level3_basic_attacks/# Level 3: Basic attacks
├── level4_advanced_attacks/ # Level 4: Advanced attacks
├── level5_caacfl/      # Level 5: Full CAAC-FL
└── analysis/           # Cross-level analysis
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
1. Reduce batch size in the run scripts
2. Use CPU instead: Set `DEVICE = 'cpu'` in run scripts
3. Reduce number of clients participating per round

### Slow Training

If training is too slow:
1. Enable GPU if available
2. Reduce number of rounds
3. Reduce dataset size in data_utils.py

### Import Errors

Make sure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're in the correct directory when running scripts

## Expected Runtime

- Level 1 (50 rounds, 10 clients): ~10-15 minutes on CPU, ~3-5 minutes on GPU
- Level 2: Similar to Level 1
- Level 3-5: Longer due to additional experiments and attack implementations

## Data Storage

CIFAR-10 dataset will be automatically downloaded to `./data/` on first run (~170MB).
This is shared across all experiments.

---

## Requirements Files Reference

The experiments directory includes multiple requirements files for different use cases:

| File | Purpose | When to Use |
|------|---------|-------------|
| `requirements.txt` | Base dependencies (no PyTorch) | When installing PyTorch separately or for reference |
| `requirements-cpu.txt` | Complete CPU installation | **Recommended for systems without GPU** |
| `requirements-gpu.txt` | Complete GPU installation | For systems with NVIDIA CUDA GPU |
| `requirements-dev.txt` | Development tools | For contributors or interactive development |
| `environment.yml` | Conda environment | For conda users |

### Dependencies Included

**Core Dependencies:**
- PyTorch 2.0+ and torchvision
- NumPy, SciPy, Pandas (data processing)
- Matplotlib, Seaborn (visualization)
- scikit-learn (machine learning utilities)
- Flower 1.5+ (federated learning framework)
- tqdm, Pillow (utilities)

**Development Dependencies** (requirements-dev.txt only):
- Jupyter, JupyterLab (interactive notebooks)
- pytest (testing)
- black, flake8, mypy, isort (code quality)
- sphinx (documentation)
- plotly (advanced visualization)

---

## GPU Support

### Checking GPU Availability

```bash
# Check if CUDA GPU is detected
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### Troubleshooting GPU Issues

1. **CUDA not available despite having GPU:**
   - Ensure NVIDIA drivers are installed: `nvidia-smi`
   - Reinstall PyTorch with correct CUDA version
   - Check CUDA toolkit version: `nvcc --version`

2. **CUDA version mismatch:**
   - Match PyTorch CUDA version with your system CUDA version
   - Use `nvidia-smi` to check CUDA version
   - Reinstall with correct version from [PyTorch website](https://pytorch.org)

3. **Out of memory errors:**
   - Reduce batch size
   - Use fewer clients per round
   - Clear CUDA cache: `torch.cuda.empty_cache()`

---

## Alternative Installation Methods

### Using pip with --user (no virtual environment)

```bash
pip install --user -r requirements-cpu.txt
```

Note: This is not recommended as it may conflict with system packages.

### Using Poetry

```bash
# Install Poetry first: https://python-poetry.org/docs/#installation

# Initialize project (creates pyproject.toml)
poetry init

# Add dependencies
poetry add torch torchvision flwr numpy pandas matplotlib scikit-learn

# Install
poetry install
```

### Using Docker (Advanced)

A Docker setup can be provided for containerized deployment. Contact maintainers if needed.

---

## Support and Troubleshooting

If you encounter issues:

1. **Check Python version**: `python --version` (must be 3.8+)
2. **Verify virtual environment is activated**: Look for `(venv)` in terminal prompt
3. **Update pip**: `pip install --upgrade pip`
4. **Clear pip cache**: `pip cache purge`
5. **Try manual installation**: Follow individual package installation steps
6. **Check system requirements**: Ensure sufficient RAM and disk space

For more help, see:
- PyTorch installation guide: https://pytorch.org/get-started/locally/
- Flower documentation: https://flower.dev/docs/
- Project issues: Check PROJECT-STATUS.md
