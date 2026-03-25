# Installing qvartools

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Linux** is required for the `pyscf` optional dependency (PySCF does not support Windows or macOS natively)
- **NVIDIA GPU + CUDA 12.x** drivers for GPU-accelerated workflows (optional)

## Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) provides fast, reproducible dependency resolution.

### CPU-only (default)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install ".[full]"
```

### GPU (CUDA 12.x)

```bash
uv pip install torch
uv pip install ".[full,gpu]"
```

### Development

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[full]" --group dev
```

## Install with pip

### CPU-only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ".[full]"
```

### GPU (CUDA 12.x)

```bash
pip install torch
pip install ".[full,gpu]"
```

### Development (editable)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[full,dev]"
```

## Docker

### CPU image

```bash
docker build -f Dockerfile.cpu -t qvartools:cpu .
docker run --rm qvartools:cpu
```

### GPU image

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -f Dockerfile.gpu -t qvartools:gpu .
docker run --rm --runtime=nvidia --gpus all qvartools:gpu
```

## Verifying the installation

```bash
# Quick import check
python -c "import qvartools; print(qvartools.__version__)"

# Run the test suite
python -m pytest tests/ -v

# Check GPU availability (if applicable)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'qvartools'`

Make sure the package is installed in the active virtual environment:

```bash
pip list | grep qvartools
```

If missing, re-run the install command. If you are developing locally, use the `-e` (editable) flag.

### PySCF fails to install on macOS / Windows

PySCF only supports Linux. Use the Docker CPU image instead, or run inside WSL2 on Windows.

### CUDA version mismatch

Ensure your NVIDIA driver supports CUDA 12.x:

```bash
nvidia-smi   # check the "CUDA Version" line
```

If your driver only supports CUDA 11.x, install the matching PyTorch wheel and replace `cupy-cuda12x` with `cupy-cuda11x`.

### `torch` installation is very slow with pip

pip resolves dependencies sequentially. Switch to `uv` for significantly faster installs:

```bash
pip install uv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### HDF5 header not found during build

Install the system HDF5 development library:

```bash
# Debian / Ubuntu
sudo apt-get install libhdf5-dev

# Fedora / RHEL
sudo dnf install hdf5-devel
```
