# Dependency, Requirements, and Docker Image Strategy

This document explains how this repo manages Python dependencies and how those dependencies map to  Docker images.

The goal is to keep the project installable, reproducible, and flexible enough to support:

- slim CPU images for lightweight pipeline stages
- CPU ML images for training/evaluation without GPUs
- Ray images for distributed local or cluster execution
- GPU images that build on top of CUDA/PyTorch base images

---

## 1. Dependency groups

The repo intentionally separates lightweight runtime dependencies from heavier ML, Ray, and dev dependencies.

### Base dependencies

The base dependencies are installed with:

```bash
python -m pip install .
```

or, for development:

```bash
python -m pip install -e .
```

These dependencies support the core pipeline code:

- contracts
- CLI utilities
- config loading
- tiling/splitting
- geospatial IO
- MLflow integration
- lightweight pipeline stages

Base dependencies should avoid heavy packages like PyTorch and Ray unless absolutely required.

### `ml` extra

Installed with:

```bash
python -m pip install -e ".[ml]"
```

This includes:

- PyTorch
- torchvision
- Transformers

Use this for single-node model training, evaluation, and inference.

### `ray` extra

Installed with:

```bash
python -m pip install -e ".[ray]"
```

This includes Ray dependencies.

In practice, Ray model execution usually needs both `ml` and `ray`:

```bash
python -m pip install -e ".[ml,ray]"
```

### `models` extra

Installed with:

```bash
python -m pip install -e ".[models]"
```

This includes model-library dependencies such as Transformers, but intentionally does not install PyTorch.

This is useful for GPU Docker images that already start from a PyTorch CUDA base image.

### `dev` extra

Installed with:

```bash
python -m pip install -e ".[dev]"
```

This includes developer tools such as:

- pytest
- ruff
- mypy
- pre-commit

---

## 2. Docker image mapping

The dependency split enables multiple image types.

### Slim CPU image

Purpose:

- contracts
- CLIs
- tiling
- splitting
- gating
- metadata inspection
- lightweight MLflow logic

Install command:

```dockerfile
RUN python -m pip install -r requirements.txt
```

Equivalent to:

```bash
python -m pip install .
```

### CPU ML image

Purpose:

- single-node training
- single-node evaluation
- model inference
- CPU model debugging

Install command:

```dockerfile
RUN python -m pip install -r requirements-ml.txt
```

Equivalent to:

```bash
python -m pip install -e ".[ml]"
```

### CPU Ray image

Purpose:

- local Ray cluster testing
- Ray Train on CPU
- Ray evaluation/inference on CPU
- fake multinode Docker Compose cluster

Install command:

```dockerfile
RUN python -m pip install -r requirements-ray.txt
```

Equivalent to:

```bash
python -m pip install -e ".[ml,ray]"
```

### GPU Ray image

Purpose:

- GPU training
- GPU inference
- GPU Ray workers
- future KubeRay/Argo execution

Base image:

```dockerfile
FROM pytorch/pytorch:<cuda-runtime-tag>
```

Install command:

```dockerfile
RUN python -m pip install -r requirements-gpu.txt
```

Equivalent to:

```bash
python -m pip install -e ".[models,ray]"
```

The GPU image relies on the base image for PyTorch/CUDA.

### Dev image or CI install

Purpose:

- linting
- unit tests
- packaging checks
- import tests

Install command:

```dockerfile
RUN python -m pip install -r requirements-dev.txt
```

If tests require ML/Ray imports, use:

```dockerfile
RUN python -m pip install -r requirements-all.txt
```

---

## 7. Validation commands

Before building Docker images, validate each dependency profile in a clean environment.

### Base install

```bash
conda create -n mlops-base-test python=3.11 -y
conda activate mlops-base-test
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import geo_mlops, rasterio, mlflow; print('base OK')"
```

### ML install

```bash
conda create -n mlops-ml-test python=3.11 -y
conda activate mlops-ml-test
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -c "import geo_mlops, torch, torchvision, transformers; print('ml OK')"
```

### Ray install

```bash
conda create -n mlops-ray-test python=3.11 -y
conda activate mlops-ray-test
python -m pip install --upgrade pip
python -m pip install -r requirements-ray.txt
python -c "import geo_mlops, torch, transformers, ray; print('ray OK')"
```

### Full local dev install

```bash
conda create -n mlops-all-test python=3.11 -y
conda activate mlops-all-test
python -m pip install --upgrade pip
python -m pip install -r requirements-all.txt
python -c "import geo_mlops, torch, transformers, ray, pytest; print('all OK')"
pytest -q
```

If all profiles install cleanly, the repo is ready for Docker image work.

---

## 8. Recommended file layout

Keep these files at the repository root:

```text
pyproject.toml
requirements.txt
requirements-dev.txt
requirements-ml.txt
requirements-ray.txt
requirements-all.txt
requirements-gpu.txt
```

Reason:

- Python packaging tools expect `pyproject.toml` at the project root.
- `pip install .` expects to run from the root containing `pyproject.toml`.
- Dockerfiles and CI pipelines commonly install from root-level requirements files.
- Keeping these files at root makes the repo easier for other engineers, CI systems, and Docker builds to understand.

Do not move `pyproject.toml` into a subdirectory unless the actual Python package root also moves.

If the number of requirements files grows later, they can be moved into a directory such as:

```text
requirements/
  base.txt
  dev.txt
  ml.txt
  ray.txt
  all.txt
  gpu.txt
```

However, for the current repo, root-level files are simpler and more standard.

---

## 10. Summary

The intended dependency strategy is:

```text
pyproject.toml
  source of truth

requirements*.txt
  small convenience wrappers around pyproject extras

Dockerfiles
  choose the right requirements file depending on image type

slim image
  requirements.txt

CPU ML image
  requirements-ml.txt

CPU Ray image
  requirements-ray.txt

GPU Ray image
  PyTorch CUDA base image + requirements-gpu.txt

dev/CI
  requirements-dev.txt or requirements-all.txt
```

This keeps the repo lightweight by default while still supporting training, Ray, and GPU execution when needed.
