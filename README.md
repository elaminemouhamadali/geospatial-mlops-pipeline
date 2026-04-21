# Geospatial MLOps Pipeline

> Production-grade, contract-driven MLOps framework for building, validating, and promoting machine learning models at scale.
> Designed to simulate real-world ML production systems with automated validation, promotion, and reproducible pipelines.
---

## 🚀 Why This Project Exists

Modern ML systems rarely fail because of model architecture — they fail because of:

- ❌ Non-reproducible data pipelines  
- ❌ Leakage between training and evaluation  
- ❌ Manual and subjective model promotion  
- ❌ Lack of standardized evaluation  
- ❌ Poor integration between research and production  

This repository addresses these challenges by providing a **production-grade ML lifecycle system** that enforces:

- Deterministic data and feature pipelines  
- Reproducible training and evaluation  
- Automated, metric-driven model promotion  
- Scalable orchestration using Kubernetes-native workflows  

> Originally designed for geospatial ML, but built as a **task-agnostic ML platform**.

At the core of this system is a contract-driven architecture that ensures reproducibility, auditability, and decoupling across all pipeline stages.

---

## ⚡ System in Action

### 📈 Experiment Tracking (MLflow)

> Example of experiment tracking and metric evolution across multiple training runs.
![MLflow Runs](./docs/mlflow_runs.png)
![MLflow Metrics](./docs/mlflow_metrics.png)

- Tracks experiments, metrics, and artifacts  
- Enables reproducibility and comparison  
- Logs full lineage across pipeline stages  

---

### 📦 Model Registry & Versioning
> Versioned model promotion workflow showing candidate → production transitions.
![MLflow Registry](./docs/model_registry.png)

- Versioned models  
- Candidate vs production separation  
- Promotion workflow with audit trail  
- Produces deployment-ready artifacts suitable for integration into real-time or batch inference systems

---

### 🧩 Pipeline Orchestration (Argo)
> Full pipeline execution DAG orchestrated via Argo, including gating and promotion steps.
![Argo](./docs/Argo.png)

- DAG-based execution  
- Containerized tasks  
- Kubernetes-native orchestration  
- Supports large-scale distributed workflows  

> The pipeline enforces strict validation gates — runs that fail performance thresholds are automatically rejected and never promoted.

---

## 🧪 Example Workflow

1. Generate dataset tiles from raw imagery  
2. Create deterministic splits (`split.json`)  
3. Train model (e.g., SegFormer)  
4. Log metrics and artifacts to MLflow  
5. Apply Gate A → select candidate model  
6. Evaluate on golden test set  
7. Apply Gate B → enforce release thresholds  
8. Register model in MLflow  
9. Promote to production-ready version  

> Supports **multi-run experimentation, automated promotion, and full lineage tracking**.

---

## 🔄 End-to-End ML Lifecycle

This system implements a full production ML lifecycle:

1. **Data/Feature Engineering** (Tiling + Dataset Contracts)  
2. **Dataset Splitting** (Leakage-aware, reproducible)  
3. **Training** (Containerized, scalable)  
4. **Validation (Gate A)** (Candidate selection)  
5. **Evaluation (Golden Test)** (True performance measurement)  
6. **Release Decision (Gate B)** (Automated promotion criteria)  
7. **Model Registry & Promotion**  

Each stage produces **versioned artifacts**, enabling full lineage and reproducibility.
This framework enables seamless transition from research experiments to production-ready systems by standardizing interfaces between data science and infrastructure layers.

---

## 🏗️ High-Level Architecture

![Architecture](./docs/mermaid-diagram-architecture.png)

*Figure: End-to-end ML system architecture from raw data to production-ready models*

---

## ⚙️ Pipeline Stages

### 1. DataOps & Feature Pipeline
- Converts raw geospatial data into structured datasets  
- Generates tiles and metadata  
- Creates **train / validation / test splits**  
- Prevents data leakage  
- Acts as a **feature engineering pipeline**

---

### 2. Training
- Fully containerized execution  
- Consumes immutable dataset contracts  
- Logs metrics, artifacts, and lineage to MLflow  

---

### 3. Validation (Gate A)
- Evaluates models on validation data  
- Selects best-performing candidate  
- Ensures fair experiment comparison  

---

### 4. Evaluation (Golden Test)
- Runs on strictly held-out dataset  
- Measures real-world performance  
- Produces auditable evaluation artifacts  

---

### 5. Model Validation & Release (Gate B)
- Converts evaluation metrics into pass/fail decisions  
- Prevents regressions  
- Enforces production-quality thresholds  

---

### 6. Model Registry & Promotion
- Registers approved models in MLflow  
- Tracks versions and aliases (candidate, production)  
- Produces **deployment-ready artifacts**

---

## 🧩 Core Design Principles

### 🔒 Contract-Based Architecture

![Contracts](./docs/mermaid-diagram-contracts.png)

Each stage communicates via structured contracts:

- `tiles_manifest.json`
- `split.json`
- `eval.json`
- `gate.json`

This enables:
- Reproducibility  
- Auditability  
- Loose coupling between stages  

---

### 🔄 Deterministic Pipelines
- Same input → same output  
- Fully reproducible across environments  
- Containerized execution  

---

### 🧪 Separation of Concerns

| Stage | Responsibility |
|------|--------|
| Validation (Gate A) | Candidate selection |
| Evaluation | Performance measurement |
| Gate B | Production readiness decision |

---

### 🚦 Automated Model Validation

![Gating](./docs/mermaid-diagram-gating.png)

- Objective, metric-driven decisions  
- Eliminates manual bias  
- Prevents performance regressions  

> This mirrors real-world ML systems where models must pass strict validation before deployment.

---

## 🏗️ MLOps Stack

- **Orchestration:** Argo Workflows  
- **Distributed Training:** Ray / KubeRay  
- **Experiment Tracking:** MLflow  
- **CI/CD:** GitLab CI  
- **Infrastructure:** Kubernetes  

---

## ⚡ Scalability & Performance

- Distributed training via Ray  
- Kubernetes-native execution  
- Parallel experimentation  
- Efficient processing of large-scale datasets  

---

## 🧩 Extensible ML Platform

This framework is **task-agnostic**:

- Task adapters define domain-specific logic  
- Core pipeline remains unchanged  
- New ML tasks plug into the same lifecycle  

Supports:
- Segmentation  
- Classification  
- Detection  
- Future generative workflows  

---

## 💡 What This Project Demonstrates

- End-to-end ML lifecycle ownership  
- Production-grade MLOps system design  
- Scalable pipeline orchestration  
- Strong separation between research and production  
- Reproducibility and model governance  

---

## 📁 Repository Structure

```bash
core/
  contracts/        # Data and evaluation contracts
  tiling/           # Tiling engine and policies
  data/             # Dataset and splitting logic
  train/            # Training engine

tasks/
  segmentation/
    building/
    water/

cli/
  generate_tiles.py
  make_splits.py
  train.py
  evaluate.py

```

----

## 🔮 Future Work
- Automated retraining and drift detection
- Online inference and monitoring
- Feature store integration
- Cross-modal generative pipelines

---

## 🤝 Contributions
This project is designed to be modular and extensible. Contributions are welcome.