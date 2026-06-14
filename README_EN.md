# Confluencia — Multi-task Computational Platform for circRNA Drug Discovery

> **Adaptive Mixture-of-Experts with Pharmacokinetic Dynamics for Small-Sample circRNA Drug Discovery**

[![CI](https://github.com/IGEM-FBH/confluencia/actions/workflows/ci.yml/badge.svg)](https://github.com/IGEM-FBH/confluencia/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)

Confluencia is a multi-task computational platform for circRNA drug discovery, integrating sample-adaptive MOE ensemble learning, RNACTM six-compartment PK model, and Mamba3Lite multi-scale sequence encoder, designed for small-sample (N<300) scenarios.

## Key Features

| Feature | Description |
|---------|-------------|
| **Sample-Adaptive MOE Ensemble** | Automatically selects and weights regression experts (Ridge/HGB/RF/MLP) based on data size |
| **RNACTM PK Model** | First six-compartment PK model for circRNA (injection→LNP→endocytosis→cytosol→translation→clearance) |
| **Mamba3Lite Encoder** | Three time-constant adaptive SSM + four-scale pooling + self-attention enhancement |
| **Bootstrap CI** | t-distribution for small samples, bootstrap percentile for large samples |
| **Stratified K-Fold** | Quantile-based binning for balanced efficacy distribution across folds |

## Core Results

| Metric | Value | Notes |
|--------|-------|-------|
| **288K IEDB AUC (allele-aware)** | **0.80** | HGB with MHC allele encoding |
| **Drug Ridge R²** | **0.984** | Best for small-sample drug prediction |
| **Mamba3Lite+Attn MAE** | **0.395** | Attention-enhanced best encoder config |
| **Drug Binding AUC** | **0.9252** | 5-Fold OOF Ensemble |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/IGEM-FBH/confluencia.git
cd confluencia

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements-shared.txt
```

### Run Tests

```bash
python -m pytest tests/test_shared_modules.py -v
```

### Launch Frontend

```bash
# Epitope module
cd confluencia-2.0-epitope && PYTHONPATH=.. streamlit run epitope_frontend.py

# Drug module
cd confluencia-2.0-drug && PYTHONPATH=.. streamlit run app.py
```

### CLI Usage

```bash
pip install -e .
confluencia --help
confluencia interactive  # Launch interactive REPL
```

## Architecture

```
Confluencia/
├── confluencia-2.0-drug/         # Drug prediction module
├── confluencia-2.0-epitope/      # Epitope prediction module
├── confluencia_shared/           # Shared library
├── confluencia_joint/            # Joint evaluation (5D)
├── confluencia_cli/              # Command-line interface
├── confluencia_circrna/          # circRNA analysis
├── benchmarks/                   # Benchmark scripts
└── docs/                         # Paper drafts
```

## Python API

```python
from confluencia_cli.bridge import ConfluenciaBridge

bridge = ConfluenciaBridge()

# PK simulation
pk = bridge.ctm_simulate(dose=200, freq=2, binding=0.72, horizon=72)

# Joint evaluation (5D)
result = bridge.joint_evaluate({
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "epitope_seq": "SLYNTVATL",
    "mhc_allele": "HLA-A*02:01",
    "dose_mg": 200, "freq_per_day": 2, "treatment_time": 72
})

# Train with your own data
result = bridge.drug_train("my_data.csv", model_name="ridge")
print(f"R²={result['r2']:.3f}")
```

## R Package

```r
devtools::install_github("IGEM-FBH/confluencia", subdir = "confluencia-rpkg")

library(confluencia)
cf_use_python("/path/to/python")

# PK simulation
pk <- cf_ctm_simulate(dose = 200, freq = 2, binding = 0.72, horizon = 72)

# Joint evaluation
result <- cf_joint_evaluate(smiles = "CC(=O)Oc1ccccc1C(=O)O",
                             epitope_seq = "SLYNTVATL",
                             mhc_allele = "HLA-A*02:01",
                             dose_mg = 200, freq_per_day = 2)
```

## Plugin System

Extend Confluencia with custom algorithms:

```python
import confluencia_cli.plugins as cf

# Register custom model
@cf.register_model("xgboost")
def create_xgb(**kwargs):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, **kwargs)

# Use custom model
from confluencia_cli.bridge import ConfluenciaBridge
bridge = ConfluenciaBridge()
result = bridge.drug_train("data.csv", model_name="xgboost")
```

## Documentation

- **Full README (Chinese)**: [README_CN.md](README_CN.md)
- **Integration Summary**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
- **Module Documentation**: [README_modules.md](README_modules.md)
- **Application Note**: [README_APPLICATION_NOTE.md](README_APPLICATION_NOTE.md)

## License

MIT License. For research/prototype use only. Code and models are for research demonstration, not clinical advice.

---

**Contact:** igem@fbh-china.org | **Repository:** https://github.com/IGEM-FBH/confluencia
