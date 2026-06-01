# Confluencia Platform

## Module Separation

This project now has two independent modules with dedicated frontends:

| Module | Focus | Frontend | Run Command |
|--------|-------|----------|-------------|
| **Drug 2.0** | Small molecule drugs | `confluencia-2.0-drug/app_drug.py` | `streamlit run confluencia-2.0-drug/app_drug.py` |
| **circRNA** | circRNA vaccines | `confluencia_circrna/app.py` | `streamlit run confluencia_circrna/app.py` |

---

## Quick Start

### Option 1: Use Launcher

```bash
# Windows
start_confluencia.bat

# Linux/Mac
./run.sh
```

### Option 2: Direct Run

```bash
# Drug Module
cd confluencia-2.0-drug
streamlit run app_drug.py

# circRNA Module
cd confluencia_circrna
streamlit run app.py
```

---

## Drug Module Features

**Small Molecule Drug Discovery**

| Page | Features |
|------|----------|
| 🧪 Molecule Input | SMILES input, validation, batch upload |
| 📊 ADMET Prediction | Absorption/Distribution/Metabolism/Excretion/Toxicity |
| 🧬 ED2Mol Generation | Structure-based molecule design |
| 📈 PK/PD Simulation | Pharmacokinetic/Pharmacodynamic modeling |
| 🧪 Molecule Evolution | ED2Mol + mutation + Pareto optimization |
| 🎯 Target Docking | Molecule-target binding prediction |

**Core Modules:**
- `ed2mol_adapter.py` - ED2Mol integration
- `evolution.py` - Molecule evolution
- `admet.py` - ADMET prediction
- `pkpd.py` - PK/PD simulation
- `docking.py` - Target docking

---

## circRNA Module Features

**circRNA Vaccine Development**

| Page | Features |
|------|----------|
| 📊 Sequence Analysis | RIG-I/TLR/PKR scoring, structure prediction, modifications |
| 🧪 Sequence Design | Evolution optimization, IRES designer, modification selector |
| 💉 Vaccine Development | IPS scoring, drug response, treatment recommendation |
| 📋 Clinical Report | Survival analysis, adverse events, PDF report |

**Core Modules:**
- `immune_sensing.py` - RIG-I/TLR/PKR (literature-based)
- `structure_prediction.py` - ViennaRNA structure
- `folding_kinetics.py` - Folding dynamics
- `cirrna_evolution.py` - Sequence evolution
- `rna_modifications.py` - m6A/IRES/miRNA/RBP
- `clinical_prediction.py` - Survival/biomarkers/AE

---

## Dependencies

### Drug Module
```bash
pip install streamlit rdkit pandas numpy plotly sklearn
```

### circRNA Module
```bash
pip install streamlit pandas numpy plotly

# Optional: ViennaRNA (Linux)
apt-get install vienna-rna
```

---

## Project Structure

```
Confluencia/
│
├── confluencia-2.0-drug/          # Drug Module
│   ├── app_drug.py                 # Drug frontend
│   └── core/
│       ├── ed2mol_adapter.py       # ED2Mol
│       ├── evolution.py            # Molecule evolution
│       ├── admet.py                 # ADMET
│       ├── pkpd.py                  # PK/PD
│       └── docking.py               # Target docking
│
├── confluencia_circrna/           # circRNA Module
│   ├── app.py                      # circRNA frontend
│   └── core/
│       ├── immune_sensing.py       # RIG-I/TLR/PKR
│       ├── structure_prediction.py # ViennaRNA
│       ├── folding_kinetics.py     # Kinetics
│       ├── cirrna_evolution.py     # Evolution
│       ├── rna_modifications.py    # m6A/IRES/miRNA
│       ├── drug_response.py        # Vaccine efficacy
│       ├── clinical_prediction.py  # Clinical outcomes
│       └── rna_docking.py          # RNA-drug docking
│
├── start_confluencia.bat          # Windows launcher
├── run.sh                          # Linux/Mac launcher
│
└── studio-electron/                # Electron desktop app (optional)
```

---

## AutoDL Deployment

```bash
# Pull latest code
git pull origin main

# Install dependencies
pip install streamlit pandas numpy plotly

# Run desired module
streamlit run confluencia_circrna/app.py
# or
streamlit run confluencia-2.0-drug/app_drug.py
```

---

## Module Comparison

| Feature | Drug Module | circRNA Module |
|---------|-------------|----------------|
| Input Type | SMILES strings | RNA sequences |
| Analysis | ADMET, PK/PD | Immune scores, structure |
| Generation | ED2Mol (structure-based) | Evolution (sequence) |
| Prediction | Binding, toxicity | Immunogenicity, survival |
| Clinical | Drug efficacy | Vaccine response |

---

## License

MIT License - See LICENSE file

## Contact

GitHub: https://github.com/IGEM-FBH/confluencia