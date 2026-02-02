# LUFNet-CPC2026

This repository provides **reproducible workflows for training and evaluating
neural-network–driven molecular dynamics models** on **2D and 3D Lennard–Jones (LJ)
systems**, as described in the accompanying *Computer Physics Communications* paper:

**Scalable Neural-Network–Driven Molecular Dynamics Simulation**  
*Computer Physics Communications* (2026)  
DOI: https://doi.org/10.1016/j.cpc.2026.110036

---

## Version information

Please use the **`main` branch**, which contains the latest stable version of the code.
Other branches are retained **for backup and development purposes only**.

- Current version: **v1.0**

---

## Repository overview

- **2D Lennard–Jones (LUFNet)**  
  The 2D LJ workflow and associated thumbnail datasets are **not included** in this
  release and will be added in a future version (expected May 2026).

- **3D Lennard–Jones (LUFNet)**  
  This release contains the complete 3D LJ workflow, including:
  - training and evaluation code
  - example (thumbnail) datasets
  - reproducibility and checkpoint-comparison utilities

---

## Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Linux (tested on Ubuntu)  
- CUDA-enabled GPU (optional but recommended)

Exact dependencies are listed in `requirements.txt` and can be installed via:

```bash
pip install -r requirements.txt
```
---

## Typical workflow

1. Inspect or test the pipeline using example data (`thumbnail_data/`)
2. Configure training parameters in `train_config.yaml`
3. Train a model using this script: `train_main.py`
4. Evaluate the trained model using this script: `test_main.py`
5. (Optional) Compare checkpoints against a baseline for reproducibility

---

## Directory structure (3D Lennard–Jones)

```
3D_LJ_Potential/
├── code/
│   ├── train_main.py
│   ├── test_main.py
│   ├── train_config.yaml
│   ├── test_config.yaml
│   ├── compare_with_baseline.py
│   ├── utils/
│   ├── data_loader/
│   ├── ML/
│   ├── hamiltonian/
│   ├── MD/
│   └── MC/
└── thumbnail_data/
```

---

## Quick start

```bash
cd 3D_LJ_Potential/code
python train_main.py
python test_main.py
```

- Training parameters are read from `train_config.yaml`
- Evaluation parameters are read from `test_config.yaml`
- Outputs are written to `results/`

---

## Data

- **`thumbnail_data/`**  
  Small example datasets sufficient for pipeline testing and debugging.

- **Full dataset of 3D Lennard–Jones**  
  The full dataset (MD/MC trajectories, energies, and forces) is deposited on Zenodo.  
  DOI: (to be added)

Update dataset paths in `train_config.yaml` to use the full dataset.

---

## Citation

```bibtex
@article{LUFNetCPC2026,
  title   = {Scalable Neural-Network--Driven Molecular Dynamics Simulation},
  journal = {Computer Physics Communications},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110036}
}
```

---

## Contact

For questions or issues related to this repository, please contact the authors listed in the corresponding paper.
