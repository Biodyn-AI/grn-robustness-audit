# Five Axes of Fragility: A Systematic Robustness Audit of GRN Inference from Single-Cell Foundation Models

This repository contains the data, analysis scripts, and manuscript for a systematic multi-axis robustness audit of gene-regulatory-network (GRN) inference from single-cell transcriptomics.

**Paper:** *Five Axes of Fragility: A Systematic Robustness Audit of Gene-Regulatory-Network Inference from Single-Cell Foundation Models*
**Author:** Ihor Kendiukhov ([ORCID](https://orcid.org/0000-0001-5342-1499))
**Target venue:** *Bioinformatics* (Oxford University Press)

## Key Findings

| Axis | Key Metric | Finding |
|------|-----------|---------|
| **Cell count** | MVCC > 3,000 | Unconstrained rankings require >3,000 cells for stability; prior constraints (DoRothEA/TRRUST) rescue small samples |
| **Resolution** | Dual-null calibration | All five datasets default to coarse clustering after dual-null calibration |
| **Rare types** | Reliability gate | 0% of rare cell types pass the reliability gate vs 18% of abundant types |
| **Donor** | Fixed-holdout Spearman | 6-8 donors needed for immune-tissue generalization; composition explains ~16% of transfer |
| **Integration** | Perturbation swap | Harmony displaces 31-45% of top-500 edges with no perturbation-confirmed improvement |

No dataset passes all five axes under calibrated criteria.

## Repository Structure

```
grn-robustness-audit/
  data/                          Result data from each robustness axis
    axis1_cell_count/            MVCC estimates, edge emergence summary
    axis2_resolution/            Dual-null calibration results
    axis3_rare_types/            Cell-type metrics, minimum cell-size guidelines
    axis4_donor/                 Composition-control split metrics
    axis5_integration/           Integration summary, precision, PCs sweep
  scripts/                       Lightweight figure-only pipeline
    generate_source_panels.py    Regenerate individual panels from CSVs
    generate_figures.py          Compose panels into multi-panel PDF figures
  source_panels/                 Pre-generated panels not derived from CSVs
  implementation/                Full reproduction package (`fragility` Python pkg)
    fragility/                   Preprocessing, panels, scorers, nulls, metrics, axes, CLI
    configs/                     YAML configs for axis runs and work packages
    scripts/                     End-to-end pipeline drivers
    tests/                       Unit tests
    Makefile                     `make audit` regenerates every CSV and figure
    pyproject.toml               Installable package definition
  paper/                         LaTeX manuscript, figures, and revision artifacts
    main.tex                     Main paper
    cover_letter.tex             Submission cover letter
    response_to_reviewers.md     Per-reviewer response with section traceability
    supplementary_methods_s2.md  AI-agent audit protocol and rubric
    generate_revised_figures.py  Regenerate all 14 figure PDFs from CSVs
    figures/                     7 main figure PDFs
    figures/supplementary/       7 supplementary figure PDFs
  Makefile                       Build automation (figure-only pipeline)
  requirements.txt               Python dependencies (figure-only pipeline)
  CITATION.cff                   Citation metadata
  LICENSE                        MIT License
```

## Reproducing the Figures and Paper

### Prerequisites

- Python 3.9+
- A LaTeX distribution (e.g. TeX Live, MacTeX)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Biodyn-AI/grn-robustness-audit.git
cd grn-robustness-audit

# Install Python dependencies
pip install -r requirements.txt

# Reproduce everything: panels, figures, and paper
make all
```

### Step-by-Step

```bash
# 1. Generate individual figure panels from CSV data
python scripts/generate_source_panels.py

# 2. Compose panels into multi-panel PDF figures
python scripts/generate_figures.py

# 3. Compile the LaTeX paper (run twice for cross-references)
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Data Description

All result CSV files are self-contained and can be inspected independently.

| File | Axis | Description |
|------|------|-------------|
| `data/axis1_cell_count/mvcc_summary.csv` | 1 | Conservative MVCC estimates per policy and size tier |
| `data/axis1_cell_count/emergence_summary.csv` | 1 | Edge emergence and DoRothEA support rates |
| `data/axis2_resolution/null_calibration.csv` | 2 | Dual-null calibration: observed vs null overlap and sensitivity |
| `data/axis3_rare_types/cell_type_metrics.csv` | 3 | Per-cell-type stability, null AUC, and reliability metrics |
| `data/axis3_rare_types/minimum_cell_size_guidelines.csv` | 3 | Minimum reliable cell-type size estimates |
| `data/axis4_donor/adversarial_composition_control_top8_split_metrics.csv` | 4 | Donor composition control split-level metrics |
| `data/axis5_integration/integration_summary.csv` | 5 | Rank Spearman and top-k overlap for Harmony variants |
| `data/axis5_integration/reference_precision.csv` | 5 | Precision@k against DoRothEA and TRRUST references |
| `data/axis5_integration/harmony_batch_pcs_sweep.csv` | 5 | Overlap and precision as a function of Harmony PCs |

## Datasets

Analyses use processed single-cell data from:
- **Tabula Sapiens** (kidney, lung, immune tissues): [tabula-sapiens-portal.ds.czbiohub.org](https://tabula-sapiens-portal.ds.czbiohub.org/)
- **Krasnow lung cohort** (external SmartSeq2 lung): see Methods
- **Delorey 2021 COVID lung** (disease cohort, CellxGene): see Methods

## Citation

If you use this code or data, please cite:

```bibtex
@article{Kendiukhov2025fragility,
  title   = {Five Axes of Fragility: A Systematic Robustness Audit of
             Gene-Regulatory-Network Inference from Single-Cell Foundation Models},
  author  = {Kendiukhov, Ihor},
  year    = {2025},
  note    = {Manuscript submitted to Bioinformatics}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
