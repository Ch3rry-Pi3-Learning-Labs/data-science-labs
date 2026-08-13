# CH3RRY PI3 Data Science Learning Labs

Friendly, mathematically careful and reproducible data-science tutorials from CH3RRY PI3.

Each Learning Lab builds a method gradually:

1. establish the problem and intuition;
2. define every mathematical symbol;
3. translate the mathematics into focused code;
4. work through a small example;
5. assemble the complete method; and
6. inspect, validate and interpret the result.

## Foundation Learning Labs

- [Gradient Descent: From Intuition to Implementation](./notebooks/01-foundations/gradient-descent/01-gradient-descent-from-intuition-to-implementation.ipynb)
- [Feature Scaling: Why Units Change Gradient Descent](./notebooks/01-foundations/feature-scaling/02-feature-scaling-why-units-change-gradient-descent.ipynb)

## Repository structure

```text
data-science-labs/
├── docs/          # House standards and publishing workflow
├── notebooks/     # Published Learning Labs arranged by subject
├── templates/     # Reusable notebook and planning templates
├── tools/         # Reproducible notebook-building and validation tools
└── requirements.txt
```

The initial repository deliberately stays small. New subject areas will be added when completed material justifies them.

## Run locally

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the repository folder in VS Code and select the `.venv` Python environment as the notebook kernel. The workspace setting points to `.venv\Scripts\python.exe`, and the environment itself is excluded from Git.

Validate the reusable template and execute every published notebook from a clean kernel with:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py
```

Published notebooks should run from beginning to end without manual repair.

## Current status

This repository is in its initial pilot stage. The structure and standards will be refined using evidence from completed tutorials and reader feedback.
