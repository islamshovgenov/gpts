## 🧪 Purpose

This repository contains a modular Python application for performing **bioequivalence analysis** based on concentration-time data from clinical trials. It supports pharmacokinetic (PK) parameter calculation, statistical testing (ANOVA, CV, GMR, CI), subject-level and aggregate plots, and Word report generation.

---

## 📁 Project Structure
.
├── app.py # Streamlit app for BE analysis
├── blood_analysis.py # Lab and clinical safety data evaluation
├── data_loader.py # Wrapper for data validation logic
├── docx_tools.py # Exports reports and statistical tables in .docx
├── loader.py # Raw Excel/CSV parsing, CT file support
├── pk.py # PK calculations (AUC, Cmax, T1/2, etc.)
├── pk_workflow.py # Combines PK, ANOVA, BE logic into one function
├── stat_tools.py # Statistical methods: CI, ANOVA, sWR, CV, GMR
├── viz.py # Low-level plotting functions (matplotlib)
├── plot_utils.py # Composite visualization logic for Streamlit
├── export_utils.py # Export helpers for UI-triggered document generation
├── requirements.txt # Dependencies
├── readme.md # General README for humans
├── AGENTS.md # Instructions for Codex agents
├── src/ # Modularized imports (e.g., data_loader, pk_workflow)
└── tests/ # Unit tests (pytest + coverage)


---

## ⚙️ Entry Point

Launch app with:

```bash
streamlit run app.py


Testing Instructions
Run all tests:

bash
Копировать
Редактировать
pytest --cov=. --cov-report=term
Minimum required coverage: 50%

Recommended coverage targets:

pk.py, stat_tools.py, pk_workflow.py > 85%

Generate visual report:

bash
Копировать
Редактировать
coverage html
start htmlcov/index.html
📦 Dependencies
Install with:

bash
Копировать
Редактировать
pip install -r requirements.txt
Required libraries:

streamlit

pandas, numpy

scipy, statsmodels

matplotlib

python-docx, openpyxl

🧠 Agent Behavior (Codex)
Use app.py as main orchestration point

When modifying or testing:

Prefer patching functions in pk.py, stat_tools.py

Use pk_workflow.compute_pk_and_stats() as integration test

Respect Streamlit component boundaries: no UI logic in utility modules

For report generation, use export_utils.export_be_tables(...)

🔐 Environment & Secrets
No API keys or credentials are required

All data is local or uploaded via Streamlit interface

If automating tests, use dummy .csv/.xlsx files in /tests/assets/

📎 Notes
App expects correct mapping of subjects and timepoints via randomization and sampling files

Analytical files can be in CT-format or per-sheet format with sample labels like A-01-02
