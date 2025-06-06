# Test Summary

This document outlines the automated tests available in this repository. All tests can be executed with `pytest`. They cover the core modules for pharmacokinetic (PK) analysis, data loading, visualization and document export.

## Application
- **tests/test_app_import.py** – verifies that `app.py` can be imported with a stubbed Streamlit environment and that default sidebar values (e.g., `dose_test`) are initialised correctly.
- **tests/test_app_run.py** – runs the application with dummy loaders and calculators to ensure main logic executes without errors.
- **tests/test_app_e2e.py** – end-to-end test using Streamlit's `AppTest` to exercise sidebar widgets, triggering plotting and export stubs.

## Blood analysis
- **tests/test_blood.py** – checks pairwise extraction, individual lab parsing and group statistics in `blood_analysis.py`.
- **tests/test_blood_extra.py** – covers Excel loaders for OAK, BHAK and OAM sheets as well as vitals and stage order parsing.

## Document exports
- **tests/test_docx_tools.py** – ensures power analysis, individual PK tables, and BE results are written to DOCX files.
- **tests/test_docx_tools_extra.py** – validates auxiliary exports such as AUC residual tables, log-transformed PK tables and SAS ANOVA reports.
- **tests/test_export_utils.py** – checks that high-level export helpers delegate to the underlying functions correctly.

## Data loading
- **tests/test_loader_extra.py** – exercises CSV randomization/time-point loading and both CT-formatted and standard Excel parsing routines.

## PK calculations
- **tests/test_pk.py** – covers AUC/Kel computations and `compute_pk` over a simple dataset.
- **tests/test_pk_workflow.py** – integration test of `compute_pk_and_stats` with helper functions patched.

## Statistics utilities
- **tests/test_stats_advanced.py** – tests `get_gmr_ci_from_model` and `nested_anova` on a minimal dataset.
- **tests/test_stats_extra.py** – validates CI calculation, within-subject variability, outlier detection and conclusion generation.

## Visualization
- **tests/test_viz.py** – ensures all plotting helpers in `viz.py` return matplotlib figures and work with basic data.

## Miscellaneous utilities
- **tests/test_utils.py** – tests CSV loaders, the full data loader, log difference statistics and CV calculation.

Running all tests yields over 80% coverage across the project.
