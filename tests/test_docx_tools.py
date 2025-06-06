import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import pandas as pd
import numpy as np
from pathlib import Path

from docx_tools import (
    cv_log,
    export_power_analysis_table,
    export_individual_pk_tables,
)


def test_cv_log_invalid():
    s = pd.Series([1.0, 0.0, 2.0])
    assert np.isnan(cv_log(s))


def test_export_power_analysis_table(tmp_path):
    out = tmp_path / "power.docx"
    path = export_power_analysis_table([1.0, 1.0, 1.0], [10, 20, 30], n=12, save_path=str(out))
    assert Path(path).is_file()


def test_export_individual_pk_tables(tmp_path):
    df = pd.DataFrame({
        "Subject": [1, 2],
        "Treatment": ["Test", "Ref"],
        "AUC0-t": [100, 110],
        "AUC0-inf": [120, 130],
        "Cmax": [10, 9],
        "Tmax": [1, 1],
        "T1/2": [2.0, 2.1],
        "Kel": [0.3, 0.3],
        "N_el": [5, 5],
        "MRT": [12, 13],
        "Tlag": [0, 0],
        "Vd": [20, 21],
        "CL": [1.0, 1.1],
    })
    out = tmp_path / "ind_pk.docx"
    path = export_individual_pk_tables(df, "Tst", "Ref", "Sub", 100, 100, save_path=str(out))
    assert Path(path).is_file()


def test_export_log_ci_and_be_tables(tmp_path):
    df = pd.DataFrame({
        'log_AUC0-t_diff': [0.1],
        'log_AUC0-inf_diff': [0.1],
        'log_Cmax_diff': [0.2],
        'SE_log_AUC0-t': [0.05],
        'SE_log_AUC0-inf': [0.05],
        'SE_log_Cmax': [0.07],
        'MSE_log_AUC0-t': [0.001],
        'MSE_log_AUC0-inf': [0.001],
        'MSE_log_Cmax': [0.002],
    })
    from docx_tools import export_log_ci_tables, export_be_result_table
    ci_path = export_log_ci_tables(df, 'Sub', save_path=str(tmp_path / 'ci.docx'))
    assert Path(ci_path).is_file()

    be_path = export_be_result_table([1.0, 1.1, 1.2], [0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [10, 15, 20], save_path=str(tmp_path / 'be.docx'))
    assert Path(be_path).is_file()
