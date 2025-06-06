import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from stat_tools import (
    ci_calc,
    calc_swr,
    make_stat_report_table,
    add_mse_se_to_pivot,
    identify_be_outlier_and_recommend,
    generate_conclusions,
)


def test_ci_calc_basic():
    log_diff = pd.Series([0.0, np.log(2)])
    ratio, lo, hi = ci_calc(log_diff)
    assert ratio > 1
    assert lo < ratio < hi


def test_calc_swr():
    logs = np.log([1.0, 2.0, 4.0])
    swr, cv = calc_swr(logs)
    assert swr > 0
    assert cv > 0


def test_make_stat_report_table():
    df = make_stat_report_table([1.0, 1.1, 1.2], [0.9, 1.0, 1.1], [1.1, 1.2, 1.3], [10, 15, 20])
    assert df.shape[0] == 3
    assert "90% ДИ" in df.columns


def test_add_mse_se_to_pivot():
    pivot = pd.DataFrame({"log_Cmax_diff": [0.1, 0.2, 0.3]})
    res = add_mse_se_to_pivot(pivot)
    assert "MSE_log_Cmax" in res.columns
    assert "SE_log_Cmax" in res.columns


def test_identify_be_outlier_and_recommend():
    df_conc = pd.DataFrame({
        "Subject": [1, 1, 2, 2],
        "Time": [0, 1, 0, 1],
        "Treatment": ["Test", "Test", "Test", "Test"],
        "Concentration": [10, 20, 10, 30],
    })
    pk_df = pd.DataFrame({
        "Subject": [1, 1, 2, 2],
        "Treatment": ["Test", "Ref", "Test", "Ref"],
        "Cmax": [100, 80, 80, 80],
    })
    outlier, rec_df = identify_be_outlier_and_recommend(df_conc, pk_df, "Cmax", 0.7, 0.9)
    assert outlier == 1
    assert set(rec_df.columns) == {"Subject", "Time", "Recommended_Concentration"}


def test_generate_conclusions():
    df = pd.DataFrame({
        "Subject": [1, 2],
        "group": ["T", "R"],
        "before": [10, 20],
        "after": [15, 10],
    })
    res = generate_conclusions(df, "ALT")
    assert res["T"]
    assert res["R"]

