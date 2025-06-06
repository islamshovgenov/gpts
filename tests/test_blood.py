import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from blood_analysis import extract_oak_pairwise, extract_individual_lab
from stat_tools import compute_group_dynamics, compute_group_iqr, compute_vitals_iqr


def test_extract_oak_pairwise():
    df = pd.DataFrame({
        "№ п/п": [1, 1, 2, 2],
        "Этап регистрации": ["Скрининг", "в конце периода 2", "Скрининг", "в конце периода 2"],
        "Гемоглобин, г/л": [100, 120, 110, 130],
    })
    res = extract_oak_pairwise(df, "Гемоглобин, г/л")
    assert list(res.columns) == ["Subject", "Before", "After"]
    assert len(res) == 2
    assert res.loc[0, "Before"] == 100
    assert res.loc[0, "After"] == 120


def test_extract_individual_lab_parse_range():
    df = pd.DataFrame({
        "№ п/п": [1, 1, 2, 2],
        "Этап регистрации": ["Скрининг", "Период 2 - в конце", "Скрининг", "Период 2 - в конце"],
        "После приема": ["T", "T", "R", "R"],
        "АЛТ": ["10-20", 25, 15, 30],
    })
    res = extract_individual_lab(df, "АЛТ", subject_col="№ п/п", group_col="После приема")
    assert len(res) == 2
    assert np.isclose(res.loc[0, "before"], 15)
    assert res.loc[0, "after"] == 25


def test_compute_group_dynamics():
    df = pd.DataFrame({
        "№ п/п": [1, 1, 2, 2],
        "Этап регистрации": ["Скрининг", "Период 2 - в конце", "Скрининг", "Период 2 - в конце"],
        "После приема": ["T", "T", "R", "R"],
        "ALT": [10, 20, 10, 15],
    })
    res = compute_group_dynamics(df, "ALT", subject_col="№ п/п", stage_col="Этап регистрации", group_col="После приема")
    assert list(res.columns) == ["После приема", "baseline", "followup"]
    assert np.isclose(res.loc[0, "baseline"], 10)


def test_compute_group_iqr():
    df = pd.DataFrame({
        "№ п/п": [1, 1, 2, 2],
        "Этап регистрации": ["Скрининг", "Период 2", "Скрининг", "Период 2"],
        "После приема": ["T", "T", "R", "R"],
        "ALT": [10, 20, 15, 30],
    })
    res = compute_group_iqr(df, "ALT", subject_col="№ п/п", stage_col="Этап регистрации", group_col="После приема")
    stages = set(res["stage"])
    assert {"baseline", "followup"} <= stages


def test_compute_vitals_iqr():
    df = pd.DataFrame({
        "Препарат": ["T", "T", "R", "R"],
        "Этап регистрации": ["A", "A", "A", "A"],
        "param": [1.0, 2.0, 3.0, 4.0],
    })
    res = compute_vitals_iqr(df, "param")
    assert len(res) == 2
    assert set(res.columns) == {"Препарат", "Этап регистрации", "q1", "median", "q3"}
