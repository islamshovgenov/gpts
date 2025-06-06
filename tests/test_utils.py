import os
import pytest
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

from loader import load_randomization, load_timepoints, parse_excel_files
from src.data_loader import load_data, DataLoaderError
from stat_tools import log_diff_stats
from docx_tools import cv_log

def test_load_randomization_and_timepoints(tmp_path):
    rand_csv = "Subject,Sequence\n1,TR\n2,RT\n"
    rand_file = tmp_path / "rand.csv"
    rand_file.write_text(rand_csv, encoding="utf-8")

    time_csv = "Code,Time\n00,0\n01,1\n"
    time_file = tmp_path / "time.csv"
    time_file.write_text(time_csv, encoding="utf-8")

    rand = load_randomization(rand_file)
    times = load_timepoints(time_file)
    assert rand == {1: "TR", 2: "RT"}
    assert times == {"00": 0, "01": 1}

def make_ct_excel(path):
    df = pd.DataFrame({
        "Subject": [1, 2],
        "Time": [1, 1],
        "Concentration, Period 1": [10, 0],
        "Concentration, Period 2": [0, 10],
    })
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="CT", index=False, startrow=28)

def test_parse_excel_files_ct(tmp_path):
    xls_path = tmp_path / "sample.xlsx"
    make_ct_excel(xls_path)
    rand = {1: "TR", 2: "RT"}
    time_dict = {"01": 1}
    df = parse_excel_files([str(xls_path)], rand, time_dict)
    assert len(df) == 4
    assert set(df["Treatment"]) == {"Test", "Ref"}

def test_data_loader_success(tmp_path):
    rand_file = tmp_path / "rand.csv"
    rand_file.write_text("Subject,Sequence\n1,TR\n2,RT\n", encoding="utf-8")
    time_file = tmp_path / "time.csv"
    time_file.write_text("Code,Time\n01,1\n", encoding="utf-8")
    xls_path = tmp_path / "data.xlsx"
    make_ct_excel(xls_path)
    df, rand_dict, time_dict = load_data(rand_file, time_file, [str(xls_path)])
    assert df.shape[0] == 4
    assert rand_dict[1] == "TR" and rand_dict[2] == "RT"
    assert time_dict == {"01": 1}

def test_data_loader_sequence_mismatch(tmp_path):
    rand_file = tmp_path / "rand.csv"
    rand_file.write_text("Subject,Sequence\n1,TR\n2,TR\n", encoding="utf-8")
    time_file = tmp_path / "time.csv"
    time_file.write_text("Code,Time\n01,1\n", encoding="utf-8")
    xls_path = tmp_path / "data.xlsx"
    make_ct_excel(xls_path)
    with pytest.raises(DataLoaderError):
        load_data(rand_file, time_file, [str(xls_path)])

def test_log_diff_stats():
    df = pd.DataFrame({
        "Cmax_Test": [2.0],
        "Cmax_Ref": [1.0],
        "AUC0-t_Test": [2.0],
        "AUC0-t_Ref": [1.0],
        "AUC0-inf_Test": [2.0],
        "AUC0-inf_Ref": [1.0],
    })
    res = log_diff_stats(df.copy())
    assert np.isclose(res.loc[0, "log_Cmax_diff"], np.log(2))
    assert np.isclose(res.loc[0, "log_AUC0-t_diff"], np.log(2))
    assert np.isclose(res.loc[0, "log_AUC0-inf_diff"], np.log(2))

def test_cv_log():
    s = pd.Series([1.0, 2.0, 3.0])
    cv = cv_log(s)
    assert cv > 0
