import io
import pandas as pd
import types
from loader import load_randomization, load_timepoints, parse_excel_files

class DummyXLS:
    def __init__(self, sheets):
        self.sheet_names = list(sheets.keys())
        self.sheets = sheets
    def parse(self, sheet, header=0):
        return self.sheets[sheet]

def test_load_basic():
    rand_csv = io.StringIO("Subject,Sequence\n1,TR\n2,RT\n")
    time_csv = io.StringIO("Code,Time\n01,0\n02,1\n")
    rand = load_randomization(rand_csv)
    tdict = load_timepoints(time_csv)
    assert rand[1] == "TR"
    assert tdict["02"] == 1

def test_parse_excel_ct(monkeypatch):
    df_ct = pd.DataFrame({
        'Subject':[1],
        'Time':[0],
        'Concentration, Period 1':[0.1],
        'Concentration, Period 2':[0.2],
    })
    monkeypatch.setattr(pd, 'ExcelFile', lambda *_a, **_k: DummyXLS({'CT': df_ct}))
    files = [types.SimpleNamespace(read=lambda: b'dummy')]
    res = parse_excel_files(files, {1:'TR'}, {'00':0})
    assert res.iloc[0]['Concentration'] == 0.1

def test_parse_excel_standard(monkeypatch):
    df_sheet = pd.DataFrame({
        'SampleLabel':['A-1-01','B-1-01'],
        'Calc. Conc., ug/ml':[0.1,0.2],
    })
    monkeypatch.setattr(pd, 'ExcelFile', lambda *_a, **_k: DummyXLS({'Sheet1': df_sheet}))
    files = [types.SimpleNamespace(read=lambda: b'dummy')]
    res = parse_excel_files(files, {1:'TR'}, {'01':0})
    assert len(res) == 2

