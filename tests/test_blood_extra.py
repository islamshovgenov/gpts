import io
import pandas as pd
import types
from blood_analysis import load_oak_sheet, load_bhak_sheet, load_oam_sheet, load_vitals_sheet, load_stage_order


def test_load_oak_bhak(monkeypatch):
    monkeypatch.setattr('blood_analysis.OAK_PARAMS', ['Гемоглобин, г/л'])
    monkeypatch.setattr('blood_analysis.BHAK_PARAMS', ['АЛТ, Ед./л'])
    df = pd.DataFrame({
        ('№ п/п',''): [1],
        ('Этап регистрации',''): ['Скрининг'],
        ('После приема',''): ['T'],
        ('Гемоглобин, г/л','значе-ние'): [100],
    })
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: df)
    res = load_oak_sheet(io.BytesIO())
    assert 'Гемоглобин, г/л' in res.columns

    df2 = pd.DataFrame({
        ('№ п/п',''): [1],
        ('Этап регистрации',''): ['Скрининг'],
        ('После приема',''): ['T'],
        ('АЛТ, Ед./л','значе-ние'): [10],
    })
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: df2)
    res2 = load_bhak_sheet(io.BytesIO())
    assert 'АЛТ, Ед./л' in res2.columns


def test_load_oam(monkeypatch):
    monkeypatch.setattr('blood_analysis.OAM_PARAM_KEYS', ['pH'])
    df = pd.DataFrame({
        ('№ п/п',''): [1],
        ('Этап регистрации',''): ['Скрининг'],
        ('После приема',''): ['T'],
        ('pH','значение'): [6.0],
    })
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: df)
    res = load_oam_sheet(io.BytesIO())
    assert 'pH' in res.columns


def test_vitals_and_stage(monkeypatch):
    df = pd.DataFrame({
        ('Период',''): [1],
        ('Препарат',''): ['T'],
        ('витальные',''): ['Скрининг'],
        ('АД сист.',''): [120],
    })
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: df)
    res = load_vitals_sheet(io.BytesIO())
    assert 'АД систолическое, мм рт. ст.' in res.columns

    mock_xls = types.SimpleNamespace(sheet_names=['этапы-процедуры'])
    monkeypatch.setattr(pd, 'ExcelFile', lambda *a, **k: mock_xls)
    stage_df = pd.DataFrame([
        ['Скрининг'],
        ['После 1'],
        ['Период'],
    ])
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: stage_df)
    order = load_stage_order(io.BytesIO())
    assert order[0] == 'Скрининг'

