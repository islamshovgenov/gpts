import pandas as pd
import types
from pathlib import Path
from docx_tools import export_auc_residual_tables, export_log_transformed_pk_tables, export_sas_anova_report


def test_export_auc_residual_tables(tmp_path):
    df = pd.DataFrame({
        'Subject':[1,2],
        'Treatment':['Test','Ref'],
        'AUC0-t':[100,90],
        'AUC0-inf':[110,100],
    })
    out = tmp_path / 'auc.docx'
    path = export_auc_residual_tables(df,'T','R','Sub',100,100,save_path=str(out))
    assert Path(path).is_file()


def test_export_log_transformed_pk(tmp_path):
    df = pd.DataFrame({
        'Subject':[1,2],
        'Treatment':['Test','Ref'],
        'AUC0-t':[100,110],
        'AUC0-inf':[120,130],
        'Cmax':[10,9],
    })
    out = tmp_path / 'ln.docx'
    path = export_log_transformed_pk_tables(df,'T','R','Sub',100,100,save_path=str(out))
    assert Path(path).is_file()


def test_export_sas_anova_report(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Subject':[1,1,2,2],
        'Sequence':['TR','TR','RT','RT'],
        'Treatment':['Test','Ref','Test','Ref'],
        'Period':[1,2,1,2],
        'AUC0-t':[10,12,11,13],
        'AUC0-inf':[15,16,14,17],
        'Cmax':[1,2,3,4]
    })
    dummy_res = {
        'ss':{'Sequence':1,'subject(Sequence)':2,'formulation':3,'Period':4,'Error':5},
        'df':{'Sequence':1,'subject(Sequence)':1,'formulation':1,'Period':1,'Error':1},
        'ms':{'Sequence':1,'subject(Sequence)':2,'formulation':3,'Period':4,'Error':5},
        'F':{'Sequence':1,'formulation':2,'subject(Sequence)':3,'Period':4},
        'p':{'Sequence':0.1,'formulation':0.2,'subject(Sequence)':0.3,'Period':0.4},
        'nested_test':{'Sequence':{'F':1.1,'p':0.05}}
    }
    monkeypatch.setattr('docx_tools.nested_anova', lambda *a, **k: dummy_res)
    class DummyModel:
        params={'C(Treatment)[T.Test]':1.0}
        bse={'C(Treatment)[T.Test]':0.1}
        tvalues={'C(Treatment)[T.Test]':10.0}
        pvalues={'C(Treatment)[T.Test]':0.01}
    monkeypatch.setattr('docx_tools.ols', lambda *a, **k: types.SimpleNamespace(fit=lambda: DummyModel()))
    out = tmp_path / 'sas.docx'
    path = export_sas_anova_report(df,'Sub',100,save_path=str(out))
    assert Path(path).is_file()

