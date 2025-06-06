import pandas as pd
import src.export_utils as eu


def test_export_be_tables(monkeypatch):
    calls = {}

    def rec(name, ret):
        def _f(*a, **k):
            calls[name] = True
            return ret
        return _f

    monkeypatch.setattr(eu, 'export_individual_pk_tables', rec('ind', 'ind.docx'))
    monkeypatch.setattr(eu, 'export_auc_residual_tables', rec('auc', 'auc.docx'))
    monkeypatch.setattr(eu, 'export_log_transformed_pk_tables', rec('ln', 'ln.docx'))
    monkeypatch.setattr(eu, 'export_sas_anova_report', rec('anova', 'anova.docx'))
    monkeypatch.setattr(eu, 'export_log_ci_tables', rec('ci', 'ci.docx'))

    paths = eu.export_be_tables(pd.DataFrame(), pd.DataFrame(), 'T', 'R', 'Sub', 100, 100)
    assert paths == {
        'individual_pk': 'ind.docx',
        'auc_residual': 'auc.docx',
        'ln_pk': 'ln.docx',
        'anova_report': 'anova.docx',
        'log_ci': 'ci.docx'
    }
    assert set(calls) == {'ind','auc','ln','anova','ci'}


def test_export_helpers(monkeypatch):
    monkeypatch.setattr(eu, 'export_be_result_table', lambda *a, **k: 'be.docx')
    monkeypatch.setattr(eu, 'export_power_analysis_table', lambda *a, **k: 'power.docx')
    assert eu.export_be_result([1],[0.9],[1.1],[10]) == 'be.docx'
    assert eu.export_power_table([1],[10], n=12) == 'power.docx'
