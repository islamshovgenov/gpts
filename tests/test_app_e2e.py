import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

class DummyFile:
    def read(self):
        return b'data'


def make_app(monkeypatch, tmp_path):
    monkeypatch.setattr(st.sidebar, "file_uploader", lambda *a, **k: DummyFile() if not k.get('accept_multiple_files') else [DummyFile()])
    monkeypatch.setattr(st.sidebar, "multiselect", lambda *a, **k: [])
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: DummyFile())
    monkeypatch.setattr(pd.DataFrame, "style", property(lambda self: types.SimpleNamespace(format=lambda *a, **k: self)))

    def fake_load_data(rand, time, xls):
        df = pd.DataFrame({
            'Subject':[1],
            'Treatment':['Test'],
            'Time':[0],
            'Concentration':[0.0]
        })
        return df, {1:'TR'}, {'01':0}
    sys.modules['src.data_loader'] = types.SimpleNamespace(load_data=fake_load_data, DataLoaderError=Exception)

    def fake_compute(df, dose_test=100, dose_ref=100):
        pk = pd.DataFrame({'Subject':[1],'Treatment':['Test'],'Cmax':[10],'AUC0-t':[100],'AUC0-inf':[110]})
        pivot = pd.DataFrame({'Cmax_Test':[10],'Cmax_Ref':[9],'AUC0-t_Test':[100],'AUC0-t_Ref':[90],'AUC0-inf_Test':[110],'AUC0-inf_Ref':[100]})
        stats = {
            'gmr':[1,1,1],
            'ci_low':[0.8,0.8,0.8],
            'ci_up':[1.2,1.2,1.2],
            'cv':[0.1,0.1,0.1],
            'anova':{
                'cmax':(None,pd.DataFrame(),None,None),
                'auc':(None,pd.DataFrame(),None,None),
                'aucinf':(None,pd.DataFrame(),None,None)
            },
            'swr':[1,1,1],
            'outlier':None,
            'recs':pd.DataFrame()
        }
        return pk, pivot, stats
    sys.modules['src.pk_workflow'] = types.SimpleNamespace(compute_pk_and_stats=fake_compute)

    sys.modules['src.plot_utils'] = types.SimpleNamespace(
        confidence_interval_plot=lambda *a,**k: None,
        individual_profile=lambda *a,**k: None,
        mean_curves=lambda *a,**k: None,
        mean_sd_plot=lambda *a,**k: None,
        all_profiles=lambda *a,**k: None,
        radar_plot=lambda *a,**k: None,
        studentized_residuals_plot=lambda *a,**k: None,
        studentized_group_plot=lambda *a,**k: None,
    )

    def fake_export(path_name):
        p = tmp_path/ path_name
        p.write_text('x')
        return p
    sys.modules['src.export_utils'] = types.SimpleNamespace(
        export_be_tables=lambda *a,**k: [fake_export('be.docx')],
        export_be_result=lambda *a,**k: fake_export('res.docx'),
        export_power_table=lambda *a,**k: fake_export('pow.docx'),
    )

    import blood_analysis
    monkeypatch.setattr(blood_analysis, 'load_oak_sheet', lambda *a,**k: None)
    monkeypatch.setattr(blood_analysis, 'load_vitals_sheet', lambda *a,**k: None)
    monkeypatch.setattr(blood_analysis, 'plot_oak_pairwise', lambda *a,**k: None)
    monkeypatch.setattr(blood_analysis, 'plot_all_oak_parameters', lambda *a,**k: None)
    monkeypatch.setattr(blood_analysis, 'load_stage_order', lambda *a,**k: [])
    monkeypatch.setattr(blood_analysis, 'vitals_params', {})

    sys.modules['stat_tools'] = types.SimpleNamespace(
        ci_calc=lambda x:(1,1,1),
        calc_swr=lambda x:(1,1),
        get_cv_intra_anova=lambda *a,**k: 0.1,
        compute_vitals_iqr=lambda *a,**k: pd.DataFrame(),
        compute_group_iqr=lambda *a,**k: pd.DataFrame(),
        make_stat_report_table=lambda *a,**k: pd.DataFrame()
    )

    return AppTest.from_file('app.py', default_timeout=5)


def test_app_e2e(monkeypatch, tmp_path):
    at = make_app(monkeypatch, tmp_path)
    at.run()
    assert at.sidebar.number_input[0].value == 100.0
    at.sidebar.number_input[0].set_value(150.0)
    at.run()
    assert at.sidebar.number_input[0].value == 150.0
    assert len(at.sidebar.number_input) >= 2

    for tbl in list(at.sidebar.selectbox[0].options):
        at.sidebar.selectbox[0].set_value(tbl)
        at.run()

    at.sidebar.selectbox[1].set_value('studentized residuals: ln(Cmax_T)')
    at.run()
