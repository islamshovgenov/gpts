import matplotlib.pyplot as plt
import sys
import types
import pandas as pd

class DummyFile:
    def read(self):
        return b'data'

class DummySidebar:
    def number_input(self, label, **kw):
        return kw.get('value')
    def text_input(self, label, value=""):
        return value
    def file_uploader(self, *a, **kw):
        if kw.get('accept_multiple_files'):
            return [DummyFile()]
        return DummyFile()
    def selectbox(self, label, options, **kw):
        return options[0]
    def checkbox(self, label, **kw):
        return False
    def multiselect(self, label, options, **kw):
        return []
    def subheader(self, *a, **k):
        pass
    def header(self, *a, **k):
        pass

class DummyStreamlit:
    def __init__(self):
        self.sidebar = DummySidebar()
    def __getattr__(self, name):
        return lambda *a, **k: None


def test_app_run(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', dummy)

    # patch pandas style to avoid jinja2 dependency
    monkeypatch.setattr(pd.DataFrame, 'style', property(lambda self: types.SimpleNamespace(format=lambda *a, **k: self)))

    def fake_load_data(rand, time, xls):
        df = pd.DataFrame({'Subject':[1], 'Treatment':['Test'], 'Time':[0], 'Concentration':[0.0]})
        return df, {1:'TR'}, {'01':0}
    monkeypatch.setitem(sys.modules, 'src.data_loader', types.SimpleNamespace(load_data=fake_load_data, DataLoaderError=Exception))


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
    monkeypatch.setitem(sys.modules, 'src.pk_workflow', types.SimpleNamespace(compute_pk_and_stats=fake_compute))
    monkeypatch.setitem(sys.modules, "src.plot_utils", types.SimpleNamespace(confidence_interval_plot=lambda *a,**k: None, individual_profile=lambda *a,**k: plt.figure(), mean_curves=lambda *a,**k: plt.figure(), mean_sd_plot=lambda *a,**k: plt.figure(), all_profiles=lambda *a,**k: plt.figure(), radar_plot=lambda *a,**k: plt.figure(), studentized_residuals_plot=lambda *a,**k: plt.figure(), studentized_group_plot=lambda *a,**k: plt.figure()))
    monkeypatch.setitem(sys.modules, 'stat_tools', types.SimpleNamespace(ci_calc=lambda x:0, calc_swr=lambda x:(1.0,1.0), get_cv_intra_anova=lambda *a,**k:0.1, compute_vitals_iqr=lambda *a, **k: pd.DataFrame(), compute_group_iqr=lambda *a, **k: pd.DataFrame(), make_stat_report_table=lambda *a, **k: pd.DataFrame()))

    import importlib
    import app
    importlib.reload(app)


