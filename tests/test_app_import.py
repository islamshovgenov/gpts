import sys
import types

class DummySidebar:
    def number_input(self, label, **kw):
        return kw.get('value')
    def text_input(self, label, value=""):
        return value
    def file_uploader(self, *a, **k):
        return None
    def selectbox(self, *a, **k):
        return None
    def checkbox(self, *a, **k):
        return False
    def multiselect(self, *a, **k):
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


def test_app_import(monkeypatch):
    dummy = DummyStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', dummy)
    import importlib
    app = importlib.import_module('app')
    assert hasattr(app, 'dose_test')
    assert app.dose_test == 100.0
