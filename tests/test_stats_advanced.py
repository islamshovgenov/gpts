import pandas as pd
import numpy as np
from stat_tools import get_gmr_ci_from_model, nested_anova


def make_df():
    return pd.DataFrame({
        "Subject": [1, 1, 2, 2, 3, 3, 4, 4],
        "Sequence": ["TR", "TR", "TR", "TR", "RT", "RT", "RT", "RT"],
        "Period": [1, 2, 1, 2, 1, 2, 1, 2],
        "Treatment": ["Test", "Ref", "Test", "Ref", "Ref", "Test", "Ref", "Test"],
        "Cmax": [10.0, 8.0, 11.0, 9.0, 9.0, 11.0, 8.5, 10.5],
    })


def test_get_gmr_ci_from_model():
    df = make_df()
    gmr, lo, hi = get_gmr_ci_from_model(df, "Cmax")
    assert lo < gmr < hi


def test_nested_anova():
    df = make_df()
    res = nested_anova(df, "Cmax")
    assert "Sequence" in res["F"]
    assert res["F"]["Sequence"] >= 0
