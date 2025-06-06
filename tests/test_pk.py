import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from pk import calc_auc, calc_kel, compute_pk


def make_simple_df():
    times = [0, 1, 2, 3, 4, 5]
    concs = [0, 10, 8, 6, 4, 2]
    data = []
    for t, c in zip(times, concs):
        data.append({
            'Subject': 1,
            'Sequence': 'TR',
            'Period': 1,
            'Treatment': 'Test',
            'Time': t,
            'Concentration': c
        })
    return pd.DataFrame(data)


# ---- calc_auc -------------------------------------------------------------

def test_calc_auc_simple():
    auc = calc_auc([0, 1, 2], [0, 2, 0])
    assert auc == 2

def test_calc_auc_unsorted():
    auc = calc_auc([2, 0, 1], [0, 0, 2])
    assert auc == 2

# ---- calc_kel -------------------------------------------------------------

def test_calc_kel():
    concs = [0, 10, 8, 6, 4, 2]
    times = [0, 1, 2, 3, 4, 5]
    kel, t_half, n = calc_kel(concs, times, min_points=4)
    assert np.isclose(kel, 0.3912023, atol=1e-6)
    assert np.isclose(t_half, 1.7718382, atol=1e-6)
    assert n == 5

def test_calc_kel_insufficient_total():
    concs = [0, 0.5, 0.4]
    times = [0, 1, 2]
    kel, t_half, n = calc_kel(concs, times, min_points=4)
    assert np.isnan(kel) and np.isnan(t_half)
    assert n == 0

def test_calc_kel_insufficient_post_tmax():
    concs = [1, 2, 3, 2, 1]
    times = [0, 1, 2, 3, 4]
    kel, t_half, n = calc_kel(concs, times, min_points=4, lloq=0.5)
    assert np.isnan(kel) and np.isnan(t_half)
    assert n == 3  # points after Tmax

# ---- compute_pk -----------------------------------------------------------

def test_compute_pk_single():
    df = make_simple_df()
    pk_table = compute_pk(df)
    row = pk_table.iloc[0]
    assert row['Cmax'] == 10
    assert row['Tmax'] == 1
    assert np.isclose(row['AUC0-t'], 29.0)
    assert np.isclose(row['AUC0-inf'], 34.112444, atol=1e-6)

def test_compute_pk_no_positive():
    df = make_simple_df()
    df['Concentration'] = 0.0
    pk_table = compute_pk(df)
    row = pk_table.iloc[0]
    assert row[['Cmax','Tmax','AUC0-t','AUC0-inf','Kel','T1/2','N_el','CL','Vd','MRT','Tlag']].isna().all()

