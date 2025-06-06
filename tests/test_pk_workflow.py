import pandas as pd
import src.pk_workflow as pw


def test_compute_pk_and_stats(monkeypatch):
    pk_table = pd.DataFrame({
        'Subject':[1,1],
        'Sequence':['TR','TR'],
        'Treatment':['Test','Ref'],
        'Period':[1,2],
        'Cmax':[10,9],
        'AUC0-t':[100,110],
        'AUC0-inf':[120,130],
        'Tmax':[1.0,1.0],
        'T1/2':[2.0,2.0],
        'Kel':[0.1,0.1],
        'CL':[1.0,1.0],
        'Vd':[20.0,21.0],
        'MRT':[12.0,12.0],
        'Tlag':[0.0,0.0],
    })

    # patch external helpers so we test integration logic only
    monkeypatch.setattr(pw, 'compute_pk', lambda df, dose_test=100, dose_ref=100: pk_table)
    monkeypatch.setattr(pw, 'log_diff_stats', lambda x: x)
    monkeypatch.setattr(pw, 'add_mse_se_to_pivot', lambda x: x)
    monkeypatch.setattr(pw, 'get_gmr_lsmeans', lambda df, param: {'AUC0-t':(1.1,0.9,1.2),
                                                                  'AUC0-inf':(1.2,1.0,1.4),
                                                                  'Cmax':(0.8,0.7,0.9)}[param])
    monkeypatch.setattr(pw, 'anova_log', lambda df, col: f'anova_{col}')
    monkeypatch.setattr(pw, 'calc_swr', lambda x: (0.1, 10))
    monkeypatch.setattr(pw, 'get_cv_intra_anova', lambda df, param: {'Cmax':12,'AUC0t':15,'AUC0inf':18}[param])
    monkeypatch.setattr(pw, 'identify_be_outlier_and_recommend', lambda **k: ('none', ['ok']))

    result_pk, result_pivot, stats = pw.compute_pk_and_stats(pd.DataFrame({'dummy':[1]}))

    expected_pivot = pk_table.pivot(index='Subject', columns='Treatment',
                                    values=['Cmax','AUC0-t','AUC0-inf','Tmax','T1/2','Kel','CL','Vd','MRT','Tlag'])
    expected_pivot.columns = [f'{p}_{t}' for p, t in expected_pivot.columns]

    assert result_pk.equals(pk_table)
    assert result_pivot.equals(expected_pivot)
    assert stats['gmr'] == [1.1, 1.2, 0.8]
    assert stats['ci_low'] == [0.9, 1.0, 0.7]
    assert stats['ci_up'] == [1.2, 1.4, 0.9]
    assert stats['anova']['cmax'] == 'anova_Cmax'
    assert stats['swr'] == [0.1, 0.1, 0.1]
    assert stats['cv'] == [15, 18, 12]
    assert stats['outlier'] == 'none'
    assert stats['recs'] == ['ok']
