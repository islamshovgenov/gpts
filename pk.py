
import numpy as np
import pandas as pd
from scipy.stats import linregress

def calc_auc(time, conc):
    idx = np.argsort(time)
    t = np.array(time)[idx]
    c = np.array(conc)[idx]
    return np.sum(np.diff(t) * (c[:-1] + c[1:]) / 2)

def calc_kel(concs, times, min_points=5, lloq=1.0):
    concs = np.array(concs)
    times = np.array(times)

    # Фильтруем: только ≥ LLOQ и без NaN
    mask = (concs >= lloq) & (~np.isnan(concs)) & (~np.isnan(times))
    concs = concs[mask]
    times = times[mask]

    if len(concs) < min_points:
        return np.nan, np.nan, 0

    # Берем только точки после Tmax
    tmax_idx = np.argmax(concs)
    concs_post = concs[tmax_idx:]
    times_post = times[tmax_idx:]

    if len(concs_post) < min_points:
        return np.nan, np.nan, len(concs_post)

    # Логарифмируем
    y = np.log(concs_post)
    x = times_post

    if np.any(~np.isfinite(y)):
        return np.nan, np.nan, 0

    slope, intercept, r_value, _, _ = linregress(x, y)
    kel = -slope if slope < 0 else np.nan
    t_half = np.log(2) / kel if kel and kel > 0 else np.nan
    return kel, t_half, len(x)

def compute_pk(df, dose_test=100, dose_ref=100, min_points=4, lloq=1.0):
    pk_detailed = []

    for (subj, period, treat), group in df.groupby(["Subject", "Period", "Treatment"]):
        dose = dose_test if treat == "Test" else dose_ref
        group = group.sort_values("Time")
        times = group["Time"].values
        concs = group["Concentration"].values

        # --- Шаг 1: если нет ни одной положительной точки (Concentration > 0) ---
        if not np.any(concs > lloq):
            # Для этого субъекта все концентрации ≤0 — сразу добавляем строку с NaN
            pk_detailed.append([
                subj,
                group["Sequence"].iloc[0],
                treat,
                period,
                *([np.nan] * 11)  # Cmax, Tmax, AUC0-t, AUC0-inf, Kel, T1/2, N_el, CL, Vd, MRT, Tlag
            ])
            continue   # <- этот continue должен быть ВНУТРИ if!

        # Если здесь — значит есть хоть одна >0, и дальше идёт полный расчёт:
        cmax = np.max(concs)
        tmax = times[np.argmax(concs)]
        auc_t = calc_auc(times, concs)

        # Kel и T1/2
        kel, t_half, n_el = calc_kel(concs, times, min_points=min_points, lloq=lloq)

        # AUC0-inf
        c_last = concs[concs > 0][-1]
        if not np.isnan(kel) and kel > 0:
            auc_inf = auc_t + c_last / kel
        else:
            auc_inf = np.nan

        # MRT, CL, Vd
        if (not np.isnan(kel) and kel > 0) and (not np.isnan(auc_inf) and auc_inf > 0):
            dose_ng = dose * 1_000_000  # мг → нг
            cl = dose_ng / auc_inf / 1000  # L/h
            if not np.isfinite(cl):
                cl = np.nan
            vd = cl / kel if not np.isnan(cl) else np.nan
            mrt = auc_inf / cmax if cmax > 0 else np.nan
        else:
            cl = vd = mrt = np.nan

        # Tlag
        try:
            tlag = group.loc[group["Concentration"] > 0, "Time"].min()
        except (KeyError, TypeError):
            tlag = np.nan

        seq = group["Sequence"].iloc[0]

        pk_detailed.append([
            subj, seq, treat, period,
            cmax, tmax, auc_t, auc_inf,
            kel, t_half, n_el, cl, vd, mrt, tlag
        ])

    return pd.DataFrame(pk_detailed, columns=[
        "Subject", "Sequence", "Treatment", "Period",
        "Cmax", "Tmax", "AUC0-t", "AUC0-inf",
        "Kel", "T1/2", "N_el", "CL", "Vd", "MRT", "Tlag"
    ])





