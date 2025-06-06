import pandas as pd
import numpy as np
from scipy.stats import gmean

from pk import compute_pk
from stat_tools import (
    log_diff_stats,
    add_mse_se_to_pivot,
    anova_log,
    calc_swr,
    get_cv_intra_anova,
    get_gmr_lsmeans,
    identify_be_outlier_and_recommend,
)


class PKComputationError(Exception):
    pass


def compute_pk_and_stats(df, dose_test=100, dose_ref=100):
    """Compute PK table and statistical summaries."""
    pk_table = compute_pk(df, dose_test=dose_test, dose_ref=dose_ref)

    pivot = pk_table.pivot(
        index="Subject",
        columns="Treatment",
        values=[
            "Cmax",
            "AUC0-t",
            "AUC0-inf",
            "Tmax",
            "T1/2",
            "Kel",
            "CL",
            "Vd",
            "MRT",
            "Tlag",
        ],
    )
    pivot.columns = [f"{param}_{trt}" for param, trt in pivot.columns]
    required = [
        "Cmax_Test",
        "Cmax_Ref",
        "AUC0-t_Test",
        "AUC0-t_Ref",
        "AUC0-inf_Test",
        "AUC0-inf_Ref",
    ]
    pivot = pivot.dropna(subset=required)
    pivot = log_diff_stats(pivot)
    pivot = add_mse_se_to_pivot(pivot)

    gmr_auc, ci_l_auc, ci_u_auc = get_gmr_lsmeans(pk_table, "AUC0-t")
    gmr_aucinf, ci_l_aucinf, ci_u_aucinf = get_gmr_lsmeans(pk_table, "AUC0-inf")
    gmr_cmax, ci_l_cmax, ci_u_cmax = get_gmr_lsmeans(pk_table, "Cmax")

    anova_cmax = anova_log(pk_table, "Cmax")
    anova_auc = anova_log(pk_table.rename(columns={"AUC0-t": "AUC0t"}), "AUC0t")
    anova_aucinf = anova_log(
        pk_table.rename(columns={"AUC0-inf": "AUC0inf"}), "AUC0inf"
    )

    sWR_cmax, cv_cmax = calc_swr(np.log(pivot["Cmax_Ref"]))
    sWR_auc, cv_auc = calc_swr(np.log(pivot["AUC0-t_Ref"]))
    sWR_aucinf, cv_aucinf = calc_swr(np.log(pivot["AUC0-inf_Ref"]))

    cv_cmax = get_cv_intra_anova(pk_table, "Cmax")
    cv_auc = get_cv_intra_anova(pk_table.rename(columns={"AUC0-t": "AUC0t"}), "AUC0t")
    cv_aucinf = get_cv_intra_anova(
        pk_table.rename(columns={"AUC0-inf": "AUC0inf"}), "AUC0inf"
    )

    outlier, recs = identify_be_outlier_and_recommend(
        df_conc=df,
        pk_df=pk_table,
        param="Cmax",
        ci_lower=ci_l_cmax,
        ci_upper=ci_u_cmax,
        be_limits=(0.8, 1.25),
    )

    stats = {
        "gmr": [gmr_auc, gmr_aucinf, gmr_cmax],
        "ci_low": [ci_l_auc, ci_l_aucinf, ci_l_cmax],
        "ci_up": [ci_u_auc, ci_u_aucinf, ci_u_cmax],
        "cv": [cv_auc, cv_aucinf, cv_cmax],
        "anova": {
            "cmax": anova_cmax,
            "auc": anova_auc,
            "aucinf": anova_aucinf,
        },
        "swr": [sWR_cmax, sWR_auc, sWR_aucinf],
        "outlier": outlier,
        "recs": recs,
    }

    return pk_table, pivot, stats
