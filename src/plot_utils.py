from viz import (
    plot_ci,
    plot_individual,
    plot_mean_curves,
    plot_individual_log,
    plot_mean_sd,
    plot_all_individual_profiles,
    plot_radar_auc_cmax,
    plot_studentized_residuals,
    plot_studentized_group,
)


def confidence_interval_plot(gmr_cmax, gmr_auc, ci_l_cmax, ci_l_auc, ci_u_cmax, ci_u_auc):
    return plot_ci(
        [gmr_cmax * 100, gmr_auc * 100],
        [ci_l_cmax * 100, ci_l_auc * 100],
        [ci_u_cmax * 100, ci_u_auc * 100],
        ["Cmax", "AUC0-t"],
    )


def individual_profile(df, subj, test_name, ref_name, log=False):
    if log:
        return plot_individual_log(df, subj, test_name, ref_name)
    return plot_individual(df, subj, test_name, ref_name)




def mean_curves(
    df,
    test_name,
    ref_name,
    log=False,
    times=None,
    xlog=False,
    xlog_threshold=1.0,
):
def mean_curves(df, test_name, ref_name, log=False, times=None, xlog=False):
    """Wrapper around :func:`plot_mean_curves` with keyword-only arguments."""
    return plot_mean_curves(
        df,
        test_name,
        ref_name,
        logscale=log,
        xticks=times,
        xlog=xlog,
        xlog_threshold=xlog_threshold,
    )





def mean_sd_plot(df, label, title):
    return plot_mean_sd(df, treatment_label=label, title=title)


def all_profiles(df, label, title):
    return plot_all_individual_profiles(df, label, title)


def radar_plot(pivot, test_label, ref_label, dose_test, dose_ref):
    return plot_radar_auc_cmax(pivot, test_label=test_label, ref_label=ref_label, dose_test=dose_test, dose_ref=dose_ref)


def studentized_residuals_plot(pk_table, param, substance):
    return plot_studentized_residuals(pk_table, param=param, substance=substance)


def studentized_group_plot(pk_table, param, group, substance):
    return plot_studentized_group(pk_table, param=param, group=group, substance=substance)
