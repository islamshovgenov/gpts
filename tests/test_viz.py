import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from viz import (
    plot_ci,
    plot_individual,
    plot_individual_log,
    plot_all_individual_profiles,
    plot_radar_auc_cmax,
    plot_studentized_group,
    plot_mean_curves,
    plot_mean_sd,
    plot_studentized_residuals,
    plot_group_dynamics,
    plot_group_iqr,
    plot_individual_changes,
    plot_vitals_dynamics,
)


def make_basic_df():
    return pd.DataFrame({
        "Subject": [1, 1, 1, 1],
        "Period": [1, 1, 2, 2],
        "Treatment": ["Test", "Test", "Ref", "Ref"],
        "Time": [0, 1, 0, 1],
        "Concentration": [0.0, 10.0, 0.0, 8.0],
    })


def test_plot_ci_basic():
    fig = plot_ci([100, 110], [90, 100], [110, 120], ["A", "B"])
    assert isinstance(fig, Figure)


def test_plot_individual_and_mean():
    df = make_basic_df()
    fig1 = plot_individual(df, 1, "T", "R")
    fig2 = plot_mean_curves(df, "T", "R", xlog=True)
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)


def test_plot_mean_sd():
    df = pd.DataFrame({
        "Time": [0, 1, 0, 1],
        "Treatment": ["Test", "Test", "Test", "Test"],
        "Concentration": [1.0, 2.0, 1.5, 2.5],
    })
    fig = plot_mean_sd(df, "Test")
    assert isinstance(fig, Figure)


def test_studentized_residuals_and_group():
    pk_df = pd.DataFrame({
        "Subject": [1, 1, 2, 2],
        "Treatment": ["Test", "Ref", "Test", "Ref"],
        "Cmax": [100, 80, 90, 90],
    })
    fig1 = plot_studentized_residuals(pk_df, param="Cmax")
    assert isinstance(fig1, Figure)


def test_group_plots():
    means_df = pd.DataFrame({
        "После приема": ["T", "R"],
        "baseline": [10, 12],
        "followup": [15, 14],
    })
    fig_dyn = plot_group_dynamics(means_df, "ALT", "U/L", group_col="После приема")
    assert isinstance(fig_dyn, Figure)

    iqr_df = pd.DataFrame({
        "После приема": ["T", "T", "R", "R"],
        "stage": ["baseline", "followup", "baseline", "followup"],
        "q1": [1, 2, 1.5, 2.5],
        "median": [1.5, 2.5, 2, 3],
        "q3": [2, 3, 2.5, 3.5],
    })
    fig_iqr = plot_group_iqr(iqr_df, "ALT", "U/L")
    assert isinstance(fig_iqr, Figure)


def test_individual_changes_and_vitals():
    df_lab = pd.DataFrame({
        "№ п/п": [1, 1, 2, 2],
        "Этап регистрации": ["Скрининг", "Период 2 - в конце", "Скрининг", "Период 2 - в конце"],
        "После приема": ["T", "T", "R", "R"],
        "ALT": [10, 20, 10, 15],
    })
    fig_chg = plot_individual_changes(df_lab, "ALT", "ALT", "U/L")
    assert isinstance(fig_chg, Figure)

    stage_order = ["при госпитализации", "через 1 ч после приема"]
    vitals_df = pd.DataFrame({
        "Препарат": ["T", "T", "R", "R"],
        "Этап регистрации": stage_order * 2,
        "q1": [10, 12, 11, 13],
        "median": [12, 13, 12, 14],
        "q3": [14, 15, 14, 15],
    })
    fig_vital = plot_vitals_dynamics(
        vitals_df, "АД", "мм рт.ст.", "АД систолическое, мм рт. ст.", stage_order
    )
    assert isinstance(fig_vital, Figure)

def test_additional_plots():
    df = make_basic_df()
    fig_log = plot_individual_log(df, 1, "T", "R", terminal_points=1)
    assert isinstance(fig_log, Figure)

    fig_all = plot_all_individual_profiles(df, "Test")
    assert isinstance(fig_all, Figure)

    pivot = pd.DataFrame({
        'AUC0-t_Test':[100],
        'AUC0-t_Ref':[90],
        'Cmax_Test':[10],
        'Cmax_Ref':[9]
    }, index=[1])
    fig_radar = plot_radar_auc_cmax(pivot, 'T', 'R', 100, 100)
    assert isinstance(fig_radar, Figure)

    pk_df = pd.DataFrame({
        'Subject':[1,2],
        'Treatment':['Test','Test'],
        'Cmax':[100,120]
    })
    fig_grp = plot_studentized_group(pk_df, param='Cmax', group='Test')
    assert isinstance(fig_grp, Figure)

