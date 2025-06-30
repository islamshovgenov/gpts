import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore
from matplotlib.ticker import (
    LogLocator,
    ScalarFormatter,
    MultipleLocator,
    AutoMinorLocator,
)
import pandas as pd


def _style_axes(ax):
    """Apply unified grid and remove top/right spines."""
    ax.grid(True, linestyle=":", alpha=0.6)
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)



# Явно задаём (min, max, step) для каждого параметра ОАК
Y_SCALES = {
    "Гемоглобин, г/л":            (110,   170, 10),
    "Гематокрит, %":              (30,   60,  5),
    "Эритроциты, 10¹²/л":         (3.5,  5.5,  0.5),
    "Лейкоциты, 10⁹/л":           (3,    10,   1),
    "СОЭ, мм/ч":                  (0,    15,   5),
    "Тромбоциты, 10⁹/л":          (100,  400,  50),
    "Базофилы, %":                (0,    2.5,  0.5),
    "Палочкоядерные нейтрофилы, %": (0,   8,    2),
    "Сегментоядерные нейтрофилы, %": (40, 80,  10),
    "Лимфоциты, %":               (10,   50,   10),
    "Моноциты, %":                (0,    16,   2),
    "Эозинофилы, %":              (0,    6,    1),
    "АЛТ, Ед./л":                  (0,   50,   10),
    "АСТ, Ед./л":                  (0,   50,   10),
    "Щелочная фосфатаза, Ед./л":   (50,  250,   50),
    "Билирубин общий, мкмоль/л":   (0,   25,    5),
    "Креатинин, мкмоль/л":         (60,  120,   10),
    "Глюкоза, ммоль/л":            (3.5,    6.5,    0.5),
    "Общий белок, г/л":            (60,  90,    5),
    "Холестерин общий, ммоль/л":   (3,    6,    1),
    # Шкалы для ОАМ
    "pH":                          (4,  7,  0.5),
    "Относительная плотность, г/мл": (1, 1.04, 0.01),
    "Белок, г/л":                  (0,    0.4,  0.1),
    "Глюкоза, ммоль/л":            (0,    6,  1),
    "Лейкоциты, в п/зр.":          (0,    8,    2),
    "Эритроциты, в п/зр.":         (0,    5,    1)
}

def plot_ci(ratios, lowers, uppers, labels):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    err_low = [r - l for r, l in zip(ratios, lowers)]
    err_hi = [u - r for u, r in zip(uppers, ratios)]
    ax.errorbar(
        ratios,
        range(len(labels)),
        xerr=[err_low, err_hi],
        fmt="o",
        color="black",
        capsize=4,
    )
    ax.axvspan(80, 125, color="green", alpha=0.1)
    ax.axvline(80, linestyle="--", color="red")
    ax.axvline(125, linestyle="--", color="red")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(70, 130)
    ax.set_xlabel("Отношение T/R (%)")
    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_individual(df, subject, test_name="Test", ref_name="Reference"):
    fig, ax = plt.subplots(figsize=(8, 4))
    df_subj = df[df["Subject"] == subject]
    colors = {"Test": "C0", "Ref": "C1"}
    markers = {"Test": "o", "Ref": "s"}
    for period in sorted(df_subj["Period"].unique()):
        group = df_subj[df_subj["Period"] == period]
        treatment = group["Treatment"].iloc[0]
        label = (
            f"{test_name} (Period {period})"
            if treatment == "Test"
            else f"{ref_name} (Period {period})"
        )
        ax.plot(
            group["Time"],
            group["Concentration"],
            marker=markers.get(treatment, "o"),
            color=colors.get(treatment, "black"),
            label=label,
        )
    ax.set_title(f"Концентрация – Время для добровольца {subject}")
    ax.set_xlabel("Время (часы)")
    ax.set_ylabel("Концентрация")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_mean_curves(mean_df, test_name="Test", ref_name="Reference", logscale=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Test": "C0", "Ref": "C1"}
    for treatment in mean_df["Treatment"].unique():
        group = mean_df[mean_df["Treatment"] == treatment]
        label = test_name if treatment == "Test" else ref_name
        ax.plot(
            group["Time"],
            group["Concentration"],
            marker="o",
            linestyle="-",
            color=colors.get(treatment, "black"),
            label=label,
            markersize=6,
            linewidth=1.5,
        )

    # Подписи
    ax.set_title("Средние концентрации – Время" + (" (лог шкала)" if logscale else ""), fontsize=14)
    ax.set_xlabel("Время (t) после приема препарата, ч", fontsize=12)
    ax.set_ylabel("Среднее значение концентраций С(t), нг/мл", fontsize=12)

    # 1) ЛОГ-ОСЬ Y
    if logscale:
        ax.set_yscale("log")

        # 1.1) Основные тики на 10^n
        yticks_major = [100, 1000, 3000, 5000, 7000]
        ax.set_yticks(yticks_major)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

        # 1.2) Минорные тики (2×10^n, 3×10^n, …, 9×10^n)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))

    # 2) ОСЬ X с «жёсткими» метками
    x_ticks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=10)

    # 2.1) Перепромежуточные (минорные) тики по X — по желанию
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))

    # 3) ЛЕГЕНДА
    legend = ax.legend(loc="best", frameon=True, fontsize=10, facecolor="white")
    legend.get_frame().set_edgecolor("black")

    # 4) СЕТКА: основные и вторичные линии
    ax.grid(which="major", linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(which="minor", linestyle=':', linewidth=0.4, alpha=0.5)

    # 5) ПОДГОНКА ОТСТУПОВ
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _style_axes(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig

def plot_individual_log(df, subject, test_name="Test", ref_name="Reference", terminal_points=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_subj = df[df["Subject"] == subject].copy()

    colors = {"Test": "blue", "Ref": "red"}
    markers = {"Test": "^", "Ref": "o"}

    # Определим максимум по концентрации (для адекватной Y-оси)
    max_c = df_subj["Concentration"].max()
    y_max = max(10, round(max_c * 1.2, 1))

    for treatment in ["Test", "Ref"]:
        for period in sorted(df_subj["Period"].unique()):
            group = df_subj[(df_subj["Treatment"] == treatment) & (df_subj["Period"] == period)]
            if group.empty:
                continue

            group = group[group["Concentration"] > 0]
            label = f"{test_name} (Period {period})" if treatment == "Test" else f"{ref_name} (Period {period})"
            ax.plot(group["Time"], group["Concentration"],
                    marker=markers[treatment], linestyle='-', color=colors[treatment], label=label)

            # Регрессия по терминальным точкам
            terminal = group.sort_values("Time").tail(terminal_points)
            if len(terminal) >= 2:
                x = terminal["Time"]
                y = np.log(terminal["Concentration"])
                slope, intercept, *_ = linregress(x, y)
                x_reg = np.linspace(x.min(), x.max(), 100)
                y_reg = np.exp(intercept + slope * x_reg)
                reg_label = f"Регрессия термин-го периода ({'Test' if treatment == 'Test' else 'Reference'})"
                ax.plot(x_reg, y_reg, linestyle="--", color=colors[treatment], label=reg_label)

    ax.set_yscale("log")
    ax.set_ylim(0.1, y_max)

    # Настроим деления и подписи
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_title("В логарифмических координатах")
    ax.set_xlabel("Время (t) после приёма препарата, ч")
    ax.set_ylabel("Концентрация C(t), нг/мл")
    _style_axes(ax)
    ax.legend()

    # Примечание
    fig.text(0.5, -0.05, "Примечание: в логарифмических координатах точки с нулевой концентрацией не отображаются,\n"
                         "т.к. логарифм нуля не существует", ha='center', fontsize=8)

    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_mean_sd(df, treatment_label, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Оставим только нужное лечение
    df = df[df["Treatment"] == treatment_label]

    # Группировка по времени
    stats = df.groupby("Time")["Concentration"].agg(['mean', 'std']).reset_index()
    stats["err"] = 2 * stats["std"]

    # Стили
    color = "blue" if treatment_label == "Test" else "red"
    marker = "o" if treatment_label == "Test" else "s"
    label_mean = "- среднее значение (Mean)"
    label_sd = "- удвоенное значение стандартного отклонения (±SD)"

    # График с ошибками
    ax.errorbar(
        stats["Time"],
        stats["mean"],
        yerr=stats["err"],
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=4,
        label=label_mean,
    )

    ax.plot(stats["Time"], stats["mean"], linestyle=":", color=color)

    # Заголовок и оси
    ax.set_xlabel("Время (t) после приёма препарата, ч")
    ax.set_ylabel("Концентрация C(t), нг/мл")
    if not title:
        title = "Кривые средней концентрации ± 2×SD"
    ax.set_title(title)
    ax.legend(loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_all_individual_profiles(df, treatment_label, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    df = df[df["Treatment"] == treatment_label]
    for subject in df["Subject"].unique():
        subj_df = df[df["Subject"] == subject]
        ax.plot(subj_df["Time"], subj_df["Concentration"], marker='o', linestyle='-', linewidth=1, alpha=0.6)

    ax.set_xlabel("Время с момента приёма препарата, ч")
    ax.set_ylabel("Концентрация C(t), нг/мл")
    _style_axes(ax)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Сводная индивидуальная концентрация – {treatment_label}")

    fig.tight_layout()
    return fig

def plot_radar_auc_cmax(
    pivot: pd.DataFrame,
    test_label: str,
    ref_label: str,
    dose_test: float,
    dose_ref: float
) -> plt.Figure:
    """
    Рисует две «портретных» радар-диаграммы (AUC0–t и Cmax),
    c отдельной легендой в каждом подграфике и стилизованными спицами.
    """
    # 1) создаём портретную фигуру
    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(8.27, 11.69),
        subplot_kw=dict(polar=True)
    )
    fig.patch.set_facecolor('white')

    # Общий список субъектов и углов
    labels = pivot.index.astype(str).tolist()
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Внутренняя функция для каждого из двух графиков
    def _one(ax, col_t, col_r, title, unit):
        # 2) данные тест/реф + «замыкание» списка
        vals_t = pivot[col_t].tolist(); vals_t += vals_t[:1]
        vals_r = pivot[col_r].tolist(); vals_r += vals_r[:1]

        # 3) геометрические средние
        gm_t = np.exp(np.log(pivot[col_t]).mean())
        gm_r = np.exp(np.log(pivot[col_r]).mean())
        gm_line_t = [gm_t] * (N+1)
        gm_line_r = [gm_r] * (N+1)

        # 4) подписи для легенды
        lbl_t = f"{test_label} — {dose_test:.0f} мг"
        lbl_r = f"{ref_label} — {dose_ref:.0f} мг"

        # 5) рисуем линии и маркеры
        ax.plot(angles, vals_t,    marker='^', label=lbl_t)
        ax.plot(angles, vals_r,    marker='o', label=lbl_r)
        ax.plot(angles, gm_line_t, '--',      label=f"Геом. ср. ({test_label})")
        ax.plot(angles, gm_line_r, '--',      label=f"Геом. ср. ({ref_label})")

        # 6) стилизация спиц и сетки
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)

        # — угловые метки (номера субъектов)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            labels,
            fontsize= 8,
            color='green',
            fontweight='bold'
        )

        # — радиальные метки (авто-диапазон на 5 кружков)
        max_val = max(pivot[col_t].max(), pivot[col_r].max(), gm_t, gm_r)
        r_ticks = np.linspace(0, max_val, 5)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels(
            [f"{int(v):,}" for v in r_ticks],
            fontsize=8
        )

        # — сетка: спицы (theta) и окружности (r)
        ax.xaxis.grid(True, color='grey', linewidth=1)
        ax.yaxis.grid(True, color='lightgrey', linewidth=0.5)
        _style_axes(ax)

        # 7) заголовок в левом верхнем углу
        ax.set_title(
            f"{title} {test_label}, {unit}",
            loc='left',
            pad=10,
            fontsize=12
        )

        # 8) отдельная легенда внутри каждого графика
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.15, 1.15),
            fontsize=8,
            frameon=False
        )

    # Строим первый (AUC0–t) и второй (Cmax)
    _one(
        axes[0],
        col_t="AUC0-t_Test", col_r="AUC0-t_Ref",
        title="AUC(0–t),", unit="нг·ч/мл"
    )
    _one(
        axes[1],
        col_t="Cmax_Test", col_r="Cmax_Ref",
        title="Cmax", unit="нг/мл"
    )
    fig.tight_layout()
    plt.close(fig)
    return fig




def plot_studentized_residuals(pk_df, param="Cmax", substance="Препарат"):
    df = pk_df.copy()
    param_col = f"{param}"
    df["log_val"] = np.log(df[param_col])

    # Формируем таблицу вида: subject | ln_param_T | ln_param_R
    pivot = df.pivot_table(index="Subject", columns="Treatment", values="log_val")

    # Рассчитываем разность логарифмов
    pivot["log_diff"] = pivot["Test"] - pivot["Ref"]

    # Стандартизируем
    pivot["z"] = zscore(pivot["log_diff"], nan_policy='omit')

    # Начинаем рисовать
    fig, ax = plt.subplots(figsize=(10, 5))
    subjects = pivot.index.astype(str).str.zfill(2)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(2.98, color='red', linestyle='--', label="Lund’s p = 0.05 (±2.98)")
    ax.axhline(-2.98, color='red', linestyle='--')

    ax.scatter(subjects, pivot["z"], color='cornflowerblue', edgecolor='black', label="studentized residuals")
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Остатки (z-score)")
    ax.set_xlabel("Рандомизационный № добровольца")
    ax.set_title(f"studentized residuals\nдля разности ln({param}_T) – ln({param}_R)\n\n{substance}")
    _style_axes(ax)
    ax.legend(loc="upper right")

    return fig

def plot_studentized_group(pk_df, param="Cmax", group="Test", substance="Препарат"):
    df = pk_df[pk_df["Treatment"] == group].copy()
    df["log_val"] = np.log(df[param])
    df = df.sort_values("Subject")

    df["z"] = zscore(df["log_val"], nan_policy='omit')
    subjects = df["Subject"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(2.98, color='red', linestyle='--', label="Lund’s p = 0.05 (±2.98)")
    ax.axhline(-2.98, color='red', linestyle='--')

    ax.scatter(subjects, df["z"], color='cornflowerblue', edgecolor='black', label="studentized residuals")
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Остатки (z-score)")
    ax.set_xlabel("Рандомизационный № добровольца")
    ax.set_title(f"studentized residuals\nдля показателя ln({param})_{group[0]}\n\n{substance}")
    _style_axes(ax)
    ax.legend(loc="upper right")
    plt.close(fig)
    return fig

def plot_group_dynamics(
    means_df,
    param_name: str,
    units: str,
    group_col: str = "После приема"
):
    """
    Рисует линию изменения средних значений до и после для групп T и R
    с индивидуальной Y-шкалой из Y_SCALES.
    """
    x = [0, 1]
    labels = ["до приема", "после приема"]

    fig, ax = plt.subplots()
    for _, row in means_df.iterrows():
        grp = row[group_col]
        y = [row["baseline"], row["followup"]]
        ax.plot(x, y, marker='o', label=grp)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{param_name}, {units}")
    ax.set_title(f"Динамика {param_name}")
    ax.legend(title="Группа")

    # — Применяем индивидуальную шкалу, если она есть в Y_SCALES —
    scale = Y_SCALES.get(param_name)
    if scale:
        y_min, y_max, y_step = scale
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 1e-8, y_step))
    # —————————————————————————————————————————————

    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_group_iqr(
    iqr_df: pd.DataFrame,
    param_name: str,
    units: str,
    y_scale: tuple[float, float, float] = None,   # ← новый аргумент
    group_col: str = "После приема"
):
    """
    Рисует динамику медианы с IQR (усиками) для групп T и R
    с опциональной пользовательской Y-шкалой.
    """
    x = [0, 1]
    labels = ["до приема", "после приема"]

    fig, ax = plt.subplots()
    for grp in iqr_df[group_col].unique():
        sub = iqr_df[iqr_df[group_col] == grp]
        b = sub[sub["stage"] == "baseline"].iloc[0]
        f = sub[sub["stage"] == "followup"].iloc[0]

        y = [b["median"], f["median"]]
        yerr = [
            [b["median"] - b["q1"], f["median"] - f["q1"]],
            [b["q3"] - b["median"], f["q3"] - f["median"]]
        ]
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label=grp)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{param_name}, {units}")
    ax.set_title(f"Динамика {param_name} с IQR")
    ax.legend(title="Группа")

    # — Выбор шкалы: сначала аргумент y_scale, иначе из Y_SCALES —
    if y_scale is not None:
        y_min, y_max, y_step = y_scale
    else:
        # при отсутствии в словаре .get вернёт None, избегаем ошибки
        scale = Y_SCALES.get(param_name)
        if scale:
            y_min, y_max, y_step = scale
        else:
            # если нет ни в y_scale, ни в Y_SCALES — не трогаем ось
            plt.tight_layout()
            plt.close(fig)
            return fig

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1e-8, y_step))

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_individual_changes(df, param_col, title, units,
                            subject_col="№ п/п",
                            stage_col="Этап регистрации",
                            group_col="После приема"):
    """
    Scatter: before vs after, цвет/маркер по группе T/R,
    45°-линия, подписи осей, легенда, сетка.
    """
    from blood_analysis import extract_individual_lab
    data = extract_individual_lab(df, param_col,
                                  subject_col, stage_col, group_col)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Скаттер по группам (T — красные круги, R — синие треугольники)
    for grp, (marker, color) in {"T":("o","red"), "R":("^","blue")}.items():
        sub = data[data[group_col] == grp]
        ax.scatter(
            sub["before"], sub["after"],
            marker=marker, color=color,
            label=grp, s=60, edgecolor="black"
        )

    # Диагональная 45°-линия
    scale = Y_SCALES.get(param_col)
    if scale:
        xmin, xmax, _ = scale
        ymin, ymax = xmin, xmax
    else:
        xmin = ymin = 0
        xmax = ymax = max(data["before"].max(), data["after"].max()) * 1.05

    ax.plot([xmin, xmax], [ymin, ymax],
            linestyle="--", color="gray", label="45°")

    # Оформление
    ax.set_title(f"{title}, {units}", fontsize=12, pad=8)
    ax.set_xlabel("на скрининге", fontsize=10)
    ax.set_ylabel("После приёма ЛП", fontsize=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ————————————————
    # Добавляем равномерные метки по шагу из шкалы:
    if scale and scale[2] is not None:
        _, _, step = scale
        # MajorLocator с вашим шагом
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))
    # ————————————————
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # Легенда
    ax.legend(title="", loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    plt.close(fig)
    return fig


# диапазоны нормы
Y_SCALES.update({
    "АД систолическое":   (95, 130, 5),
    "АД систолическое, мм рт. ст.": (95, 130, 5),
    "АД диастолическое":  (60,  89,  None),
    "АД диастолическое, мм рт. ст.": (55,  90, 5),
    "ЧСС":                (60,  90,  None),
    "ЧСС, уд/мин": (55, 90, 5),
    "ЧДД":                (16,  20,  None),
    "ЧДД, в мин": (15, 20, 1),
    "Температура тела":   (36.3, 37.0, None),
    "Температура тела, °C" : (36.3, 37.0, 0.1),
})
# порядок этапов регистрации для витальных функций
STAGE_ORDER = [
    "при госпитализации",
    "через 1 ч после приема",
    "через 2 ч после приема",
    "через 6 ч после приема",
    "через 12 ч после приема",
    "через 24 ч после приема",
]

def plot_vitals_dynamics(
    iqr_df,
    title: str,
    unit: str,
    param_col: str,
    stage_order: list[str],
    group_col: str = "Препарат",
    stage_col: str = "Этап регистрации"
):
    fig, ax = plt.subplots(figsize=(10, 4))

    # те же Y_SCALES…
    xmin, xmax, step = Y_SCALES.get(param_col, (None, None, None))
    ymin, ymax = xmin, xmax

    for grp, color, marker in [("T","C0","o"), ("R","C3","s")]:
        sub = (
            iqr_df[iqr_df[group_col] == grp]
            .groupby(stage_col)[["q1","median","q3"]]
            .first()
            .reindex(stage_order)
        )
        ax.errorbar(
            x=range(len(stage_order)),
            y=sub["median"],
            yerr=[sub["median"]-sub["q1"], sub["q3"]-sub["median"]],
            fmt=marker+"-",
            color=color, label=grp, capsize=4
        )

    ax.set_title(f"{title}, {unit}", pad=8)
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"{title}, {unit}", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout(pad=2.0)
    if step:
        ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(title="Группа", loc="upper right", frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig
