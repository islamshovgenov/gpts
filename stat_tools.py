
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import f as f_dist
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf

def log_diff_stats(pivot_df):
    # 1. Проверяем наличие ключевых колонок
    required = [
        "Cmax_Test", "Cmax_Ref",
        "AUC0-t_Test", "AUC0-t_Ref",
        "AUC0-inf_Test", "AUC0-inf_Ref"
    ]
    missing = [c for c in required if c not in pivot_df.columns]
    if missing:
        raise KeyError(f"Для log_diff_stats не хватает колонок: {missing}")

    # 2. Инициализируем все лог-колонки NaN
    for col in [
        "ln_Cmax_Test", "ln_Cmax_Ref",
        "ln_AUC0-t_Test", "ln_AUC0-t_Ref",
        "ln_AUC0-inf_Test", "ln_AUC0-inf_Ref",
        "log_Cmax_diff", "log_AUC0-t_diff", "log_AUC0-inf_diff",
        "log_Cmax", "log_AUC", "log_AUCinf"
    ]:
        pivot_df[col] = np.nan

    # 3. Рассчитываем ln и diff только там, где оба значения >0
    # Cmax
    mask = (pivot_df["Cmax_Test"] > 0) & (pivot_df["Cmax_Ref"] > 0)
    pivot_df.loc[mask, "ln_Cmax_Test"] = np.log(pivot_df.loc[mask, "Cmax_Test"])
    pivot_df.loc[mask, "ln_Cmax_Ref"]  = np.log(pivot_df.loc[mask, "Cmax_Ref"])
    pivot_df.loc[mask, "log_Cmax_diff"] = (
        pivot_df.loc[mask, "ln_Cmax_Test"] - pivot_df.loc[mask, "ln_Cmax_Ref"]
    )

    # AUC0–t
    mask = (pivot_df["AUC0-t_Test"] > 0) & (pivot_df["AUC0-t_Ref"] > 0)
    pivot_df.loc[mask, "ln_AUC0-t_Test"] = np.log(pivot_df.loc[mask, "AUC0-t_Test"])
    pivot_df.loc[mask, "ln_AUC0-t_Ref"]  = np.log(pivot_df.loc[mask, "AUC0-t_Ref"])
    pivot_df.loc[mask, "log_AUC0-t_diff"] = (
        pivot_df.loc[mask, "ln_AUC0-t_Test"] - pivot_df.loc[mask, "ln_AUC0-t_Ref"]
    )

    # AUC0–∞
    mask = (pivot_df["AUC0-inf_Test"] > 0) & (pivot_df["AUC0-inf_Ref"] > 0)
    pivot_df.loc[mask, "ln_AUC0-inf_Test"] = np.log(pivot_df.loc[mask, "AUC0-inf_Test"])
    pivot_df.loc[mask, "ln_AUC0-inf_Ref"]  = np.log(pivot_df.loc[mask, "AUC0-inf_Ref"])
    pivot_df.loc[mask, "log_AUC0-inf_diff"] = (
        pivot_df.loc[mask, "ln_AUC0-inf_Test"] - pivot_df.loc[mask, "ln_AUC0-inf_Ref"]
    )

    # 4. Алиасы для старого кода
    pivot_df["log_Cmax"]   = pivot_df["log_Cmax_diff"]
    pivot_df["log_AUC"]    = pivot_df["log_AUC0-t_diff"]
    pivot_df["log_AUCinf"] = pivot_df["log_AUC0-inf_diff"]

    return pivot_df




def ci_calc(log_diff, alpha=0.1):
    mean_log = log_diff.mean()
    se_log = log_diff.std(ddof=1) / np.sqrt(len(log_diff))
    t_val = t.ppf(1 - alpha / 2, df=len(log_diff) - 1)
    ci_lower = np.exp(mean_log - t_val * se_log)
    ci_upper = np.exp(mean_log + t_val * se_log)
    ratio = np.exp(mean_log)
    return ratio, ci_lower, ci_upper

def anova_log(df, colname):
    df = df.copy()
    df["log"] = np.log(df[colname])
    df["Subject"] = df["Subject"].astype(str)
    df["Period"] = df["Period"].astype(str)
    df["Sequence"] = df["Sequence"].astype(str)
    df["Treatment"] = df["Treatment"].astype(str)
    df["Subject_in_Sequence"] = df["Sequence"] + "_" + df["Subject"]

    model = ols("log ~ C(Sequence) + C(Period) + C(Treatment) + C(Subject_in_Sequence)", data=df).fit()
    result = sm.stats.anova_lm(model, typ=3)
    result["mean_sq"] = result["sum_sq"] / result["df"]
    mse = result.loc["Residual", "mean_sq"] if "Residual" in result.index else model.mse_resid
    df_s = result.loc["Residual", "df"] if "Residual" in result.index else model.df_resid

    return model, result, mse, df_s


def calc_swr(log_vals):
    sWR = np.std(log_vals, ddof=1)
    cv = np.sqrt(np.exp(sWR ** 2) - 1) * 100
    return sWR, cv

def make_stat_report_table(gmr_vals, ci_lows, ci_ups, cvs):
    params = ["AUC₀–t", "AUC₀–∞", "Cmax"]
    rows = []

    for param, gmr, ci_low, ci_up, cv in zip(params, gmr_vals, ci_lows, ci_ups, cvs):
        gmr_fmt = f"{gmr*100:.2f}%"
        ci_fmt = f"{ci_low*100:.2f}% – {ci_up*100:.2f}%"
        cv_fmt = f"{cv:.2f}%"
        rows.append({
            "Параметр": param,
            "CV_intra, %": cv_fmt,
            "GMR (μT/μR)": gmr_fmt,
            "90% ДИ": ci_fmt,
            "Диапазон БЭ": "80,00% – 125,00%"
        })

    return pd.DataFrame(rows)

def add_mse_se_to_pivot(pivot_df):
    from scipy.stats import sem
    params = ["Cmax", "AUC0-t", "AUC0-inf"]

    for param in params:
        diff_col = f"log_{param}_diff"
        if diff_col in pivot_df.columns:
            mse = np.var(pivot_df[diff_col], ddof=1)
            se = sem(pivot_df[diff_col], nan_policy='omit')
            pivot_df[f"MSE_log_{param}"] = [mse] * len(pivot_df)
            pivot_df[f"SE_log_{param}"] = [se] * len(pivot_df)

    return pivot_df

def get_cv_intra_anova(df, param):
    df = df.copy()
    df["log_val"] = np.log(df[param])
    model = smf.ols("log_val ~ C(Sequence) + C(Period) + C(Treatment) + C(Subject):C(Sequence)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    ms_error = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
    return np.sqrt(np.exp(ms_error) - 1) * 100

def get_gmr_ci_from_model(df, param, alpha=0.1):
    df = df.copy()
    df = df[df[param].notna()]  # Удалить строки с NaN
    df["log_val"] = np.log(df[param])

    model = smf.ols("log_val ~ C(Sequence) + C(Period) + C(Treatment) + C(Subject):C(Sequence)", data=df).fit()

    coef = model.params["C(Treatment)[T.Test]"]
    se = model.bse["C(Treatment)[T.Test]"]

    # 90% CI
    t = model.t_test("C(Treatment)[T.Test]")
    ci_low, ci_upp = t.conf_int(alpha=alpha).flatten()

    gmr = np.exp(coef)
    ci_low = np.exp(ci_low)
    ci_upp = np.exp(ci_upp)
    return gmr, ci_low, ci_upp


def get_gmr_lsmeans(df, param, alpha=0.1):
    df = df.copy()
    df = df[df[param].notna()]
    df["log_val"] = np.log(df[param])
    model = smf.ols("log_val ~ C(Sequence) + C(Period) + C(Treatment) + C(Subject):C(Sequence)", data=df).fit()

    # Средние значения логов по Treatment-группам
    lsm_ref = model.predict(df.assign(Treatment="Ref")).mean()
    lsm_test = model.predict(df.assign(Treatment="Test")).mean()
    diff = lsm_test - lsm_ref
    gmr = np.exp(diff)

    # SE и CI по формуле из SAS
    mse = model.mse_resid
    n = len(df["Subject"].unique())
    se = np.sqrt(2 * mse / n)
    dfree = model.df_resid
    tval = t.ppf(1 - alpha / 2, dfree)
    ci_low = np.exp(diff - tval * se)
    ci_high = np.exp(diff + tval * se)

    return gmr, ci_low, ci_high


def mixed_anova(df, colname):
    """
    Вложенная ANOVA:
     - между‐субъектный фактор: Sequence
     - внутри‐субъектные факторы: Treatment, Period
     возвращает DataFrame с SS, DF, F Value и Pr > F
    """
    df2 = df.copy()
    df2 = df2.dropna(subset=[colname, "Subject", "Sequence", "Period", "Treatment"])
    df2["log_val"] = np.log(df2[colname])
    # Все факторы — строки
    for c in ("Subject","Sequence","Period","Treatment"):
        df2[c] = df2[c].astype(str)
    # запускаем repeated‐measures ANOVA
    aov = AnovaRM(df2,
                  depvar="log_val",
                  subject="Subject",
                  within=["Treatment","Period"],
                  between=["Sequence"]).fit()
    return aov.anova_table



def nested_anova(df, col):
    """
    Делает ANOVA для 2×2 кроссовера с Sequence (между-субъектный),
    Subject(Sequence) (случайный, вложенный), Treatment и Period (внутри-субъектные).
    Возвращает словарь со sum_sq, df, mean_sq, F и p для каждого фактора
    и отдельно вычисляет residual MS и nested test для Sequence.
    """
    # 1) Лог-преобразуем
    df2 = df.copy()
    df2["lnY"] = np.log(df2[col])
    # 2) Глобальный средний
    GM = df2["lnY"].mean()
    # 3) Число повторов на субъекта (должно быть 2)
    n_rep = int(df2.groupby("Subject")["lnY"].count().iloc[0])
    # 4) Количество субъектов и последовательностей
    n_subj      = df2["Subject"].nunique()
    n_sequence  = df2["Sequence"].nunique()
    # 5) SS для Sequence
    seq_means = df2.groupby("Sequence")["lnY"].mean()
    SS_seq = n_rep * ((seq_means - GM) ** 2).sum()
    df_seq = n_sequence - 1
    # 6) SS для Subject(Sequence)
    # 6a) средние по субъекту
    subj_means = df2.groupby("Subject")["lnY"].mean()
    # 6b) для каждой subject достаём mean его Sequence
    subj_seq = df2.groupby("Subject")["Sequence"].first().map(seq_means)
    SS_subj = n_rep * ((subj_means - subj_seq) ** 2).sum()
    df_subj = n_subj - n_sequence
    # 7) SS для Treatment
    trt_means = df2.groupby("Treatment")["lnY"].mean()
    SS_trt = n_subj * ((trt_means - GM) ** 2).sum()
    df_trt = df2["Treatment"].nunique() - 1
    # 8) SS для Period
    per_means = df2.groupby("Period")["lnY"].mean()
    SS_per = n_subj * ((per_means - GM) ** 2).sum()
    df_per = df2["Period"].nunique() - 1
    # 9) Общий SS и DF
    SS_tot = ((df2["lnY"] - GM) ** 2).sum()
    df_tot = len(df2) - 1
    # 10) Остаточная (residual) SS и DF
    SS_err = SS_tot - (SS_seq + SS_subj + SS_trt + SS_per)
    df_err = df_tot - (df_seq + df_subj + df_trt + df_per)
    # 11) Mean Squares
    MS = {
        "Sequence":         SS_seq / df_seq,
        "subject(Sequence)": SS_subj / df_subj,
        "formulation":      SS_trt / df_trt,
        "Period":           SS_per / df_per,
        "Error":            SS_err / df_err
    }
    # 12) F-тесты против residual MS
    F = {factor: MS[factor] / MS["Error"] for factor in ("Sequence","formulation","subject(Sequence)","Period")}
    p = {factor: 1 - f_dist.cdf(F[factor],
                                df_seq   if factor=="Sequence" else
                                df_subj  if factor=="subject(Sequence)" else
                                df_trt   if factor=="formulation" else
                                df_per,
                                df_err)
         for factor in F}
    # 13) Специальный nested-test для Sequence против MS(subject)
    F_seq_n = MS["Sequence"] / MS["subject(Sequence)"]
    p_seq_n = 1 - f_dist.cdf(F_seq_n, df_seq, df_subj)
    # 14) Собираем всё в словарь
    return {
        "ss":  {"Sequence": SS_seq,  "subject(Sequence)": SS_subj,  "formulation": SS_trt,  "Period": SS_per,  "Error": SS_err},
        "df":  {"Sequence": df_seq, "subject(Sequence)": df_subj, "formulation": df_trt, "Period": df_per, "Error": df_err},
        "ms":  MS,
        "F":   F,
        "p":   p,
        "nested_test": {"Sequence": {"F": F_seq_n, "p": p_seq_n}}
    }
def compute_group_dynamics(
    oak_df: pd.DataFrame,
    param_col: str,
    subject_col: str = "№ п/п",
    stage_col: str   = "Этап регистрации",
    group_col: str   = "После приема"
) -> pd.DataFrame:
    # 1) Скрининг → baseline
    df_base = (
        oak_df[oak_df[stage_col]
              .str.contains("Скрин", case=False, na=False)]
        [[subject_col, param_col]]
        .rename(columns={param_col: "baseline"})
    )

    # 2) Только «Период 2 – в конце периода» → followup + группа
    df_end = (
        oak_df[oak_df[stage_col]
              .str.contains(r"Период\s*2", case=False, na=False)]
        [[subject_col, group_col, param_col]]
        .rename(columns={param_col: "followup"})
    )

    # 3) Объединяем и удаляем неполные записи
    merged = pd.merge(df_base, df_end, on=subject_col, how="inner")
    merged = merged.dropna(subset=["followup"])

    # 4) Средние по группам
    return (
        merged
        .groupby(group_col)[["baseline", "followup"]]
        .mean()
        .reset_index()
    )


def compute_group_iqr(
    oak_df: pd.DataFrame,
    param_col: str,
    subject_col: str = "№ п/п",
    stage_col: str   = "Этап регистрации",
    group_col: str   = "После приема"
) -> pd.DataFrame:
    """
    Считает для каждой из групп T/R и для двух этапов (baseline = Скрининг, followup = Период 2)
    квартили Q1, медиану и Q3 по параметру param_col и возвращает DataFrame
    с колонками [group_col, stage, q1, median, q3].
    """
    # 1) baseline → Скрининг, сначала парсим диапазоны, потом к числу
    df_base = (
        oak_df.loc[
            oak_df[stage_col].str.contains("Скрин", na=False),
            [subject_col, param_col]
        ]
        .rename(columns={param_col: "baseline"})
    )

    def _parse_range(x):
        if isinstance(x, str) and "-" in x:
            low, high = x.split("-", 1)
            try:
                return (float(low.strip()) + float(high.strip())) / 2
            except:
                return x
        return x

    df_base["baseline"] = df_base["baseline"].apply(_parse_range)
    df_base["baseline"] = pd.to_numeric(df_base["baseline"], errors="coerce")

    # 2) followup → «Период 2»
    df_end = (
        oak_df.loc[
            oak_df[stage_col].str.contains(r"Период\s*2", na=False),
            [subject_col, group_col, param_col]
        ]
        .rename(columns={param_col: "followup"})
    )

    df_end["followup"] = df_end["followup"].apply(_parse_range)
    df_end["followup"] = pd.to_numeric(df_end["followup"], errors="coerce")

    # 3) Объединяем и убираем недостающие
    merged = pd.merge(df_base, df_end, on=subject_col, how="inner")
    merged = merged.dropna(subset=["followup"])

    # Если после объединения нет данных — выходим сразу
    if merged.empty:
        return pd.DataFrame(columns=[group_col, "stage", "q1", "median", "q3"])


    # 4) Считаем квартильные метрики для baseline и followup
    base_q = (
        merged
        .groupby(group_col)["baseline"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.50: "median", 0.75: "q3"})
        .reset_index()
        .assign(stage="baseline")
    )

    foll_q = (
        merged
        .groupby(group_col)["followup"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.50: "median", 0.75: "q3"})
        .reset_index()
        .assign(stage="followup")
    )

    # 5) Склеиваем две таблицы (baseline + followup) и возвращаем в нужном порядке колонок
    iqr_df = pd.concat([base_q, foll_q], ignore_index=True)
    return iqr_df[[group_col, "stage", "q1", "median", "q3"]]
def compute_vitals_iqr(
    df: pd.DataFrame,
    param_col: str,
    group_col: str = "Препарат",
    stage_col: str = "Этап регистрации"
) -> pd.DataFrame:
    """
    Считает Q1/Median/Q3 для каждого сочетания:
       – продукт (T или R)
       – этап регистрации (при госпитализации, через 1 ч, 2 ч …)
    без фильтрации по Period.
    """
    # 1) оставляем только колонки Препарат, Этап регистрации и интересующий параметр
    sel = df[[group_col, stage_col, param_col]].copy()

    # 2) приводим к числу и отбрасываем пустые
    sel[param_col] = pd.to_numeric(sel[param_col], errors="coerce")
    sel = sel.dropna(subset=[param_col])

    # 3) считаем квартели
    q = sel.groupby([group_col, stage_col])[param_col].quantile([0.25, 0.5, 0.75])
    iqr_df = q.unstack(level=-1)
    iqr_df.columns = ["q1", "median", "q3"]

    return iqr_df.reset_index()

def identify_be_outlier_and_recommend(df_conc: pd.DataFrame,
                                      pk_df: pd.DataFrame,
                                      param: str,
                                      ci_lower: float,
                                      ci_upper: float,
                                      be_limits: tuple = (0.8, 1.25)
                                     ):
    """
    1) Проверяет, лежит ли 90% CI по param в пределах be_limits.
    2) Если нет — через pivot вычисляет log-разность Test/Ref,
       находит субъект с max|log_diff|.
    3) Формирует рекомендации: вместо его кривой —
       средняя концентрация остальных добровольцев.
    """
    # 1) проверка BE
    if not (be_limits[0] <= ci_lower <= ci_upper <= be_limits[1]):
        # 2) pivot и лог-разница
        pivot = pk_df.pivot(index="Subject",
                            columns="Treatment",
                            values=param)
        pivot.columns = list(pivot.columns)  # ['Ref', 'Test']
        pivot = pivot.rename(columns={
            'Test': f'{param}_Test',
            'Ref':  f'{param}_Ref'
        })

        # 2.1) проверяем наличие колонок
        test_col = f"{param}_Test"
        ref_col  = f"{param}_Ref"
        for col in (test_col, ref_col):
            if col not in pivot.columns:
                raise KeyError(f"В identify_be_outlier_and_recommend не найдена колонка «{col}»")

        # 2.2) создаём маску для безопасного логарифмирования (никаких нулей)
        mask = (pivot[test_col] > 0) & (pivot[ref_col] > 0)
        diff_col = f"log_{param}_diff"
        pivot[diff_col] = np.nan
        pivot.loc[mask, diff_col] = np.log(pivot.loc[mask, test_col] / pivot.loc[mask, ref_col])

        # выбираем «виновника»
        outlier = pivot[diff_col].abs().idxmax()

        # 3) рекомендации по концентрациям
        rec_list = []
        for t in sorted(df_conc["Time"].unique()):
            mean_other = (
                df_conc[
                  (df_conc["Time"] == t) &
                  (df_conc["Treatment"] == "Test") &
                  (df_conc["Subject"] != outlier)
                ]["Concentration"]
                .mean()
            )
            rec_list.append({
                "Subject": outlier,
                "Time": t,
                "Recommended_Concentration": mean_other
            })
        rec_df = pd.DataFrame(rec_list)

        return outlier, rec_df

    # BE соблюдена
    return None, None

def generate_conclusions(df,
                         param_name: str,
                         group_col: str = "group",
                         before_col: str = "before",
                         after_col: str = "after",
                         test_label: str = "исследуемого препарата",
                         ref_label: str = "препарата сравнения") -> dict:
    """
    df: DataFrame с колонками [Subject, group, before, after]
    param_name: человекочитаемое название параметра (для текста)
    Возвращает dict с ключами 'T' и 'R' — списки строк:
      "- повышение уровня АЛТ (на 43%);"
    """
    conclusions = {"T": [], "R": []}
    for grp, label in [("T", test_label), ("R", ref_label)]:
        sub = df[df[group_col] == grp]
        for _, row in sub.iterrows():
            b, a = row[before_col], row[after_col]
            if b is None or b == 0:
                continue
            pct = (a - b) / b * 100
            verb = "повышение" if pct > 0 else "снижение"
            conclusions[grp].append(
                f"- {verb} уровня {param_name} (на {abs(pct):.0f}%);"
            )
    return conclusions

