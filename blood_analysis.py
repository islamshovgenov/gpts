import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Константы параметров ===
OAK_PARAMS = [
    "Гемоглобин, г/л", "Гематокрит, %", "Эритроциты, 10¹²/л",
    "Лейкоциты, 10⁹/л", "Базофилы, %", "Палочкоядерные нейтрофилы, %",
    "Сегментоядерные нейтрофилы, %", "Лимфоциты, %", "Моноциты, %",
    "Эозинофилы, %", "Тромбоциты, 10⁹/л", "СОЭ, мм/ч"
]
BHAK_PARAMS = [
    "АЛТ, Ед./л", "АСТ, Ед./л", "Щелочная фосфатаза, Ед./л",
    "Билирубин общий, мкмоль/л", "Креатинин, мкмоль/л",
    "Глюкоза, ммоль/л", "Общий белок, г/л", "Холестерин общий, ммоль/л"
]
OAM_PARAM_KEYS = [
    "pH",
    "Относительная плотность, г/мл",
    "Белок, г/л",
    "Глюкоза, ммоль/л",
    "Лейкоциты, в п/зр.",
    "Эритроциты, в п/зр."
]

def load_oak_sheet(xls_file):
    xls_file.seek(0)
    raw = pd.read_excel(
        xls_file,
        sheet_name="ОАК",
        header=[1, 4],
        engine="openpyxl"
    ).dropna(axis=1, how="all")

    def make_name(l0, l1):
        if l0 == "№ п/п": return "№ п/п"
        if l0 == "Этап регистрации" or l1 == "Этап регистрации":
            return "Этап регистрации"
        if l1 == "После приема" or l0 == "После приема":
            return "После приема"
        if l1 == "значе-ние" and l0 in OAK_PARAMS:
            return l0
        return None

    raw.columns = [make_name(l0, l1) for (l0, l1) in raw.columns]

    # Валидация обязательных колонок
    required = ["№ п/п", "Этап регистрации", "После приема"] + OAK_PARAMS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"В load_oak_sheet отсутствуют колонки: {missing}")

    df = raw.loc[:, required].copy().reset_index(drop=True)
    return df


def extract_oak_pairwise(df, param_col, subject_col="№ п/п", stage_col="Этап регистрации"):
    records = []
    for subj in df[subject_col].unique():
        grp = df[df[subject_col] == subj]
        before = grp[grp[stage_col].str.lower().str.contains("скри", na=False)]
        after  = grp[grp[stage_col].str.lower().str.contains("конце", na=False)]
        if before.empty or after.empty:
            continue
        try:
            b = float(before[param_col].values[0])
            a = float(after[param_col].values[0])
        except (ValueError, TypeError):
            continue
        records.append({"Subject": subj, "Before": b, "After": a})
    return pd.DataFrame(records)

def plot_oak_pairwise(df_pair, title, ylabel):
    """График сравнения до/после по параметру"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.errorbar(["до", "после"],
                [df_pair["Before"].mean(), df_pair["After"].mean()],
                yerr=[df_pair["Before"].std(), df_pair["After"].std()],
                fmt='o-', capsize=4, color='blue')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    return fig

def plot_all_oak_parameters(df, param_dict):
    """Цикл по всем параметрам"""
    figs = []
    for param_col, (title, ylabel) in param_dict.items():
        pair_df = extract_oak_pairwise(df, param_col)
        if pair_df.empty:
            continue
        fig = plot_oak_pairwise(pair_df, title, ylabel)
        figs.append((title, fig))
    return figs


def load_bhak_sheet(xls_file):
    xls_file.seek(0)
    raw = pd.read_excel(
        xls_file,
        sheet_name="БХАК",
        header=[1, 4],
        engine="openpyxl"
    ).dropna(axis=1, how="all")

    def make_name(l0, l1):
        if l0 == "№ п/п": return "№ п/п"
        if l0 == "Этап регистрации" or l1 == "Этап регистрации":
            return "Этап регистрации"
        if l1 == "После приема" or l0 == "После приема":
            return "После приема"
        if l1 == "значе-ние" and l0 in BHAK_PARAMS:
            return l0
        return None

    raw.columns = [make_name(a, b) for a, b in raw.columns]
    required = ["№ п/п", "Этап регистрации", "После приема"] + BHAK_PARAMS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"В load_bhak_sheet отсутствуют колонки: {missing}")
    df = raw.loc[:, required].copy().reset_index(drop=True)
    return df

def load_oam_sheet(xls_file):
    xls_file.seek(0)
    raw = pd.read_excel(
        xls_file,
        sheet_name="ОАМ",
        header=[1, 4],
        engine="openpyxl",
        engine_kwargs={"data_only": True}
    ).dropna(axis=1, how="all")

    def make_name(l0, l1):
        if l0 == "№ п/п": return "№ п/п"
        if "Этап регистрации" in (l0, l1): return "Этап регистрации"
        if "После приема" in (l0, l1): return "После приема"
        if isinstance(l1, str) and l1.lower().startswith("знач"):
            key0 = l0.strip().rstrip(",")
            if key0.lower() == "рн": return "pH"
            for pk in OAM_PARAM_KEYS:
                base = pk.split(",")[0]
                if key0.startswith(base): return pk
        return None

    raw.columns = [make_name(a, b) for a, b in raw.columns]
    required = ["№ п/п", "Этап регистрации", "После приема"] + OAM_PARAM_KEYS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"В load_oam_sheet отсутствуют колонки: {missing}")
    df = raw.loc[:, required].copy().reset_index(drop=True)
    return df


def extract_individual_lab(df, param_col,
                           subject_col="№ п/п",
                           stage_col="Этап регистрации",
                           group_col="После приема"):
    required = [subject_col, stage_col, group_col, param_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"В extract_individual_lab отсутствуют колонки: {missing}")

    df_before = (
        df[df[stage_col].str.contains("Скрин", na=False)]
        [[subject_col, param_col]]
        .rename(columns={param_col: "before"})
    )
    df_after = (
        df[df[stage_col].str.contains(r"Период\s*2", na=False)]
        [[subject_col, group_col, param_col]]
        .rename(columns={param_col: "after"})
    )
    merged = pd.merge(df_before, df_after,
                      on=subject_col, how="inner")

    def _parse_range(x):
        if isinstance(x, str) and "-" in x:
            low, high = x.split("-", 1)
            try:
                return (float(low.strip()) + float(high.strip())) / 2
            except (ValueError, TypeError):
                return np.nan
        return x

    merged["before"] = merged["before"].apply(_parse_range)
    merged["after"]  = merged["after"].apply(_parse_range)
    merged["before"] = pd.to_numeric(merged["before"], errors="coerce")
    merged["after"]  = pd.to_numeric(merged["after"], errors="coerce")
    merged = merged.dropna(subset=["before", "after"]).reset_index(drop=True)
    return merged


# словарь параметров витальных функций – здесь, чтобы load_vitals_sheet мог его увидеть
vitals_params = {
    "АД систолическое, мм рт. ст.":    ("АД систолическое",    "мм рт. ст."),
    "АД диастолическое, мм рт. ст.":   ("АД диастолическое",   "мм рт. ст."),
    "ЧСС, уд/мин":                     ("ЧСС",                 "уд./мин"),
    "ЧДД, в мин":                      ("ЧДД",                 "в мин"),
    "Температура тела, °C":            ("Температура тела",    "°C"),
}


# blood_analysis.py

def load_vitals_sheet(xls_file):
    xls_file.seek(0)
    raw = pd.read_excel(
        xls_file,
        sheet_name="витальные",
        header=[0, 1],
        engine="openpyxl",
        engine_kwargs={"data_only": True}
    ).dropna(axis=1, how="all")

    # делаем одноуровневые имена колонок
    raw.columns = [str(col[0]).strip() for col in raw.columns]

    # 1️⃣ Ищем имя колонки с этапами регистрации (любой вариант "витальн⋯")
    candidates = [c for c in raw.columns if "витальн" in c.lower()]
    if not candidates:
        raise KeyError(f"Не найден столбец этапов регистрации в листе 'витальные'. "
                       f"Доступные колонки: {raw.columns.tolist()}")
    stage_old = candidates[0]

    # 2️⃣ Собираем map: динамический для этапов + остальное как раньше
    rename_map = {
        "Период":                   "Период",
        "Препарат":                 "Препарат",
        stage_old:                  "Этап регистрации",
        "АД сист.":                 "АД систолическое, мм рт. ст.",
        "АД диаст.":                "АД диастолическое, мм рт. ст.",
        "ЧСС":                      "ЧСС, уд/мин",
        "ЧДД":                      "ЧДД, в мин",
        "Темпер. тела":             "Температура тела, °C",
    }

    df = raw.rename(columns=rename_map)

    # 3️⃣ Оставляем только нужные колонки
    keep = ["Период", "Препарат", "Этап регистрации"] + list(vitals_params)
    df = df.loc[:, [c for c in keep if c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
    return df



def load_stage_order(
    xls_file,
    sheet_name: str = "этапы-процедуры"
) -> list[str]:
    """
    Из листа «этапы-процедуры» берём колонку со «Скрининг»,
    затем вниз по ней все непустые строки до первой строки,
    начинающейся на «Период».
    """
    xls_file.seek(0)
    xls = pd.ExcelFile(xls_file)

    # 1) Находим лист, имя которого содержит «этап»
    candidates = [s for s in xls.sheet_names if "этап" in s.lower()]
    if not candidates:
        raise KeyError(f"Нет листа с 'этап' в имени. Доступные: {xls.sheet_names!r}")
    sheet = candidates[0]

    # 2) Читаем его целиком без заголовков
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)

    # 3) Ищем точную ячейку «Скрининг» (по подстроке)
    mask = raw.apply(lambda col: col.str.contains("скринин", case=False, na=False))
    locs = list(zip(*mask.values.nonzero()))
    if not locs:
        raise KeyError(f"Не найден заголовок 'Скрининг' на листе '{sheet}'")
    row0, col0 = locs[0]

    # 4) Забираем весь столбец начиная с этой ячейки вниз
    col_vals = raw.iloc[row0:, col0].astype(str).str.strip().tolist()

    # 5) Берём все значения после «Скрининг» до первой «Период…»
    stages = []
    for v in col_vals[1:]:     # пропускаем сам «Скрининг»
        if not v or v.lower().startswith("период"):
            break
        stages.append(v)

    return ["Скрининг"] + stages
