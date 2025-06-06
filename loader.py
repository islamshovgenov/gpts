import io
import pandas as pd
import re
import numpy as np

def load_randomization(file):
    df = pd.read_csv(file, encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    return dict(zip(df["Subject"], df["Sequence"]))

def load_timepoints(file):
    df = pd.read_csv(file)
    return dict(zip(df["Code"].astype(str).str.zfill(2), df["Time"]))

def parse_excel_files(xlsx_files, rand_dict, time_dict):
    data = []
    for uploaded_file in xlsx_files:
        # Определяем, как читать: путь или поток
        try:
            if hasattr(uploaded_file, "read"):
                # Streamlit UploadedFile: читаем байты и оборачиваем в BytesIO
                content = uploaded_file.read()
                xls = pd.ExcelFile(io.BytesIO(content))
            else:
                # Обычная строка пути
                xls = pd.ExcelFile(uploaded_file)
        except Exception as e:
            print(f"[WARNING] Невозможно открыть файл {getattr(uploaded_file,'name',uploaded_file)} как Excel: {e}")
            continue

        # ------ Новый блок: формат CT, конверсия pg→µg ------
        if 'CT' in xls.sheet_names:
            df_ct = xls.parse('CT', header=28)
            required = {
                'Subject',
                'Time',
                'Concentration, Period 1',
                'Concentration, Period 2'
            }
            if required.issubset(df_ct.columns):
                print(f"→ Загружаем CT-формат из {uploaded_file}")
                for _, row in df_ct.iterrows():
                    if pd.isna(row['Subject']) or pd.isna(row['Time']):
                        continue
                    subj = int(row['Subject'])
                    t    = float(row['Time'])
                    seq  = rand_dict.get(subj, 'TR')

                    for period in (1, 2):
                        raw = row[f'Concentration, Period {period}']
                        if pd.isna(raw):
                            continue
                        try:
                            conc = float(raw)
                        except (ValueError, TypeError):
                            continue

                        if (seq == 'TR' and period == 1) or (seq == 'RT' and period == 2):
                            treat = 'Test'
                        else:
                            treat = 'Ref'

                        data.append([
                            subj, seq, period, treat,
                            None, None,
                            t,        # Time
                            conc      # Concentration уже в ng/mL
                        ])
                continue
            else:
                print(xls.sheet_names)
                print(df_ct.columns.tolist())

        for sheet in xls.sheet_names:
            if sheet == "CT" or sheet == "ValueList_Helper":
                continue
            df = xls.parse(sheet, header=1)
            for _, row in df.iterrows():
                try:
                    sample = str(row.get("SampleLabel") or row.get("Name") or row.get("Sample"))
                    sample = sample.strip()
                    if sample.endswith(".0"):
                        sample = sample[:-2]

                    if not isinstance(sample, str) or not sample.startswith(("A-", "B-")):
                        continue
                    match = re.match(r"([AB])-(\d+)-(\d+)", sample)
                    if not match:
                        continue
                    period = 1 if match.group(1) == 'A' else 2
                    subject = int(match.group(2))
                    timepoint = int(match.group(3))
                    timepoint_code = match.group(3).zfill(2)
                    real_time = time_dict.get(timepoint_code)
                    if real_time is None:
                        continue
                    seq = rand_dict.get(subject, 'TR')
                    treatment = 'Test' if (seq == 'TR' and period == 1) or (seq == 'RT' and period == 2) else 'Ref'
                    raw_ug = row.get("Calc. Conc., ug/ml")
                    if pd.isna(raw_ug):
                        raw_ug = row.iloc[13] if len(row) > 13 else (row.iloc[12] if len(row) > 12 else None)
                    if raw_ug is None or pd.isna(raw_ug):
                        continue
                    try:
                        conc_val = float(raw_ug) * 1000  # µg/mL → ng/mL
                    except ValueError:
                        continue
                    if conc_val < 1:
                        conc_val = 0.0
                    if not np.isnan(conc_val):
                        data.append([subject, seq, period, treatment, sample, timepoint, real_time, conc_val])
                except (ValueError, KeyError, TypeError):
                    continue
    df = pd.DataFrame(data, columns=[
        "Subject", "Sequence", "Period", "Treatment",
        "SampleLabel", "TimeCode", "Time", "Concentration"
    ])
    return df
