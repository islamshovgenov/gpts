import pandas as pd
from collections import Counter

from loader import load_randomization, load_timepoints, parse_excel_files


class DataLoaderError(Exception):
    pass


def load_data(rand_file, time_file, xlsx_files):
    """Load and validate concentration data from user uploads."""
    rand_dict = load_randomization(rand_file)

    seq_counts = Counter(rand_dict.values())
    if seq_counts.get("TR", 0) != seq_counts.get("RT", 0):
        raise DataLoaderError(
            f"Sequence imbalance: TR={seq_counts.get('TR',0)}, RT={seq_counts.get('RT',0)}"
        )

    time_dict = load_timepoints(time_file)

    df = parse_excel_files(xlsx_files, rand_dict, time_dict)

    file_points = len(time_dict)
    if 0 in map(float, time_dict.values()):
        file_points -= 1
    anal_points = df["Time"].nunique()
    if file_points != anal_points:
        raise DataLoaderError(
            f"Timepoints mismatch: {file_points} in file vs {anal_points} in data"
        )

    expected_subjects = set(rand_dict.keys())
    found_subjects = set(df["Subject"].unique())
    missing = expected_subjects - found_subjects
    if missing:
        raise DataLoaderError(
            f"Missing subjects in analytic files: {sorted(missing)}"
        )
    unexpected = found_subjects - expected_subjects
    if unexpected:
        raise DataLoaderError(
            f"Unexpected subjects in analytic files: {sorted(unexpected)}"
        )

    return df, rand_dict, time_dict
