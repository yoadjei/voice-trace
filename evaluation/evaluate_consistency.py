import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

from pipeline.eval_subset import get_eval_id_set
from pipeline.lang_config import EVAL_LANGUAGES_FIVE


LANGUAGES = list(EVAL_LANGUAGES_FIVE)
FIELDS = ["injury_type", "severity", "body_region", "victim_sex", "victim_age_group"]
DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
RESULTS_DIR = Path(__file__).parent / "results"


def load_extraction(lang):
    path = DATA_DIR / f"extraction_{lang}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"failed to load {lang}: {e}")
        return None


def load_gold():
    path = DATA_DIR / "gold_annotations.csv"
    if not path.exists():
        raise FileNotFoundError(f"gold annotations not found at {path}")
    return pd.read_csv(path)


def compute_kappa(y1, y2, field_name):
    # handle mismatched ids: inner join implicit via aligned indices
    if len(y1) != len(y2):
        raise ValueError(f"mismatched lengths for {field_name}: {len(y1)} vs {len(y2)}")

    # check for missing values
    mask = y1.notna() & y2.notna()
    if not mask.any():
        warnings.warn(f"no valid pairs for {field_name}")
        return np.nan

    y1_valid = y1[mask]
    y2_valid = y2[mask]

    # check for single unique label (no variance = undefined kappa)
    if len(y1_valid.unique()) <= 1 or len(y2_valid.unique()) <= 1:
        warnings.warn(f"{field_name}: insufficient variance (single label or all null)")
        return np.nan

    # encode labels to integers for kappa computation
    all_labels = pd.concat([y1_valid, y2_valid]).unique()
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    y1_enc = y1_valid.map(label_map).values
    y2_enc = y2_valid.map(label_map).values

    return cohen_kappa_score(y1_enc, y2_enc)


def main():
    subset = get_eval_id_set()

    # load all extractions
    extractions = {}
    for lang in LANGUAGES:
        data = load_extraction(lang)
        if data is not None:
            if subset is not None:
                data = data[data["id"].isin(subset)]
            extractions[lang] = data
        else:
            print(f"warning: skipping {lang} (file not found or unreadable)")

    if len(extractions) < 2:
        print("error: fewer than 2 language files available")
        sys.exit(1)

    # load gold
    try:
        gold = load_gold()
        if subset is not None:
            gold = gold[gold["id"].isin(subset)]
    except FileNotFoundError as e:
        print(f"error: {e}")
        sys.exit(1)

    # create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # compute cross-language kappa
    langs = list(extractions.keys())
    cross_kappa_rows = []

    for i, lang1 in enumerate(langs):
        for lang2 in langs[i+1:]:
            df1 = extractions[lang1]
            df2 = extractions[lang2]

            # inner join on id
            merged = df1[["id"] + FIELDS].merge(
                df2[["id"] + FIELDS], on="id", suffixes=("_1", "_2")
            )

            if merged.empty:
                print(f"warning: no common ids between {lang1} and {lang2}")
                continue

            row = {"lang1": lang1, "lang2": lang2}
            for field in FIELDS:
                col1 = f"{field}_1"
                col2 = f"{field}_2"
                kappa = compute_kappa(merged[col1], merged[col2], f"{lang1}-{lang2}:{field}")
                row[field] = kappa

            cross_kappa_rows.append(row)

    cross_df = pd.DataFrame(cross_kappa_rows)

    # compute vs-gold kappa
    vs_gold_rows = []

    for lang in langs:
        df = extractions[lang]

        # inner join on id with gold
        merged = df[["id"] + FIELDS].merge(
            gold[["id"] + FIELDS], on="id", suffixes=("_ext", "_gold")
        )

        if merged.empty:
            print(f"warning: no common ids between {lang} and gold")
            continue

        row = {"language": lang}
        for field in FIELDS:
            col_ext = f"{field}_ext"
            col_gold = f"{field}_gold"
            kappa = compute_kappa(merged[col_ext], merged[col_gold], f"{lang}:gold:{field}")
            row[field] = kappa

        vs_gold_rows.append(row)

    vs_gold_df = pd.DataFrame(vs_gold_rows)

    # save and print results
    cross_kappa_path = RESULTS_DIR / "kappa_cross_language.csv"
    vs_gold_path = RESULTS_DIR / "kappa_vs_gold.csv"

    cross_df.to_csv(cross_kappa_path, index=False)
    vs_gold_df.to_csv(vs_gold_path, index=False)

    print("\n=== Cross-Language Kappa ===")
    print(cross_df.to_string(index=False))
    print(f"\nsaved to {cross_kappa_path}")

    print("\n=== Language vs. Gold Kappa ===")
    print(vs_gold_df.to_string(index=False))
    print(f"\nsaved to {vs_gold_path}")


if __name__ == "__main__":
    main()
