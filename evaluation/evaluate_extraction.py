# computes precision/recall/f1 for claude extraction vs gold annotations.
# reads: data/synthetic/gold_annotations.csv + pipeline_results(_{lang}).csv
# saves: evaluation/results/extraction_results.csv (twi) or extraction_results_{lang}.csv
# run: python -m evaluation.evaluate_extraction
#      python -m evaluation.evaluate_extraction --lang ga
#      python -m evaluation.evaluate_extraction --all
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

from pipeline.eval_subset import get_eval_id_set
from pipeline.lang_config import EVAL_LANGUAGES_FIVE

GOLD_PATH = Path("data/synthetic/gold_annotations.csv")


def _pipeline_and_output(lang: str) -> tuple[Path, Path]:
    if lang == "twi":
        return (
            Path("data/synthetic/pipeline_results.csv"),
            Path("evaluation/results/extraction_results.csv"),
        )
    return (
        Path(f"data/synthetic/pipeline_results_{lang}.csv"),
        Path(f"evaluation/results/extraction_results_{lang}.csv"),
    )

CATEGORICAL_FIELDS = ["injury_type", "severity", "body_region", "victim_sex", "victim_age_group"]


def compute_field_f1(gold: list, pred: list, fields: list = None) -> dict:
    # compute per-field macro-f1 and overall macro average across fields
    if fields is None:
        fields = CATEGORICAL_FIELDS
    if not gold or not pred:
        empty = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        return {f: empty for f in fields} | {"macro_f1": 0.0}

    results = {}
    f1_scores = []

    for field in fields:
        y_true = [str(r.get(field, "unknown")).lower() for r in gold]
        y_pred = [str(r.get(field, "unknown")).lower() for r in pred]
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        field_f1 = macro.get("f1-score", 0.0)
        results[field] = {
            "f1": round(field_f1, 4),
            "precision": round(macro.get("precision", 0.0), 4),
            "recall": round(macro.get("recall", 0.0), 4),
        }
        f1_scores.append(field_f1)

    results["macro_f1"] = round(float(np.mean(f1_scores)), 4)
    return results


def run_extraction_evaluation(lang: str) -> dict | None:
    pipeline_path, output_path = _pipeline_and_output(lang)
    if not GOLD_PATH.exists():
        raise SystemExit(f"missing {GOLD_PATH}")
    if not pipeline_path.exists():
        print(f"[{lang}] skip — pipeline file not found: {pipeline_path}")
        return None

    gold_df = pd.read_csv(GOLD_PATH)
    pred_df = pd.read_csv(pipeline_path)
    subset = get_eval_id_set()
    if subset is not None:
        gold_df = gold_df[gold_df["id"].isin(subset)]
        pred_df = pred_df[pred_df["id"].isin(subset)]
    merged = gold_df.merge(pred_df, on="id", suffixes=("_gold", "_pred"))

    gold = [{f: merged.iloc[i][f"{f}_gold"] for f in CATEGORICAL_FIELDS} for i in range(len(merged))]
    pred = [{f: merged.iloc[i][f"{f}_pred"] for f in CATEGORICAL_FIELDS} for i in range(len(merged))]

    results = compute_field_f1(gold, pred)

    rows = [{"field": f, **v} for f, v in results.items() if f != "macro_f1"]
    rows.append({"field": "MACRO AVG", "f1": results["macro_f1"], "precision": None, "recall": None})
    results_df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[{lang}] post-ASR extraction vs gold (n={len(merged)} merged ids)")
    print(results_df.to_string(index=False))
    print(f"\nheadline macro-f1: {results['macro_f1']:.4f}")
    print(f"saved -> {output_path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Extraction F1 vs gold from pipeline_results (post-ASR path).")
    p.add_argument(
        "--lang",
        choices=list(EVAL_LANGUAGES_FIVE),
        default=None,
        help="single language (default: twi if neither --lang nor --all)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="run for all languages in EVAL_LANGUAGES_FIVE",
    )
    args = p.parse_args()
    if args.all:
        for lang in EVAL_LANGUAGES_FIVE:
            run_extraction_evaluation(lang)
    elif args.lang is not None:
        run_extraction_evaluation(args.lang)
    else:
        run_extraction_evaluation("twi")


if __name__ == "__main__":
    main()
