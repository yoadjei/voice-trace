# track 2 evaluation: translation round-trip fidelity + extraction accuracy across all 6 languages.
# inputs:
#   data/synthetic/narratives_en.csv         — original english narratives (reference)
#   data/synthetic/roundtrip_{lang}_en.csv   — back-translated english per language
#   data/synthetic/extraction_{lang}.csv     — extraction results per language
#   data/synthetic/gold_annotations.csv      — gold labels for extraction F1
# outputs:
#   evaluation/results/t2_fidelity.csv       — BERTScore + BLEU per language
#   evaluation/results/t2_extraction_f1.csv  — F1 per field per language
# run: python -m evaluation.evaluate_translation
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from bert_score import score as bert_score_fn
import sacrebleu

from pipeline.eval_subset import get_eval_id_set
from pipeline.lang_config import EVAL_LANGUAGES_FIVE

SYNTHETIC_DIR = Path("data/synthetic")
RESULTS_DIR = Path("evaluation/results")
LANGUAGES = list(EVAL_LANGUAGES_FIVE)
EVAL_FIELDS = ["injury_type", "severity", "body_region", "victim_sex", "victim_age_group"]

# location_description excluded (free text, not categorical)
# mechanism excluded (not in headline 6-field schema)


def _load_gold() -> pd.DataFrame:
    path = SYNTHETIC_DIR / "gold_annotations.csv"
    df = pd.read_csv(path)
    return df[["id"] + EVAL_FIELDS].copy()


def _compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    # sentence-level BLEU via sacrebleu, averaged across corpus
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(result.score, 2)


def _compute_bertscore(references: list[str], hypotheses: list[str]) -> tuple[float, float, float]:
    # returns (precision, recall, f1) averaged across examples
    P, R, F1 = bert_score_fn(hypotheses, references, lang="en", model_type="bert-base-uncased", verbose=False)
    return round(P.mean().item(), 4), round(R.mean().item(), 4), round(F1.mean().item(), 4)


def _f1_per_field(gold: pd.DataFrame, pred: pd.DataFrame, fields: list[str]) -> dict[str, float]:
    # compute macro f1 per field on aligned (id inner-joined) rows
    merged = gold.merge(pred[["id"] + fields], on="id", how="inner", suffixes=("_gold", "_pred"))
    if merged.empty:
        return {f: float("nan") for f in fields}
    results = {}
    for field in fields:
        gcol = f"{field}_gold" if f"{field}_gold" in merged.columns else field
        pcol = f"{field}_pred" if f"{field}_pred" in merged.columns else field
        g = merged[gcol].fillna("unknown").astype(str).tolist()
        p = merged[pcol].fillna("unknown").astype(str).tolist()
        try:
            results[field] = round(f1_score(g, p, average="macro", zero_division=0), 4)
        except Exception:
            results[field] = float("nan")
    return results


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    en_df = pd.read_csv(SYNTHETIC_DIR / "narratives_en.csv")
    subset = get_eval_id_set()
    if subset is not None:
        en_df = en_df[en_df["id"].isin(subset)]
    gold = _load_gold()
    if subset is not None:
        gold = gold[gold["id"].isin(subset)]

    fidelity_rows = []
    f1_rows = []

    for lang in LANGUAGES:
        roundtrip_path = SYNTHETIC_DIR / f"roundtrip_{lang}_en.csv"
        extraction_path = SYNTHETIC_DIR / f"extraction_{lang}.csv"

        # --- fidelity (BERTScore + BLEU) ---
        if not roundtrip_path.exists():
            print(f"[{lang}] roundtrip file missing — skipping fidelity")
        else:
            rt = pd.read_csv(roundtrip_path)
            if subset is not None:
                rt = rt[rt["id"].isin(subset)]
            merged = en_df[["id", "narrative_en"]].merge(rt[["id", "roundtrip_en"]], on="id", how="inner")
            if merged.empty:
                print(f"[{lang}] no matched rows for fidelity — skipping")
            else:
                refs = merged["narrative_en"].fillna("").tolist()
                hyps = merged["roundtrip_en"].fillna("").tolist()
                print(f"[{lang}] computing BERTScore ({len(refs)} pairs)...")
                bp, br, bf = _compute_bertscore(refs, hyps)
                bleu = _compute_bleu(refs, hyps)
                fidelity_rows.append({
                    "language": lang,
                    "n": len(refs),
                    "bertscore_p": bp,
                    "bertscore_r": br,
                    "bertscore_f1": bf,
                    "bleu": bleu,
                })
                print(f"  BERTScore F1={bf}  BLEU={bleu}")

        # --- extraction F1 ---
        if not extraction_path.exists():
            print(f"[{lang}] extraction file missing — skipping F1")
        else:
            pred = pd.read_csv(extraction_path)
            if subset is not None:
                pred = pred[pred["id"].isin(subset)]
            f1s = _f1_per_field(gold, pred, EVAL_FIELDS)
            row = {"language": lang, **f1s}
            # macro average across fields (excluding nan)
            vals = [v for v in f1s.values() if v == v]
            row["macro_avg"] = round(sum(vals) / len(vals), 4) if vals else float("nan")
            f1_rows.append(row)
            print(f"[{lang}] extraction F1: {f1s}")

    if fidelity_rows:
        fid_df = pd.DataFrame(fidelity_rows)
        out = RESULTS_DIR / "t2_fidelity.csv"
        fid_df.to_csv(out, index=False)
        print(f"\nfidelity results → {out}")
        print(fid_df.to_string(index=False))
    else:
        print("\nno fidelity results to save")

    if f1_rows:
        f1_df = pd.DataFrame(f1_rows)
        out = RESULTS_DIR / "t2_extraction_f1.csv"
        f1_df.to_csv(out, index=False)
        print(f"\nextraction F1 results → {out}")
        print(f1_df.to_string(index=False))
    else:
        print("\nno extraction F1 results to save")


if __name__ == "__main__":
    run()
