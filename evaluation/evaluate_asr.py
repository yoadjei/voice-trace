# computes wer for khaya asr output vs reference narratives.
# reads per-language narrative csv + matching asr_transcripts[ _{lang}].csv
# saves: evaluation/results/asr_results_{lang}.csv (or asr_results.csv for twi)
# run: python -m evaluation.evaluate_asr
#       python -m evaluation.evaluate_asr --lang ga
# languages: ASR_EVAL_LANGS in pipeline/lang_config.py (five langs; Gurene omitted)
import argparse
import pandas as pd
from jiwer import wer
from pathlib import Path
from pipeline.eval_subset import get_eval_id_set
from pipeline.lang_config import ASR_EVAL_LANGS

DATA_SYN = Path("data/synthetic")


def _paths(lang: str) -> tuple[Path, Path, Path]:
    col = f"narrative_{lang}"
    if lang == "twi":
        narratives = DATA_SYN / "narratives_twi.csv"
        legacy = DATA_SYN / "asr_transcripts.csv"
        named = DATA_SYN / "asr_transcripts_twi.csv"
        asr_path = legacy if legacy.exists() else named
        out = Path("evaluation/results/asr_results.csv")
    else:
        narratives = DATA_SYN / f"narratives_{lang}.csv"
        asr_path = DATA_SYN / f"asr_transcripts_{lang}.csv"
        out = Path(f"evaluation/results/asr_results_{lang}.csv")
    return narratives, asr_path, out


def compute_wer(references: list, hypotheses: list) -> dict:
    valid = [(r, h) for r, h in zip(references, hypotheses) if r and h]
    if not valid:
        return {"overall_wer": None, "n_references": 0}
    refs, hyps = zip(*valid)
    overall = wer(list(refs), list(hyps))
    return {"overall_wer": round(overall, 4), "n_references": len(valid)}


def run_asr_evaluation(lang: str) -> dict:
    narratives_path, asr_path, output_path = _paths(lang)
    col = f"narrative_{lang}"

    if not narratives_path.exists():
        raise SystemExit(f"missing {narratives_path}")
    if not asr_path.exists():
        raise SystemExit(f"missing {asr_path}")

    ref_df = pd.read_csv(narratives_path)
    asr_df = pd.read_csv(asr_path)
    merged = ref_df.merge(asr_df, on="id")
    subset = get_eval_id_set()
    if subset is not None:
        merged = merged[merged["id"].isin(subset)]

    en_df = pd.read_csv(DATA_SYN / "narratives_en.csv")[["id", "injury_type"]]
    merged = merged.merge(en_df, on="id", how="left")

    overall = compute_wer(
        merged[col].tolist(),
        merged["asr_transcript"].tolist()
    )

    per_type = []
    for injury_type, group in merged.groupby("injury_type"):
        type_wer = compute_wer(
            group[col].tolist(),
            group["asr_transcript"].tolist()
        )
        per_type.append({"injury_type": injury_type, **type_wer})

    results_df = pd.DataFrame(per_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n[{lang}] overall wer: {overall['overall_wer']:.1%} (n={overall['n_references']})")
    print(results_df.to_string(index=False))
    print(f"\nsaved -> {output_path}")
    return overall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lang",
        default="twi",
        choices=sorted(ASR_EVAL_LANGS),
        help="language with narratives_{lang}.csv + asr_transcripts(_{lang}).csv",
    )
    args = p.parse_args()
    run_asr_evaluation(args.lang)


if __name__ == "__main__":
    main()
