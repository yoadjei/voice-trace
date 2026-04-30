# optional cleanup: keep only evaluation_subset_ids in *derived* artifacts (not narratives or gold).
# does not touch narratives_*.csv, narratives_en.csv, gold_annotations*.csv, narratives_all_langs.csv.
#
# dry-run (default): print what would change
# apply: rewrite csvs to subset rows + delete wavs for ids outside subset
#
# run:  python -m data_gen.prune_derived_to_subset
#       python -m data_gen.prune_derived_to_subset --apply
import argparse
from pathlib import Path

import pandas as pd

from pipeline.eval_subset import SUBSET_PATH, get_eval_id_set

DATA_SYN = Path("data/synthetic")
AUDIO = DATA_SYN / "audio"


def _csv_candidates() -> list[Path]:
    out = []
    for p in DATA_SYN.glob("roundtrip_*_en.csv"):
        out.append(p)
    for p in DATA_SYN.glob("extraction_*.csv"):
        out.append(p)
    for name in (
        "asr_transcripts.csv",
        "translations_en.csv",
        "pipeline_results.csv",
    ):
        q = DATA_SYN / name
        if q.exists():
            out.append(q)
    for p in DATA_SYN.glob("asr_transcripts_*.csv"):
        out.append(p)
    for p in DATA_SYN.glob("translations_en_*.csv"):
        out.append(p)
    for p in DATA_SYN.glob("pipeline_results_*.csv"):
        out.append(p)
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually delete/prune (default is dry-run)")
    args = p.parse_args()

    subset = get_eval_id_set()
    if subset is None:
        raise SystemExit(
            f"missing {SUBSET_PATH} — nothing to prune. "
            "Generate with: python -m data_gen.select_eval_subset"
        )

    ids = sorted(subset)
    print(f"subset n={len(ids)} (from {SUBSET_PATH})")

    # csvs
    for path in _csv_candidates():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if "id" not in df.columns:
            print(f"skip {path}: no id column")
            continue
        before = len(df)
        filt = df[df["id"].isin(subset)].copy()
        after = len(filt)
        if before == after:
            print(f"ok {path.name} ({after} rows)")
            continue
        print(f"{'WRITE' if args.apply else 'would write'} {path.name}: {before} -> {after} rows")
        if args.apply:
            filt.to_csv(path, index=False)

    # wavs
    removed = 0
    if AUDIO.exists():
        for wav in sorted(AUDIO.rglob("narrative_*.wav")):
            try:
                stem = wav.stem  # narrative_042
                wid = int(stem.split("_")[-1])
            except (ValueError, IndexError):
                continue
            if wid not in subset:
                print(f"{'DELETE' if args.apply else 'would delete'} data/synthetic/{wav.relative_to(DATA_SYN)}")
                if args.apply:
                    wav.unlink()
                    removed += 1

    if args.apply:
        print(f"\ndone. removed {removed} wav file(s) outside subset.")
    else:
        print("\n(dry-run) pass --apply to prune files.")


if __name__ == "__main__":
    main()
