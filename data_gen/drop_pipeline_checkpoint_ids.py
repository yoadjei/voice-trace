# Remove narrative id(s) from pipeline checkpoint CSVs so run_pipeline_batch.py will re-run them.
#
#   python -m data_gen.drop_pipeline_checkpoint_ids --ids 84,86,89 --lang twi
#   python -m data_gen.drop_pipeline_checkpoint_ids --ids 84,86 --all-langs
#   python -m data_gen.drop_pipeline_checkpoint_ids --ids-file path/to/ids.txt --all-langs --dry-run
#
# Twi uses legacy filenames; ga/ewe/dagbani use asr_transcripts_{lang}.csv, etc.
import argparse
from pathlib import Path

import pandas as pd

from pipeline.lang_config import FULL_VOICE_LANGS

DATA_SYN = Path("data/synthetic")


def _checkpoint_paths(lang: str) -> tuple[Path, Path, Path]:
    if lang == "twi":
        return (
            DATA_SYN / "asr_transcripts.csv",
            DATA_SYN / "translations_en.csv",
            DATA_SYN / "pipeline_results.csv",
        )
    return (
        DATA_SYN / f"asr_transcripts_{lang}.csv",
        DATA_SYN / f"translations_en_{lang}.csv",
        DATA_SYN / f"pipeline_results_{lang}.csv",
    )


def _parse_ids(args: argparse.Namespace) -> set[int]:
    out: set[int] = set()
    if args.ids.strip():
        for part in args.ids.split(","):
            part = part.strip()
            if part:
                out.add(int(part))
    if args.ids_file:
        p = Path(args.ids_file)
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(int(line))
    if not out:
        raise SystemExit("Provide --ids ... and/or --ids-file with at least one id.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Drop id(s) from ASR / translation / pipeline_results checkpoint CSVs."
    )
    p.add_argument("--ids", default="", help="Comma-separated narrative ids (e.g. 84,86,89)")
    p.add_argument(
        "--ids-file",
        default=None,
        help="Text file: one id per line (# comments ok)",
    )
    p.add_argument(
        "--lang",
        default=None,
        choices=sorted(FULL_VOICE_LANGS),
        help="One language: twi | ga | ewe | dagbani",
    )
    p.add_argument(
        "--all-langs",
        action="store_true",
        help="Apply to every language in FULL_VOICE_LANGS (twi, ga, ewe, dagbani)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files",
    )
    args = p.parse_args()

    if not args.all_langs and args.lang is None:
        raise SystemExit("Pass --lang <lang> or --all-langs")
    if args.all_langs and args.lang is not None:
        raise SystemExit("Use either --lang or --all-langs, not both")

    drop = _parse_ids(args)
    langs = sorted(FULL_VOICE_LANGS) if args.all_langs else [args.lang]

    print(f"Ids to drop: {sorted(drop)} ({len(drop)} total)", flush=True)

    for lang in langs:
        asr_p, tr_p, fin_p = _checkpoint_paths(lang)
        print(f"\n--- {lang} ---", flush=True)
        for path, label in [(asr_p, "asr"), (tr_p, "translations"), (fin_p, "pipeline_results")]:
            if not path.exists():
                print(f"  skip {label}: not found {path}", flush=True)
                continue
            df = pd.read_csv(path)
            if "id" not in df.columns:
                print(f"  skip {label}: no id column in {path}", flush=True)
                continue
            before = len(df)
            kept = df[~df["id"].isin(drop)].copy()
            removed = before - len(kept)
            if removed == 0:
                ids_series = pd.to_numeric(df["id"], errors="coerce").dropna()
                if len(ids_series):
                    lo, hi = int(ids_series.min()), int(ids_series.max())
                    print(
                        f"  {label}: 0 rows removed - ids {sorted(drop)} not in file "
                        f"(n={before}, id range {lo}-{hi}) ({path.name})",
                        flush=True,
                    )
                else:
                    print(
                        f"  {label}: 0 rows removed - no numeric ids in file ({path.name})",
                        flush=True,
                    )
                continue
            print(f"  {label}: removed {removed} row(s), {before} -> {len(kept)} ({path.name})", flush=True)
            if not args.dry_run:
                kept.to_csv(path, index=False)

    if args.dry_run:
        print("\nDry-run: no files written.", flush=True)
    else:
        print("\nDone. Re-run: python run_pipeline_batch.py --lang <lang>", flush=True)


if __name__ == "__main__":
    main()
