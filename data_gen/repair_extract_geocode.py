# Re-run Claude extract + geocode for pipeline rows that already have translated_en
# (e.g. after API credit errors left rows with no first_aid). Does not call ASR or translate.
#
#   python -m data_gen.repair_extract_geocode --lang twi
#       → end-to-end: all rows that match the failed-extract heuristic (no --ids needed).
#   python -m data_gen.repair_extract_geocode --lang twi --ids 107,108
#   python -m data_gen.repair_extract_geocode --lang twi --failed-only --min-id 107
#       → optional: restrict to ids >= N or explicit ids
import argparse
import sys
import time

import pandas as pd
from pathlib import Path

from pipeline.extract import extract
from pipeline.geocode import geocode

SYN = Path("data/synthetic")

# pipeline extract() failure (e.g. billing) leaves all schema unknown + empty first_aid
_EX_KEYS = [
    "injury_type",
    "mechanism",
    "severity",
    "body_region",
    "victim_sex",
    "victim_age_group",
    "location_description",
]


def _row_failed_extract(row: pd.Series) -> bool:
    te = str(row.get("translated_en") or "")
    if len(te.strip()) < 30:
        return False
    fa = row.get("first_aid")
    if fa is None or (isinstance(fa, float) and pd.isna(fa)):
        fa = ""
    if str(fa).strip():
        return False  # successful run produced first aid (even if injury_type is legitimately unknown)
    return all(str(row.get(k) or "") == "unknown" for k in _EX_KEYS)


def _paths(lang: str) -> tuple[Path, Path]:
    if lang == "twi":
        return SYN / "translations_en.csv", SYN / "pipeline_results.csv"
    return SYN / f"translations_en_{lang}.csv", SYN / f"pipeline_results_{lang}.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Re-extract + geocode from existing translations.")
    p.add_argument("--lang", default="twi", help="twi (legacy CSV names) or ga, ewe, …")
    p.add_argument("--ids", default="", help="Comma-separated narrative ids (e.g. 107,108)")
    p.add_argument(
        "--failed-only",
        action="store_true",
        help="Rows that look like failed extract: empty first_aid + all extraction fields unknown",
    )
    p.add_argument(
        "--min-id",
        type=int,
        default=None,
        metavar="N",
        help="With --failed-only: only ids >= N (e.g. 107 after a billing error on the last batch)",
    )
    p.add_argument(
        "--unknown-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args()
    if args.unknown_only:
        print("[repair] --unknown-only is deprecated; use --failed-only (stricter).", flush=True)
        args.failed_only = True
    if not args.ids.strip() and not args.failed_only:
        args.failed_only = True

    trans_path, final_path = _paths(args.lang)
    if not trans_path.exists() or not final_path.exists():
        raise SystemExit(f"missing {trans_path} or {final_path}")

    trans = pd.read_csv(trans_path).set_index("id")
    df = pd.read_csv(final_path)

    if args.ids.strip():
        want = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
    elif args.failed_only:
        want = {int(row["id"]) for _, row in df.iterrows() if _row_failed_extract(row)}
    else:
        raise SystemExit("Pass --ids ... or run without --ids for full failed-row repair.")

    if args.min_id is not None:
        want = {i for i in want if i >= args.min_id}

    if not want:
        print("No ids to repair.")
        return

    want_sorted = sorted(want)
    print(f"Repairing {len(want)} id(s) end-to-end (ascending id): {want_sorted}", flush=True)

    for n, tid in enumerate(want_sorted, start=1):
        idxs = df.index[df["id"] == tid].tolist()
        if not idxs:
            continue
        i = idxs[0]
        row = df.loc[i]
        if tid not in trans.index:
            print(f"  skip id={tid} — no translations row")
            continue
        en = str(trans.loc[tid, "translated_en"])
        if not en.strip():
            print(f"  skip id={tid} — empty translated_en")
            continue
        print(f"  id={tid} ({n}/{len(want)}) …", flush=True)
        result = extract(en)
        if result.get("fatal_billing"):
            print(
                "STOP: Anthropic billing/credits error. Add credits, then re-run (progress saved).",
                flush=True,
            )
            sys.exit(1)
        extraction = result["extraction"]
        first_aid = result["first_aid"]
        loc = extraction.get("location_description") or "unknown"
        geo = geocode(loc)
        for k, v in extraction.items():
            df.at[i, k] = v
        df.at[i, "first_aid"] = first_aid
        df.at[i, "lat"] = geo.get("lat")
        df.at[i, "lng"] = geo.get("lng")
        df.at[i, "facility_name"] = geo.get("facility_name")
        df.at[i, "facility_dist_km"] = geo.get("facility_dist_km")
        df.to_csv(final_path, index=False)
        time.sleep(0.5)

    print(f"Done. Wrote {final_path}")


if __name__ == "__main__":
    main()
