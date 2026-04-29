# first-pass auto-annotation of 126 english narratives using claude.
# produces data/synthetic/gold_annotations_pass1.csv for human review.
# the human second pass edits gold_annotations_pass1.csv, then saves as gold_annotations.csv.
# run: python -m data_gen.annotate_gold
# re-run failed ids only: python -m data_gen.annotate_gold --ids 98,108
import argparse
import time
import pandas as pd
from pathlib import Path
from pipeline.extract import extract

NARRATIVES_PATH = Path("data/synthetic/narratives_en.csv")
PASS1_OUTPUT = Path("data/synthetic/gold_annotations_pass1.csv")

# org limit was 5 req/min for sonnet-4-6 — stay safely under (~4.5/min)
DELAY_SEC = 13.0

EVAL_FIELDS = ["injury_type", "mechanism", "severity", "body_region", "victim_sex", "victim_age_group", "location_description"]


def _load_done_ids(path: Path) -> set:
    if path.exists():
        try:
            return set(pd.read_csv(path)["id"].tolist())
        except Exception:
            pass
    return set()


def _merge_pass1(rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if PASS1_OUTPUT.exists():
        existing = pd.read_csv(PASS1_OUTPUT)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["id"], keep="last").to_csv(PASS1_OUTPUT, index=False)
    else:
        new_df.to_csv(PASS1_OUTPUT, index=False)


def run(only_ids: set[int] | None = None) -> None:
    df = pd.read_csv(NARRATIVES_PATH)
    if only_ids is not None:
        df = df[df["id"].isin(only_ids)].sort_values("id")
        if len(df) != len(only_ids):
            missing = only_ids - set(df["id"].tolist())
            print(f"warning: narrative ids not in csv: {sorted(missing)}")
    total = len(df)
    done = _load_done_ids(PASS1_OUTPUT) if only_ids is None else set()

    print(f"annotating {total} narratives (pass 1 via claude)...")
    if only_ids is not None:
        print(f"  (reannotate mode: ids {sorted(only_ids)})")

    for j, (_, row) in enumerate(df.iterrows(), 1):
        row_id = int(row["id"])
        if only_ids is None and row_id in done:
            print(f"  [{j}/{total}] SKIP — exists")
            continue

        text = row.get("narrative_en", "")
        if not isinstance(text, str) or not text.strip():
            print(f"  [{j}/{total}] SKIP — empty narrative")
            result = {"extraction": {k: "unknown" for k in EVAL_FIELDS}}
        else:
            result = extract(text)

        extraction = result["extraction"]
        row_out = {
            "id": row_id,
            "narrative_en": text,
            **{f: extraction.get(f, "unknown") for f in EVAL_FIELDS},
        }
        _merge_pass1([row_out])
        print(f"  [{j}/{total}] annotated id={row_id}: {extraction.get('injury_type')} / {extraction.get('severity')}")
        time.sleep(DELAY_SEC)

    saved = len(pd.read_csv(PASS1_OUTPUT)) if PASS1_OUTPUT.exists() else 0
    print(f"\ndone. {saved} rows in {PASS1_OUTPUT}")
    print(f"\nnext: review {PASS1_OUTPUT}, correct any errors, then copy to data/synthetic/gold_annotations.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Gold annotation pass 1 via Claude extract()")
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help="comma-separated narrative ids to (re)annotate, e.g. 98,108",
    )
    args = p.parse_args()
    only = None
    if args.ids.strip():
        only = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
    run(only_ids=only)


if __name__ == "__main__":
    main()
