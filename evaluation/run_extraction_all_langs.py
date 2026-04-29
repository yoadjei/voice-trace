# runs claude extraction on round-trip back-translated english for all 6 languages.
# produces data/synthetic/extraction_{lang}.csv per language.
# these files are consumed by evaluate_translation.py and evaluate_consistency.py.
# run single lang: python -m evaluation.run_extraction_all_langs --lang twi
# run all:         python -m evaluation.run_extraction_all_langs
import sys
import time
import pandas as pd
from pathlib import Path
from pipeline.eval_subset import get_eval_id_set
from pipeline.extract import extract
from pipeline.lang_config import EVAL_LANGUAGES_FIVE

SYNTHETIC_DIR = Path("data/synthetic")
LANGUAGES = list(EVAL_LANGUAGES_FIVE)
EVAL_FIELDS = ["injury_type", "mechanism", "severity", "body_region", "victim_sex", "victim_age_group", "location_description"]


def _load_done_ids(path: Path) -> set:
    if path.exists():
        try:
            return set(pd.read_csv(path)["id"].tolist())
        except Exception:
            pass
    return set()


def run_for_lang(lang: str) -> None:
    input_path = SYNTHETIC_DIR / f"roundtrip_{lang}_en.csv"
    output_path = SYNTHETIC_DIR / f"extraction_{lang}.csv"

    if not input_path.exists():
        print(f"[{lang}] roundtrip file not found: {input_path} — skipping")
        return

    df = pd.read_csv(input_path)
    subset = get_eval_id_set()
    if subset is not None:
        df = df[df["id"].isin(subset)].copy().sort_values("id")
        print(f"[{lang}] evaluation subset: {len(df)} rows")
    done = _load_done_ids(output_path)
    total = len(df)
    new_rows = []

    print(f"\n--- {lang.upper()} ({total} rows) ---")

    for i, row in df.iterrows():
        row_id = int(row["id"])
        if row_id in done:
            print(f"  [{i+1}/{total}] SKIP — exists")
            continue

        text = row.get("roundtrip_en", "")
        if not isinstance(text, str) or not text.strip():
            print(f"  [{i+1}/{total}] SKIP — empty")
            result = {"extraction": {k: "unknown" for k in EVAL_FIELDS}, "first_aid": ""}
        else:
            result = extract(text)

        extraction = result["extraction"]
        new_rows.append({"id": row_id, **extraction})
        print(f"  [{i+1}/{total}] extracted id={row_id}")

        # save after each row
        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.drop_duplicates(subset=["id"], keep="last").to_csv(output_path, index=False)
        else:
            pd.DataFrame(new_rows).to_csv(output_path, index=False)

        new_rows = []  # already flushed
        time.sleep(1)  # light throttle — claude api

    saved = len(pd.read_csv(output_path)) if output_path.exists() else 0
    print(f"done. {saved}/{total} rows in {output_path}")


def run_all() -> None:
    for lang in LANGUAGES:
        run_for_lang(lang)


if __name__ == "__main__":
    args = sys.argv[1:]
    lang_flag = next(
        (args[i + 1] for i, a in enumerate(args) if a == "--lang" and i + 1 < len(args)),
        None,
    )
    if lang_flag:
        run_for_lang(lang_flag)
    else:
        run_all()
