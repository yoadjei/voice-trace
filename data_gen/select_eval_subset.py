# builds data/synthetic/evaluation_subset_ids.txt — stratified sample of narrative ids for khaya budget.
# default n=80 from narratives_en.csv by injury_type. run: python -m data_gen.select_eval_subset
# re-run with different n: python -m data_gen.select_eval_subset --n 100
import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

NARRATIVES_EN = Path("data/synthetic/narratives_en.csv")
OUTPUT_PATH = Path("data/synthetic/evaluation_subset_ids.txt")


def select_ids(n: int, seed: int = 42) -> list[int]:
    df = pd.read_csv(NARRATIVES_EN)
    if n >= len(df):
        return sorted(df["id"].astype(int).tolist())

    # stratify on injury_type (fallback if too few per class)
    strat = df["injury_type"] if df["injury_type"].nunique() > 1 else None
    try:
        sample, _ = train_test_split(
            df,
            train_size=n,
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        # rare: class too small for stratify
        rng = random.Random(seed)
        idx = rng.sample(range(len(df)), n)
        sample = df.iloc[sorted(idx)]

    return sorted(sample["id"].astype(int).tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80, help="number of ids (default 80)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="print counts only")
    args = p.parse_args()

    if not NARRATIVES_EN.exists():
        raise SystemExit(f"missing {NARRATIVES_EN}")

    ids = select_ids(args.n, args.seed)
    df = pd.read_csv(NARRATIVES_EN)
    sub = df[df["id"].isin(ids)]

    print(f"selected n={len(ids)} ids (requested {args.n})")
    print("injury_type counts in subset:")
    print(sub["injury_type"].value_counts().sort_index())

    if args.dry_run:
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# stratified subset from {NARRATIVES_EN.name} (n={len(ids)}, seed={args.seed})\n"
        f"# used by roundtrip, tts, run_pipeline_batch, run_extraction_all_langs when present\n"
    )
    OUTPUT_PATH.write_text(header + "\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
