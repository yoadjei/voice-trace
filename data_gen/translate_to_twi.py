# translates english narratives to twi via haiku (draft) + sonnet (review).
# reads:  data/synthetic/narratives_en.csv
# writes: data/synthetic/narratives_twi.csv
# run:         python -m data_gen.translate_to_twi
# run fresh:   python -m data_gen.translate_to_twi --fresh
import sys
import time
import pandas as pd
from pathlib import Path
from pipeline.translate import translate_en_to_twi

INPUT_PATH = Path("data/synthetic/narratives_en.csv")
OUTPUT_PATH = Path("data/synthetic/narratives_twi.csv")


def translate_to_twi(fresh: bool = False) -> pd.DataFrame:
    # resumable — skips rows where narrative_twi already populated
    # fresh=True deletes any existing output and starts over
    if fresh and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print("fresh run — deleted existing output")

    df = pd.read_csv(INPUT_PATH)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(
            existing.loc[existing["narrative_twi"].notna() & (existing["narrative_twi"] != ""), "id"]
        )
        twi_map = dict(zip(existing["id"], existing["narrative_twi"]))
        print(f"resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining")
    else:
        done_ids = set()
        twi_map = {}

    for i, row in df.iterrows():
        row_id = row["id"]
        if row_id in done_ids:
            continue

        en_text = row["narrative_en"]
        if not isinstance(en_text, str) or not en_text.strip():
            twi_map[row_id] = ""
            continue

        twi = translate_en_to_twi(en_text)
        twi_map[row_id] = twi

        # save after each row so progress is never lost
        df["narrative_twi"] = df["id"].map(twi_map)
        df.to_csv(OUTPUT_PATH, index=False)

        print(f"[{i+1}/{len(df)}] id={row_id} | done")
        time.sleep(1)  # small buffer between claude calls

    df["narrative_twi"] = df["id"].map(twi_map)
    df.to_csv(OUTPUT_PATH, index=False)
    empty = df["narrative_twi"].isna().sum() + (df["narrative_twi"] == "").sum()
    print(f"\ndone. {len(df) - empty}/{len(df)} translated — saved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    translate_to_twi(fresh="--fresh" in sys.argv)
