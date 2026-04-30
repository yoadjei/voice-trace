# translates english narratives into all 5 ghanaian languages via claude sonnet.
# reads:  data/synthetic/narratives_en.csv
# writes: data/synthetic/narratives_all_langs.csv
#         data/synthetic/narratives_{lang}.csv  (one per language, id + narrative only)
# run single lang:  python -m data_gen.translate_all_langs --lang ewe
# run all:          python -m data_gen.translate_all_langs
# run fresh:        python -m data_gen.translate_all_langs --fresh
# run lang fresh:   python -m data_gen.translate_all_langs --lang ewe --fresh
import sys
import time
import pandas as pd
from pathlib import Path
from pipeline.translate import translate_en_to_lang

INPUT_PATH = Path("data/synthetic/narratives_en.csv")
OUTPUT_PATH = Path("data/synthetic/narratives_all_langs.csv")

LANGUAGES = ["ewe", "ga", "dagbani", "fante", "gurene"]


def _lang_path(lang: str) -> Path:
    return INPUT_PATH.parent / f"narratives_{lang}.csv"


def _load_lang_csv(lang: str) -> pd.Series:
    # load existing per-language translations as a Series indexed by id
    p = _lang_path(lang)
    if not p.exists():
        return pd.Series(dtype=str)
    df = pd.read_csv(p)
    col = f"narrative_{lang}"
    if col not in df.columns:
        return pd.Series(dtype=str)
    return df.set_index("id")[col]


def translate_lang(lang: str, fresh: bool = False) -> pd.DataFrame:
    col = f"narrative_{lang}"
    lang_path = _lang_path(lang)

    if fresh and lang_path.exists():
        lang_path.unlink()
        print(f"fresh run — deleted {lang_path}")

    en_df = pd.read_csv(INPUT_PATH)
    existing = _load_lang_csv(lang)

    # map existing translations back onto en_df
    en_df[col] = en_df["id"].map(existing).astype(object)

    done = en_df[col].notna() & (en_df[col] != "")
    remaining = (~done).sum()
    print(f"\n--- {lang.upper()} ({done.sum()} done, {remaining} remaining) ---")

    for i, row in en_df.iterrows():
        if pd.notna(en_df.at[i, col]) and en_df.at[i, col] != "":
            continue

        en_text = row["narrative_en"]
        if not isinstance(en_text, str) or not en_text.strip():
            en_df.at[i, col] = ""
            continue

        translation = translate_en_to_lang(en_text, lang=lang)

        # reject english fallbacks — leave null so they get retried next run
        if translation.strip() == en_text.strip():
            print(f"  [{i+1}/{len(en_df)}] {lang} id={row['id']} | FALLBACK — skipping")
            continue

        en_df.at[i, col] = translation

        # save per-language csv after every row — id + translation only
        en_df[["id", col]].to_csv(lang_path, index=False)

        print(f"  [{i+1}/{len(en_df)}] {lang} id={row['id']} | done")
        time.sleep(1)

    en_df[["id", col]].to_csv(lang_path, index=False)
    empty = en_df[col].isna().sum() + (en_df[col] == "").sum()
    print(f"done. {len(en_df) - empty}/{len(en_df)} translated — saved to {lang_path}")
    return en_df


def translate_all_langs(fresh: bool = False, langs: list[str] | None = None) -> None:
    targets = langs or LANGUAGES
    for lang in targets:
        translate_lang(lang, fresh=fresh)

    # merge all per-language files into narratives_all_langs.csv
    base = pd.read_csv(INPUT_PATH)
    for lang in LANGUAGES:
        col = f"narrative_{lang}"
        existing = _load_lang_csv(lang)
        base[col] = base["id"].map(existing)
    base.to_csv(OUTPUT_PATH, index=False)
    print(f"\nall done — merged to {OUTPUT_PATH}")


if __name__ == "__main__":
    args = sys.argv[1:]
    fresh = "--fresh" in args
    lang_flag = next((args[i + 1] for i, a in enumerate(args) if a == "--lang" and i + 1 < len(args)), None)
    if lang_flag:
        translate_lang(lang_flag, fresh=fresh)
    else:
        translate_all_langs(fresh=fresh)
