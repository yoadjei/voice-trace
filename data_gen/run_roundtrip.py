# back-translates local-language narratives to english via khaya api.
# reads:  data/synthetic/narratives_{lang}.csv for each language
# writes: data/synthetic/roundtrip_{lang}_en.csv with back-translated english
# run single lang: python -m data_gen.run_roundtrip --lang twi
# run all:         python -m data_gen.run_roundtrip
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pipeline.eval_subset import get_eval_id_set
from pipeline.translate import khaya_translate_response_text
from pipeline.khaya_client import KHAYA_RATE_SLEEP, key_count, next_key
from pipeline.lang_config import EVAL_LANGUAGES_FIVE

load_dotenv()

TRANSLATE_URL = "https://translation-api.ghananlp.org/v1/translate"

LANGUAGES = list(EVAL_LANGUAGES_FIVE)
DATA_DIR = Path("data/synthetic")

# khaya source language codes for back-translation to english
ROUNDTRIP_CODES = {
    "twi": "tw",
    "ga": "gaa",
    "ewe": "ee",
    "fante": "fat",
    "dagbani": "dag",
    "gurene": "gur",
}


def translate_via_khaya(text: str, source_lang: str, target_lang: str = "en") -> str:
    # call khaya translate api. returns translated text or raises on fatal error.
    payload = {
        "text": text,
        "source_language": source_lang,
        "target_language": target_lang,
    }

    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            headers = {
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": next_key(),
            }
            resp = requests.post(
                TRANSLATE_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            parsed = khaya_translate_response_text(result) if isinstance(result, dict) else None
            return parsed if parsed is not None else text.strip()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 403:
                body = e.response.text[:300] if e.response is not None else ""
                raise Exception(f"quota exhausted: {body}") from e
            if status == 429:
                wait = 30 * (attempt + 1)
                print(f"    rate limited — sleeping {wait}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                continue
            # 5xx = Khaya server fault / overload — common and usually transient; retry with backoff
            if status in (500, 502, 503, 504):
                wait = 15 + min(90, attempt * 20)
                print(f"    HTTP {status} (server) — sleeping {wait}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                continue
            raise Exception(f"HTTP {status}: {e}") from e
        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "resolve" in err or "timeout" in err:
                print(f"    connection error — retrying in 15s (attempt {attempt+1}/{max_attempts})")
                time.sleep(15)
            else:
                raise

    raise Exception("translate: max retries exceeded")


def backTranslate_lang(lang: str) -> None:
    # read local-language narratives, back-translate to english, save output.
    input_path = DATA_DIR / f"narratives_{lang}.csv"
    if not input_path.exists():
        print(f"[{lang}] no csv found at {input_path} — skipping")
        return

    khaya_code = ROUNDTRIP_CODES.get(lang)
    if not khaya_code:
        print(f"[{lang}] no khaya lang code — skipping")
        return

    output_path = DATA_DIR / f"roundtrip_{lang}_en.csv"

    # load input
    df = pd.read_csv(input_path)
    col = f"narrative_{lang}"
    if col not in df.columns:
        print(f"[{lang}] column '{col}' not found in csv — skipping")
        return

    subset = get_eval_id_set()
    if subset is not None:
        df = df[df["id"].isin(subset)].copy().sort_values("id")
        print(f"[{lang}] evaluation subset: {len(df)} ids (data/synthetic/evaluation_subset_ids.txt)")

    # load existing output if present
    existing = {}
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if subset is not None:
            existing_df = existing_df[existing_df["id"].isin(subset)]
        existing = {int(k): v for k, v in zip(existing_df["id"], existing_df["roundtrip_en"])}

    total = len(df)
    print(f"\n--- {lang.upper()} ({total} rows) ---")

    output_data = []
    for j, (_, row) in enumerate(df.iterrows(), 1):
        row_id = int(row["id"])

        # skip if already done
        if row_id in existing:
            print(f"  [{j}/{total}] id={row_id} SKIP — exists")
            output_data.append({"id": row_id, "roundtrip_en": existing[row_id]})
            continue

        text = row.get(col, "")
        if not isinstance(text, str) or not text.strip():
            print(f"  [{j}/{total}] id={row_id} SKIP — empty")
            output_data.append({"id": row_id, "roundtrip_en": ""})
            continue

        try:
            backTranslated = translate_via_khaya(text, khaya_code, "en")
            output_data.append({"id": row_id, "roundtrip_en": backTranslated})
            print(f"  [{j}/{total}] id={row_id} saved")
        except Exception as e:
            err_msg = str(e).lower()
            if "quota" in err_msg:
                print(f"  [{j}/{total}] id={row_id} ERROR: quota exhausted — stopping")
                # save progress before stopping
                pd.DataFrame(output_data).to_csv(output_path, index=False)
                return
            else:
                print(f"  [{j}/{total}] id={row_id} ERROR: {e} — skipping row")
                output_data.append({"id": row_id, "roundtrip_en": ""})

        # save after each row
        pd.DataFrame(output_data).to_csv(output_path, index=False)
        # ~10 req/min per key; with 2 keys alternating, ~3s keeps combined rate sane
        time.sleep(max(3.0, KHAYA_RATE_SLEEP / max(1, key_count())))

    # must write after SKIPs too — last rows can be all SKIPs with no translate/save in-loop
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    print(f"done. {len(output_data)}/{total} rows in {output_path}")


def backTranslate_all() -> None:
    for lang in LANGUAGES:
        backTranslate_lang(lang)


if __name__ == "__main__":
    args = sys.argv[1:]
    lang_flag = next(
        (args[i + 1] for i, a in enumerate(args) if a == "--lang" and i + 1 < len(args)),
        None,
    )
    if lang_flag:
        backTranslate_lang(lang_flag)
    else:
        backTranslate_all()
