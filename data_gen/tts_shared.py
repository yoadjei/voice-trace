# standalone tts script for ewe, fante, dagbani, gurene
# generates .wav audio from ghanaian language narratives via khaya tts v2.
#
# setup:
#   pip install requests pandas scipy numpy python-dotenv
#
# put your khaya api key in a .env file next to this script:
#   KHAYA_API_KEY=your_key_here
#
# folder structure expected:
#   data/synthetic/narratives_ewe.csv
#   data/synthetic/narratives_fante.csv
#   data/synthetic/narratives_dagbani.csv
#   data/synthetic/narratives_gurene.csv
#
# output goes to:
#   data/synthetic/audio/{lang}/narrative_{id:03d}.wav
#
# run single lang:  python tts_shared.py --lang ewe
# run all four:     python tts_shared.py

import io
import os
import re
import sys
import time
import numpy as np
import requests
import pandas as pd
import scipy.io.wavfile as wavfile
from pathlib import Path
from dotenv import load_dotenv

# allow running as script: add repo root for pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.eval_subset import get_eval_id_set

load_dotenv()

KHAYA_KEY = os.getenv("KHAYA_API_KEY")
TTS_URL = "https://translation-api.ghananlp.org/tts/v2/synthesize"
HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": KHAYA_KEY,
}

# khaya tts v2 language codes
LANG_CODES = {
    "ewe": "ewe",
    "fante": "fat",
    "dagbani": "dag",
    "gurene": "gur",
}

LANGUAGES = ["ewe", "fante", "dagbani", "gurene"]
DATA_DIR = Path("data/synthetic")
AUDIO_DIR = Path("data/synthetic/audio")
MAX_CHARS = 450


def _normalize_for_tts(text: str) -> str:
    # collapse repeated diacritic vowels that cause tts looping (ɛɛ→ɛ, ɔɔ→ɔ)
    # source csv is not modified — only applied at synthesis time
    text = re.sub(r'ɛɛ+', 'ɛ', text)
    text = re.sub(r'ɔɔ+', 'ɔ', text)
    text = re.sub(r'(.)\1{2,}', lambda m: m.group(1) * 2, text)
    return text


def _chunk_text(text: str) -> list[str]:
    # split at sentence boundaries to stay under MAX_CHARS per request
    if len(text) <= MAX_CHARS:
        return [text]
    sentences = [s.strip() for s in text.replace("— ", ". ").split(". ") if s.strip()]
    chunks, current = [], ""
    for s in sentences:
        candidate = f"{current}. {s}" if current else s
        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks or [text[:MAX_CHARS]]


def _concat_wavs(wav_bytes_list: list[bytes]) -> bytes:
    # merge multiple wav byte strings into one — handles float32 wav (format 3)
    arrays, rate = [], None
    for data in wav_bytes_list:
        r, samples = wavfile.read(io.BytesIO(data))
        rate = r
        arrays.append(samples)
    combined = np.concatenate(arrays)
    out = io.BytesIO()
    wavfile.write(out, rate, combined)
    return out.getvalue()


def generate_audio_for_lang(lang: str) -> None:
    csv_path = DATA_DIR / f"narratives_{lang}.csv"
    if not csv_path.exists():
        print(f"[{lang}] no csv found at {csv_path} — skipping")
        return

    lang_code = LANG_CODES.get(lang)
    if not lang_code:
        print(f"[{lang}] unknown language — skipping")
        return

    if not KHAYA_KEY:
        print("ERROR: KHAYA_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    out_dir = AUDIO_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    col = f"narrative_{lang}"
    if col not in df.columns:
        print(f"[{lang}] column '{col}' not found in csv — skipping")
        return

    subset = get_eval_id_set()
    if subset is not None:
        df = df[df["id"].isin(subset)].copy().sort_values("id")
        print(f"[{lang}] evaluation subset: {len(df)} wav targets")

    total = len(df)
    print(f"\n--- {lang.upper()} ({total} rows) ---")

    for i, row in df.iterrows():
        text = row.get(col, "")
        if not isinstance(text, str) or not text.strip():
            print(f"  [{i+1}/{total}] SKIP — empty")
            continue

        wav_path = out_dir / f"narrative_{int(row['id']):03d}.wav"
        if wav_path.exists():
            print(f"  [{i+1}/{total}] SKIP — exists")
            continue

        chunks = _chunk_text(_normalize_for_tts(text))
        chunk_wavs = []
        failed = False

        for chunk in chunks:
            for attempt in range(3):
                try:
                    resp = requests.post(
                        TTS_URL,
                        headers=HEADERS,
                        json={"text": chunk, "language": lang_code, "format": "wav"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    chunk_wavs.append(resp.content)
                    break
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "?"
                    if status == 403:
                        print(f"  [{i+1}/{total}] QUOTA EXCEEDED — stopping")
                        print("  swap your KHAYA_API_KEY and rerun — existing files will be skipped")
                        return
                    print(f"  [{i+1}/{total}] HTTP {status} — skipping row: {e}")
                    failed = True
                    break
                except Exception as e:
                    err = str(e).lower()
                    if "connection" in err or "resolve" in err or "timeout" in err:
                        print(f"  [{i+1}/{total}] connection error — retrying in 10s (attempt {attempt+1}/3)")
                        time.sleep(10)
                    else:
                        print(f"  [{i+1}/{total}] ERROR: {e}")
                        failed = True
                        break
            if failed:
                break

        if not failed and chunk_wavs:
            audio = _concat_wavs(chunk_wavs) if len(chunk_wavs) > 1 else chunk_wavs[0]
            wav_path.write_bytes(audio)
            print(f"  [{i+1}/{total}] saved {wav_path.name}" + (f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""))

        time.sleep(0.5)

    saved = sum(1 for f in out_dir.glob("*.wav") if f.is_file())
    print(f"done. {saved}/{total} files in {out_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    lang_flag = next(
        (args[i + 1] for i, a in enumerate(args) if a == "--lang" and i + 1 < len(args)),
        None,
    )
    if lang_flag:
        if lang_flag not in LANG_CODES:
            print(f"unknown lang '{lang_flag}'. choose from: {', '.join(LANG_CODES)}")
            sys.exit(1)
        generate_audio_for_lang(lang_flag)
    else:
        for lang in LANGUAGES:
            generate_audio_for_lang(lang)
