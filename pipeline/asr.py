import os
import time
import requests
from dotenv import load_dotenv

from pipeline.khaya_client import next_key

load_dotenv()

ASR_BASE = "https://translation-api.ghananlp.org/asr"
# v3: next-gen ASR (punctuation, word timing, long-form). v1: legacy benchmark-compatible.
_ASR_VER = (os.getenv("KHAYA_ASR_VERSION") or "v3").strip().lower()
if _ASR_VER not in ("v1", "v3"):
    _ASR_VER = "v3"
ASR_URL = f"{ASR_BASE}/{_ASR_VER}/transcribe"
ASR_TIMEOUT = int(os.getenv("KHAYA_ASR_TIMEOUT") or "120")


class KhayaQuotaExceededError(RuntimeError):
    """Khaya API returned HTTP 403 (quota exceeded or subscription limit)."""


def _transcript_from_json(data) -> str:
    """Normalize v1/v3 JSON into a single transcript string."""
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, dict):
        return ""
    for key in ("transcript", "text", "full_text", "full_transcript"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    words = data.get("words") or data.get("segments")
    if isinstance(words, list) and words:
        parts: list[str] = []
        for w in words:
            if isinstance(w, str):
                parts.append(w)
            elif isinstance(w, dict):
                t = w.get("word") or w.get("text") or w.get("word_text")
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return " ".join(parts).strip()
    nested = data.get("result") or data.get("data")
    if isinstance(nested, dict):
        return _transcript_from_json(nested)
    return ""


def transcribe(wav_path: str, language: str = "tw") -> str:
    """Transcribe ``.wav`` via Khaya ASR (default API version from ``KHAYA_ASR_VERSION``).

    Request shape matches Khaya OpenAPI: raw WAV bytes, ``language`` query param,
    ``Content-Type: audio/wav``, ``Ocp-Apim-Subscription-Key`` header.

    Set ``KHAYA_ASR_VERSION=v1`` to force the legacy v1 endpoint for comparable WER runs.

    Uses ``KHAYA_API_KEY`` and ``KHAYA_API_KEY_2`` (if set) in rotation via ``next_key()``.
    """
    max_attempts = 6
    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        for attempt in range(max_attempts):
            headers = {
                "Content-Type": "audio/wav",
                "Ocp-Apim-Subscription-Key": next_key(),
            }
            resp = requests.post(
                ASR_URL,
                headers=headers,
                params={"language": language},
                data=audio_bytes,
                timeout=ASR_TIMEOUT,
            )
            if resp.status_code == 403:
                snippet = (resp.text or "")[:300].replace("\n", " ")
                raise KhayaQuotaExceededError(
                    "Khaya ASR returned HTTP 403 (quota exceeded or subscription issue). "
                    f"Check ghananlp.org plan / billing period. Response: {snippet!r}"
                )
            if resp.status_code == 429:
                wait = 20 + min(120, attempt * 25)
                print(f"[asr] rate limited (429) — sleeping {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = 15 + min(90, attempt * 20)
                print(f"[asr] HTTP {resp.status_code} — sleeping {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return _transcript_from_json(data)
    except KhayaQuotaExceededError:
        raise
    except Exception as e:
        print(f"[asr] ERROR on {wav_path}: {e}")
        return ""
    print(f"[asr] ERROR on {wav_path}: max retries exceeded")
    return ""
