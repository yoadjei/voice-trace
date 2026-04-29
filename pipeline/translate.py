import os
import re
import time
import requests
import anthropic
from dotenv import load_dotenv
from pipeline.lang_prompts import LANG_PROMPTS
from pipeline.khaya_client import next_key

load_dotenv()

_KHAYA_URL = "https://translation-api.ghananlp.org/v1/translate"

# ASR→English for run_pipeline_batch: khaya (default) | khaya_claude (polish) | claude (Claude only, no Khaya MT)
# Set in .env: VOICETRACE_TRANSLATE_MODE=khaya_claude
_PIPELINE_LANG_LABELS = {
    "twi": "Twi (Akan)",
    "ga": "Ga",
    "ewe": "Ewe",
    "dagbani": "Dagbani",
}


def khaya_translate_response_text(response_json: dict) -> str | None:
    # GhanaNLP /v1/translate JSON shape varies: often {"text": "<translated>", "source_language":..., "target_language":...}
    # Older clients used translated_text / translatedText.
    if not isinstance(response_json, dict):
        return None
    for key in ("translated_text", "translatedText", "text", "out", "translation"):
        v = response_json.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def khaya_translate(text: str, source_lang: str, target_lang: str, retries: int = 3) -> str:
    # translate text via khaya api (handles both directions, e.g. tw→en or en→tw).
    # source_lang/target_lang use khaya codes: tw, gaa, ee, fat, dag, gur, en.
    # alternates KHAYA_API_KEY / KHAYA_API_KEY_2 per request via next_key().
    payload = {"text": text, "source_language": source_lang, "target_language": target_lang}

    for attempt in range(retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": next_key(),
            }
            resp = requests.post(_KHAYA_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            out = khaya_translate_response_text(data) if isinstance(data, dict) else None
            if out is None:
                return text
            return out
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 403:
                print(f"[khaya_translate] quota exhausted")
                return text
            if status == 429:
                wait = 30 * (attempt + 1)
                print(f"[khaya_translate] rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"[khaya_translate] HTTP {status}: {e}")
            break
        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "timeout" in err:
                print(f"[khaya_translate] connection error — retrying in 10s")
                time.sleep(10)
            else:
                print(f"[khaya_translate] ERROR: {e}")
                break

    return text  # fallback: return original on failure


_claude_client = None


def _get_claude():
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _claude_client


def _parse_retry_delay(err: Exception, default: float = 60.0) -> float:
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(err))
    return float(match.group(1)) + 2 if match else default


def _translate_mode() -> str:
    m = os.getenv("VOICETRACE_TRANSLATE_MODE", "khaya").strip().lower()
    if m in ("khaya", "khaya_claude", "claude"):
        return m
    return "khaya"


def _claude_asr_to_english_retry(asr_text: str, pipeline_lang: str, retries: int = 4) -> str:
    label = _PIPELINE_LANG_LABELS.get(pipeline_lang, pipeline_lang)
    system = (
        "You translate emergency phone reports from Ghanaian languages into clear English for hospital triage. "
        "Preserve personal names, place names, and injury facts. Do not invent details. "
        "Output only the English paragraph, no preamble."
    )
    user = (
        f"The following is noisy automatic speech recognition text (may include misspellings). "
        f"It is meant to be {label}. Translate into clear English.\n\n{asr_text}"
    )
    for attempt in range(retries):
        try:
            client = _get_claude()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "overloaded" in err_str.lower() or "rate" in err_str.lower():
                wait = _parse_retry_delay(e, default=30.0)
                print(f"[translate:claude-asr] rate limited — sleeping {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"[translate:claude-asr] ERROR: {e}")
                break
    return asr_text


def _claude_polish_en_retry(rough_en: str, asr_text: str, pipeline_lang: str, retries: int = 4) -> str:
    label = _PIPELINE_LANG_LABELS.get(pipeline_lang, pipeline_lang)
    system = (
        "You rewrite rough machine translations into clear English for emergency medical triage. "
        "Preserve facts, names, and places. Fix grammar and meaning using the transcript as context. "
        "Do not invent injuries or locations. Output only the English paragraph."
    )
    user = (
        f"Source language: {label}\n\n"
        f"Rough English (machine translation, may be wrong):\n{rough_en}\n\n"
        f"Noisy ASR transcript (same utterance, for context):\n{asr_text}\n\n"
        "Rewrite one clear English paragraph."
    )
    for attempt in range(retries):
        try:
            client = _get_claude()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "overloaded" in err_str.lower() or "rate" in err_str.lower():
                wait = _parse_retry_delay(e, default=30.0)
                print(f"[translate:claude-polish] rate limited — sleeping {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"[translate:claude-polish] ERROR: {e}")
                break
    return rough_en


def translate_asr_to_english(asr_text: str, khaya_code: str, pipeline_lang: str) -> str:
    """
    ASR → English for the VoiceTrace batch pipeline.
    VOICETRACE_TRANSLATE_MODE:
      khaya — GhanaNLP only (default, lowest cost)
      khaya_claude — Khaya then Claude polish (recommended when MT is noisy)
      claude — Claude only from ASR (no Khaya translate call; higher Anthropic cost)
    """
    if not (asr_text or "").strip():
        return ""
    mode = _translate_mode()
    if mode == "claude":
        return _claude_asr_to_english_retry(asr_text, pipeline_lang)
    if mode == "khaya_claude":
        rough = khaya_translate(asr_text, khaya_code, "en")
        return _claude_polish_en_retry(rough, asr_text, pipeline_lang)
    return khaya_translate(asr_text, khaya_code, "en")


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Used by pipeline.pipeline.run_pipeline (single Twi .wav). Delegates to translate_asr_to_english."""
    if target_lang != "en":
        return khaya_translate(text, source_lang, target_lang)
    return translate_asr_to_english(text, source_lang, "twi")


_TWI_SYSTEM_PROMPT = """\
You are a native speaker of Ghanaian Twi (Asante Akan) with deep expertise in spoken, colloquial Twi \
as used in everyday Ghanaian life, especially in emergency and health contexts.

Key linguistic rules you must follow:
- "Please" → "Me pawoɔ kyɛw" (NEVER "Medaase" — that means "Thank you")
- "Thank you" → "Medaase"
- "Hello / Hello there" → "Agoo" or "Ɛte sɛn"
- "Help" / "Help me" → "Boa me" or "Mmoa me"
- "Accident" → "akwantu (or kar) accident" — keep the word "accident"
- "Hospital" → "Ɔyaresabea" or "hospital" (both acceptable)
- "Pain / it's painful" → "Ɛyɛ me yaw" or "ɛyɛ yaw"
- "He/she fell" → "Ɔtuu fam"
- "He/she is injured" → "Wɔabubu no" or "ɔwɔ mpaem"
- "Blood" → "Mogya"
- "Bone / fracture" → "Dompe (broken bone)" → "ne dompe atwa"
- "Unconscious" → "Ɔnhunu hwee" or "ɔanyɛ hunu"
- "Conscious / awake" → "Ɔhunu ne ho"
- "Child" → "Abofra"
- "Old person / elderly" → "Opanyin"
- "Please send an ambulance" → "Me pawoɔ kyɛw, soma ambulance mmra"
- Road: "ɔkwan" | Market: "dwam" | Vehicle: "kar" | Motorcycle: "motobaik" or "motor"
- Trotro, aboboyaa, pragyia — keep these as-is (Ghanaian vehicle terms)

Do NOT translate proper nouns:
- Personal names (Kwame, Ama, Kofi, Abena, Mensah, Akosua, etc.)
- Ghanaian place names (Kumasi, Accra, Tamale, Takoradi, Bolgatanga, Ho, Sunyani, etc.)
- School/institution names (Presec, Mfantsipim, Opoku Ware, KNUST, etc.)

Style: informal, spoken Twi — as a real Ghanaian would speak on a phone call to an emergency dispatcher. \
Natural contractions and flow. Not formal/written Twi.\
"""


def translate_en_to_lang(en_text: str, lang: str = "twi", retries: int = 3) -> str:
    # translate english to the given ghanaian language using sonnet
    system = _TWI_SYSTEM_PROMPT if lang == "twi" else LANG_PROMPTS.get(lang)
    if not system:
        raise ValueError(f"unsupported language: {lang}")

    prompt = (
        "Translate the following English emergency narrative into natural spoken "
        f"{'Twi (Akan)' if lang == 'twi' else lang.capitalize()}.\n"
        "Return ONLY the translation — no explanation, no English, no preamble.\n\n"
        f"English:\n{en_text}"
    )
    for attempt in range(retries):
        try:
            client = _get_claude()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "overloaded" in err_str.lower() or "rate" in err_str.lower():
                wait = _parse_retry_delay(e, default=30.0)
                print(f"[translate:{lang}] rate limited — sleeping {wait:.0f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif "connection" in err_str.lower() or "timeout" in err_str.lower():
                print(f"[translate:{lang}] connection error — retrying in 5s (attempt {attempt+1}/{retries}): {e}")
                time.sleep(5)
            else:
                print(f"[translate:{lang}] ERROR: {e}")
                break
    return en_text


# convenience alias for twi (used by existing translate_to_twi.py)
def translate_en_to_twi(en_text: str, retries: int = 3) -> str:
    return translate_en_to_lang(en_text, lang="twi", retries=retries)
