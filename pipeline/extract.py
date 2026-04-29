import json
import os
import re
import time
import anthropic
from anthropic import APIStatusError
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

EXTRACTION_SCHEMA = {
    "injury_type": "unknown",
    "mechanism": "unknown",
    "severity": "unknown",
    "body_region": "unknown",
    "victim_sex": "unknown",
    "victim_age_group": "unknown",
    "location_description": "unknown",
}

# 6 headline fields for evaluation (mechanism retained in output but excluded from F1)
EVAL_FIELDS = ["injury_type", "severity", "body_region", "victim_sex", "victim_age_group", "location_description"]


def _load_prompt() -> str:
    # load extraction + first_aid prompt template
    return (PROMPTS_DIR / "extraction_prompt.txt").read_text()


def _parse_response(raw: str) -> tuple[dict, str]:
    # parse dual-output response; returns (extraction_dict, first_aid_str)
    extraction = EXTRACTION_SCHEMA.copy()
    first_aid = ""
    cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if "extraction" in parsed and isinstance(parsed["extraction"], dict):
            for key in EXTRACTION_SCHEMA:
                val = parsed["extraction"].get(key)
                if val:
                    extraction[key] = str(val)
        if "first_aid" in parsed and isinstance(parsed["first_aid"], str):
            first_aid = parsed["first_aid"].strip()
    except (json.JSONDecodeError, ValueError):
        pass  # return all-unknown defaults
    return extraction, first_aid


def _fatal_billing_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "credit balance" in s or ("billing" in s and "upgrade" in s)


def _transient_extract_error(e: BaseException) -> bool:
    s = str(e).lower()
    if any(
        x in s
        for x in (
            "timeout",
            "timed out",
            "interrupted",
            "connection",
            "reset",
            "temporarily unavailable",
            "overloaded",
        )
    ):
        return True
    if isinstance(e, APIStatusError) and e.status_code in (429, 500, 502, 503, 529):
        return True
    return False


def extract(english_text: str) -> dict:
    """
    extract structured fields + first aid guidance from english text.
    returns {"extraction": {...7 keys...}, "first_aid": "...", optional "fatal_billing": True}
    never raises.
    """
    _timeout = float(os.getenv("ANTHROPIC_TIMEOUT") or "300")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=_timeout)
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{TRANSLATED_TEXT}", english_text)

    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            extraction, first_aid = _parse_response(raw)
            return {"extraction": extraction, "first_aid": first_aid}
        except APIStatusError as e:
            if e.status_code == 429:
                wait = 15 + min(90, attempt * 20)
                print(f"[extract] rate limited — sleeping {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
                continue
            if _transient_extract_error(e) and attempt + 1 < max_attempts:
                wait = 12 + min(120, attempt * 18)
                print(f"[extract] transient API error — sleeping {wait}s (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(wait)
                continue
            print(f"[extract] error: {e}")
            out = {"extraction": EXTRACTION_SCHEMA.copy(), "first_aid": ""}
            if _fatal_billing_error(e):
                out["fatal_billing"] = True
            return out
        except Exception as e:
            err = str(e).lower()
            if _transient_extract_error(e) and attempt + 1 < max_attempts:
                wait = 15 + min(120, attempt * 20)
                print(f"[extract] transient error — sleeping {wait}s (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(wait)
                continue
            if ("429" in str(e) or "rate_limit" in err) and attempt + 1 < max_attempts:
                wait = 20 + min(90, attempt * 25)
                print(f"[extract] rate limited (fallback) — sleeping {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
                continue
            print(f"[extract] error: {e}")
            out = {"extraction": EXTRACTION_SCHEMA.copy(), "first_aid": ""}
            if _fatal_billing_error(e):
                out["fatal_billing"] = True
            return out
    print("[extract] error: exhausted retries after rate limits")
    return {"extraction": EXTRACTION_SCHEMA.copy(), "first_aid": ""}
