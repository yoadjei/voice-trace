# VoiceTrace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a Twi-language injury surveillance pipeline (Khaya ASR → Khaya translate → Claude extract → SQLite/CSV → Streamlit) on 120 synthetic narratives, with results ready to fill the abstract by April 21.

**Architecture:** Two separate flows — `data_gen/` generates synthetic test inputs (English → Twi → TTS audio), and `pipeline/` is the research contribution (ASR → translate → extract → geocode → store). Each pipeline stage is a standalone module with one callable function. All intermediate outputs are saved to disk so any stage can be re-run independently without re-calling APIs.

**Tech Stack:** Python 3.11+, `anthropic`, `requests`, `python-dotenv`, `jiwer`, `scikit-learn`, `pandas`, `streamlit`, `geopy`, `pytest`

---

## File Map

| File | Responsibility |
|---|---|
| `.env` | API keys (gitignored) |
| `requirements.txt` | All dependencies pinned |
| `data/distributions/ghana_injury_distributions.json` | Locked sampling weights — never changed after Task 1 |
| `data/synthetic/narratives_en.csv` | 120 Claude-generated English narratives |
| `data/synthetic/narratives_twi.csv` | Khaya-translated Twi versions |
| `data/synthetic/audio/*.wav` | Khaya TTS audio files |
| `data/synthetic/asr_transcripts.csv` | Raw Khaya ASR output |
| `data/synthetic/translations_en.csv` | ASR transcripts translated back to English |
| `data/synthetic/gold_annotations.csv` | Manual ground truth (7 fields × 120 rows) |
| `pipeline/asr.py` | `transcribe(wav_path) -> str` |
| `pipeline/translate.py` | `translate(text, source_lang, target_lang) -> str` |
| `pipeline/extract.py` | `extract(english_text) -> dict` |
| `pipeline/geocode.py` | `geocode(location_str) -> dict` |
| `pipeline/pipeline.py` | `run_pipeline(wav_path) -> dict` (end-to-end orchestrator) |
| `data_gen/generate_narratives.py` | Generates `narratives_en.csv` via Claude |
| `data_gen/translate_to_twi.py` | Generates `narratives_twi.csv` via Khaya |
| `data_gen/tts.py` | Generates `audio/*.wav` via Khaya TTS |
| `evaluation/evaluate_asr.py` | Computes WER, saves to `evaluation/results/asr_results.csv` |
| `evaluation/evaluate_extraction.py` | Computes F1 per field, saves to `evaluation/results/extraction_results.csv` |
| `dashboard/app.py` | Streamlit dashboard: map + table view |
| `prompts/extraction_prompt.txt` | Claude extraction prompt (loaded at runtime, not hardcoded) |
| `tests/test_extract.py` | Tests for extraction JSON parsing and schema validation |
| `tests/test_pipeline.py` | Tests for pipeline orchestrator with mocked stages |
| `tests/test_evaluate.py` | Tests for WER and F1 computation logic |

---

## Task 1: Project Setup

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `data/distributions/ghana_injury_distributions.json`
- Create: `prompts/extraction_prompt.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p data/distributions data/synthetic/audio data/raw evaluation/results pipeline data_gen dashboard prompts tests paper/figures docs/superpowers/plans
touch pipeline/__init__.py data_gen/__init__.py evaluation/__init__.py
```

- [ ] **Step 2: Create `.gitignore`**

```
.env
data/raw/
data/synthetic/audio/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
.venv/
```

- [ ] **Step 3: Create `.env.example`**

```
KHAYA_API_KEY=your_khaya_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Copy to `.env` and fill in real keys:
```bash
cp .env.example .env
```

- [ ] **Step 4: Create `requirements.txt`**

```
anthropic==0.40.0
requests==2.32.3
python-dotenv==1.0.1
jiwer==3.0.4
scikit-learn==1.5.2
pandas==2.2.3
streamlit==1.40.2
geopy==2.4.1
pytest==8.3.3
```

Install:
```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Create `data/distributions/ghana_injury_distributions.json`**

```json
{
  "_sources": {
    "injury_type": "Boateng et al. 2019 (Korle-Bu Teaching Hospital, n=17860)",
    "sex": "Boateng et al. 2019",
    "age_group": "Opoku et al. 2025 (systematic review, n=46 studies)",
    "severity": "Opoku et al. 2025",
    "body_region": "Boateng et al. 2019 / Mesic et al. 2024",
    "location_type": "BRRI / Opoku et al. 2025"
  },
  "injury_type": {
    "rta": 0.391,
    "fall": 0.197,
    "assault": 0.120,
    "occupational": 0.164,
    "burn": 0.085,
    "drowning": 0.043
  },
  "sex": {
    "male": 0.678,
    "female": 0.322
  },
  "age_group": {
    "child": 0.18,
    "youth": 0.45,
    "adult": 0.27,
    "elderly": 0.10
  },
  "severity": {
    "minor": 0.52,
    "moderate": 0.31,
    "severe": 0.17
  },
  "body_region": {
    "head_neck": 0.38,
    "limb": 0.44,
    "trunk": 0.18
  },
  "location_type": {
    "highway": 0.41,
    "urban_road": 0.35,
    "home": 0.24
  },
  "n_narratives": 120
}
```

- [ ] **Step 6: Create `prompts/extraction_prompt.txt`**

```
You are an injury surveillance assistant for the Ghana health system.
You will receive a spoken community report of an injury, transcribed from Twi and
translated to English. The report may be informal, imprecise, or incomplete.

Extract the following structured fields from the report. If a field cannot be
determined from the text, return "unknown". Do not infer or assume beyond what
is stated.

Return ONLY valid JSON with these exact keys:
{
  "injury_type": one of [rta, fall, assault, burn, drowning, occupational, unknown],
  "mechanism": brief description (max 10 words) or "unknown",
  "severity": one of [minor, moderate, severe, unknown],
  "body_region": one of [head_neck, upper_limb, lower_limb, trunk, multiple, unknown],
  "victim_sex": one of [male, female, unknown],
  "victim_age_group": one of [child, youth, adult, elderly, unknown],
  "location_description": place name or description (max 15 words) or "unknown"
}

Report: """
{TRANSLATED_TEXT}
"""
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore .env.example requirements.txt data/distributions/ghana_injury_distributions.json prompts/extraction_prompt.txt
git commit -m "feat: project scaffold, distributions, extraction prompt"
```

---

## Task 2: Claude Extraction Module

Build and test the extraction module first — it's the core research contribution and has no external dependency beyond the API.

**Files:**
- Create: `pipeline/extract.py`
- Create: `tests/test_extract.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extract.py
import json
import pytest
from unittest.mock import patch, MagicMock
from pipeline.extract import extract, _parse_response, VALID_SCHEMA

class TestParseResponse:
    def test_valid_json_returned_as_dict(self):
        raw = json.dumps({
            "injury_type": "rta",
            "mechanism": "pedestrian hit by vehicle",
            "severity": "moderate",
            "body_region": "lower_limb",
            "victim_sex": "male",
            "victim_age_group": "youth",
            "location_description": "Accra-Kumasi highway near Suhum"
        })
        result = _parse_response(raw)
        assert result["injury_type"] == "rta"
        assert result["victim_sex"] == "male"

    def test_missing_keys_filled_with_unknown(self):
        raw = json.dumps({"injury_type": "fall"})
        result = _parse_response(raw)
        assert result["mechanism"] == "unknown"
        assert result["severity"] == "unknown"
        assert result["body_region"] == "unknown"
        assert result["victim_sex"] == "unknown"
        assert result["victim_age_group"] == "unknown"
        assert result["location_description"] == "unknown"

    def test_invalid_json_returns_all_unknown(self):
        result = _parse_response("this is not json")
        assert result["injury_type"] == "unknown"
        assert result["mechanism"] == "unknown"

    def test_json_wrapped_in_markdown_code_block(self):
        raw = '```json\n{"injury_type": "burn", "mechanism": "cooking fire", "severity": "minor", "body_region": "upper_limb", "victim_sex": "female", "victim_age_group": "adult", "location_description": "home kitchen"}\n```'
        result = _parse_response(raw)
        assert result["injury_type"] == "burn"

class TestExtract:
    def test_extract_calls_claude_and_returns_dict(self):
        mock_response_text = json.dumps({
            "injury_type": "fall",
            "mechanism": "fell from tree",
            "severity": "moderate",
            "body_region": "upper_limb",
            "victim_sex": "male",
            "victim_age_group": "child",
            "location_description": "farm near Ejura"
        })
        with patch("pipeline.extract.anthropic.Anthropic") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text=mock_response_text)
            ]
            result = extract("A boy fell from a tree on the farm.")
        assert result["injury_type"] == "fall"
        assert isinstance(result, dict)
        assert set(result.keys()) == set(VALID_SCHEMA.keys())
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_extract.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.extract'`

- [ ] **Step 3: Implement `pipeline/extract.py`**

```python
import json
import os
import re
import anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

VALID_SCHEMA = {
    "injury_type": "unknown",
    "mechanism": "unknown",
    "severity": "unknown",
    "body_region": "unknown",
    "victim_sex": "unknown",
    "victim_age_group": "unknown",
    "location_description": "unknown",
}


def _load_prompt() -> str:
    return (PROMPTS_DIR / "extraction_prompt.txt").read_text()


def _parse_response(raw: str) -> dict:
    """Parse Claude's response into a validated dict. Always returns all 7 keys."""
    result = VALID_SCHEMA.copy()
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        for key in VALID_SCHEMA:
            if key in parsed and parsed[key]:
                result[key] = str(parsed[key])
    except (json.JSONDecodeError, ValueError):
        pass  # return all-unknown default
    return result


def extract(english_text: str) -> dict:
    """
    Extract structured injury fields from an English injury report.
    Returns a dict with 7 keys. Never raises — returns all 'unknown' on failure.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{TRANSLATED_TEXT}", english_text)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return _parse_response(raw)
    except Exception as e:
        print(f"[extract] ERROR: {e}")
        return VALID_SCHEMA.copy()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_extract.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/extract.py tests/test_extract.py
git commit -m "feat: Claude extraction module with schema validation"
```

---

## Task 3: Khaya API Verification (Thin Slice)

Before writing the other pipeline modules, verify that both APIs respond correctly on a single example. This is a manual verification step — not automated.

**Files:**
- Create: `scripts/verify_apis.py` (throwaway script, not committed to main modules)

- [ ] **Step 1: Create verification script**

```python
# scripts/verify_apis.py
"""
Run this once to verify all API integrations work before building the pipeline.
Usage: python scripts/verify_apis.py
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

KHAYA_KEY = os.getenv("KHAYA_API_KEY")
KHAYA_HEADERS = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": KHAYA_KEY}
KHAYA_TRANSLATE_URL = "https://translation-api.ghananlp.org/v1/translate"
KHAYA_TTS_URL = "https://tts.ghananlp.org/v1/tts"
KHAYA_ASR_URL = "https://asr.ghananlp.org/v1/transcribe"

SAMPLE_EN = "A young man was hit by a motorbike on the main road and broke his leg."
SAMPLE_TWI = ""  # will be filled after translation step

def test_translate_en_to_twi():
    print("\n=== 1. Translate EN → TWI ===")
    resp = requests.post(
        KHAYA_TRANSLATE_URL,
        headers=KHAYA_HEADERS,
        json={"in": SAMPLE_EN, "lang": "en-tw"},
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:300]}")
    assert resp.status_code == 200, f"Translation failed: {resp.text}"
    return resp.json().get("translatedText") or resp.json().get("out") or resp.text

def test_tts(twi_text: str):
    print("\n=== 2. Khaya TTS (TWI → WAV) ===")
    resp = requests.post(
        KHAYA_TTS_URL,
        headers=KHAYA_HEADERS,
        json={"text": twi_text, "language": "tw"},
    )
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")
    assert resp.status_code == 200, f"TTS failed: {resp.text[:300]}"
    with open("verify_test.wav", "wb") as f:
        f.write(resp.content)
    print("Saved: verify_test.wav")
    return "verify_test.wav"

def test_asr(wav_path: str):
    print("\n=== 3. Khaya ASR (WAV → TWI text) ===")
    with open(wav_path, "rb") as f:
        resp = requests.post(
            KHAYA_ASR_URL,
            headers={"Ocp-Apim-Subscription-Key": KHAYA_KEY},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"language": "tw"},
        )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:300]}")
    assert resp.status_code == 200, f"ASR failed: {resp.text}"
    return resp.json().get("transcript") or resp.text

def test_translate_twi_to_en(twi_text: str):
    print("\n=== 4. Translate TWI → EN ===")
    resp = requests.post(
        KHAYA_TRANSLATE_URL,
        headers=KHAYA_HEADERS,
        json={"in": twi_text, "lang": "tw-en"},
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:300]}")
    assert resp.status_code == 200, f"Translation failed: {resp.text}"
    return resp.json().get("translatedText") or resp.json().get("out") or resp.text

def test_claude_extract(english_text: str):
    print("\n=== 5. Claude Extraction ===")
    from pipeline.extract import extract
    result = extract(english_text)
    print(f"Extracted: {result}")
    assert result["injury_type"] != ""
    return result

if __name__ == "__main__":
    twi = test_translate_en_to_twi()
    print(f"TWI: {twi}")
    time.sleep(0.5)

    wav = test_tts(twi)
    time.sleep(0.5)

    asr_twi = test_asr(wav)
    print(f"ASR output: {asr_twi}")
    time.sleep(0.5)

    en_back = test_translate_twi_to_en(asr_twi)
    print(f"Back-translated EN: {en_back}")
    time.sleep(0.5)

    result = test_claude_extract(en_back)

    print("\n=== ALL STAGES PASSED ===")
    print("Original EN:", SAMPLE_EN)
    print("Back-translated EN:", en_back)
    print("Extracted:", result)
```

- [ ] **Step 2: Run the verification script**

```bash
mkdir -p scripts
python scripts/verify_apis.py
```

Expected: all 5 stages print `Status: 200` and a non-empty response.

**If any stage fails:** Check the actual Khaya API docs at https://ghananlp.org for correct endpoint URLs, request format, and auth header name. The script above uses the most likely format but the exact field names (`in`, `lang`, `translatedText`) may differ. Update the script and the corresponding pipeline modules to match.

- [ ] **Step 3: Note actual Khaya API field names**

After running the script, record the actual request/response field names from working calls here in a comment at the top of `pipeline/asr.py`, `pipeline/translate.py`, and `data_gen/tts.py`. This is your reference for the next tasks.

- [ ] **Step 4: Clean up and commit**

```bash
git add scripts/verify_apis.py
git commit -m "chore: api verification script — confirms Khaya + Claude integration"
```

---

## Task 4: Pipeline Modules (ASR, Translate, Geocode)

**Files:**
- Create: `pipeline/asr.py`
- Create: `pipeline/translate.py`
- Create: `pipeline/geocode.py`

> **Note:** Use the exact Khaya API field names confirmed in Task 3. The code below uses the most commonly documented field names — update them to match actual API behaviour if they differ.

- [ ] **Step 1: Create `pipeline/asr.py`**

```python
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

KHAYA_KEY = os.getenv("KHAYA_API_KEY")
ASR_URL = "https://asr.ghananlp.org/v1/transcribe"


def transcribe(wav_path: str, language: str = "tw") -> str:
    """
    Transcribe a .wav audio file using Khaya ASR.
    Returns the transcript string. Returns empty string on failure.
    """
    headers = {"Ocp-Apim-Subscription-Key": KHAYA_KEY}
    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                ASR_URL,
                headers=headers,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"language": language},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("transcript", data.get("text", ""))
    except Exception as e:
        print(f"[asr] ERROR on {wav_path}: {e}")
        return ""
```

- [ ] **Step 2: Create `pipeline/translate.py`**

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

KHAYA_KEY = os.getenv("KHAYA_API_KEY")
TRANSLATE_URL = "https://translation-api.ghananlp.org/v1/translate"


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using Khaya translation API.
    lang_pair format: "en-tw" or "tw-en"
    Returns translated string. Returns original text on failure.
    """
    lang_pair = f"{source_lang}-{target_lang}"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": KHAYA_KEY,
    }
    try:
        resp = requests.post(
            TRANSLATE_URL,
            headers=headers,
            json={"in": text, "lang": lang_pair},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("translatedText", data.get("out", text))
    except Exception as e:
        print(f"[translate] ERROR ({source_lang}→{target_lang}): {e}")
        return text
```

- [ ] **Step 3: Create `pipeline/geocode.py`**

```python
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

_geolocator = Nominatim(user_agent="voicetrace-ghana-injury-surveillance")


def geocode(location_str: str) -> dict:
    """
    Geocode a location string to lat/lng using Nominatim (free, no key needed).
    Returns dict with keys: lat, lng, display_name. Returns None values on failure.
    Appends ', Ghana' to improve accuracy.
    """
    if not location_str or location_str == "unknown":
        return {"lat": None, "lng": None, "display_name": None}

    query = f"{location_str}, Ghana"
    try:
        time.sleep(1)  # Nominatim rate limit: 1 request/second
        location = _geolocator.geocode(query, timeout=10)
        if location:
            return {
                "lat": location.latitude,
                "lng": location.longitude,
                "display_name": location.address,
            }
    except GeocoderTimedOut:
        print(f"[geocode] Timeout for: {query}")
    except Exception as e:
        print(f"[geocode] ERROR for '{query}': {e}")

    return {"lat": None, "lng": None, "display_name": None}
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/asr.py pipeline/translate.py pipeline/geocode.py
git commit -m "feat: ASR, translation, and geocoding pipeline modules"
```

---

## Task 5: Pipeline Orchestrator

**Files:**
- Create: `pipeline/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock


class TestRunPipeline:
    def test_returns_dict_with_all_expected_keys(self):
        with patch("pipeline.pipeline.transcribe", return_value="Saa ɔbarima bi wɔ fie no mu"), \
             patch("pipeline.pipeline.translate", return_value="A man was injured at home"), \
             patch("pipeline.pipeline.extract", return_value={
                 "injury_type": "fall",
                 "mechanism": "fell at home",
                 "severity": "minor",
                 "body_region": "lower_limb",
                 "victim_sex": "male",
                 "victim_age_group": "adult",
                 "location_description": "home"
             }), \
             patch("pipeline.pipeline.geocode", return_value={"lat": 5.6, "lng": -0.2, "display_name": "Accra, Ghana"}):
            from pipeline.pipeline import run_pipeline
            result = run_pipeline("fake/path.wav")

        expected_keys = {
            "asr_transcript", "translated_text", "injury_type", "mechanism",
            "severity", "body_region", "victim_sex", "victim_age_group",
            "location_description", "lat", "lng"
        }
        assert set(result.keys()) == expected_keys

    def test_asr_failure_returns_unknown_fields(self):
        with patch("pipeline.pipeline.transcribe", return_value=""), \
             patch("pipeline.pipeline.translate", return_value=""), \
             patch("pipeline.pipeline.extract", return_value={
                 "injury_type": "unknown", "mechanism": "unknown", "severity": "unknown",
                 "body_region": "unknown", "victim_sex": "unknown",
                 "victim_age_group": "unknown", "location_description": "unknown"
             }), \
             patch("pipeline.pipeline.geocode", return_value={"lat": None, "lng": None, "display_name": None}):
            from pipeline.pipeline import run_pipeline
            result = run_pipeline("fake/path.wav")

        assert result["asr_transcript"] == ""
        assert result["injury_type"] == "unknown"
        assert result["lat"] is None
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.pipeline'`

- [ ] **Step 3: Implement `pipeline/pipeline.py`**

```python
import time
from pipeline.asr import transcribe
from pipeline.translate import translate
from pipeline.extract import extract
from pipeline.geocode import geocode


def run_pipeline(wav_path: str) -> dict:
    """
    Run the full VoiceTrace pipeline on a single .wav file.
    Returns a flat dict with all extracted fields + geocoordinates.
    Never raises — returns partial results if any stage fails.
    """
    # Stage 1: ASR (Twi audio → Twi text)
    asr_transcript = transcribe(wav_path)
    time.sleep(0.5)

    # Stage 2: Translate (Twi → English)
    translated_text = translate(asr_transcript, source_lang="tw", target_lang="en") if asr_transcript else ""
    time.sleep(0.5)

    # Stage 3: Extract (English → structured JSON)
    extracted = extract(translated_text) if translated_text else {
        "injury_type": "unknown", "mechanism": "unknown", "severity": "unknown",
        "body_region": "unknown", "victim_sex": "unknown",
        "victim_age_group": "unknown", "location_description": "unknown",
    }

    # Stage 4: Geocode
    geo = geocode(extracted.get("location_description", "unknown"))

    return {
        "asr_transcript": asr_transcript,
        "translated_text": translated_text,
        **extracted,
        "lat": geo["lat"],
        "lng": geo["lng"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end pipeline orchestrator with stage isolation"
```

---

## Task 6: Data Generation — English Narratives

**Files:**
- Create: `data_gen/generate_narratives.py`

- [ ] **Step 1: Implement `data_gen/generate_narratives.py`**

```python
"""
Generates 120 epidemiologically grounded English injury narratives using Claude.
Saves to data/synthetic/narratives_en.csv.

Run: python -m data_gen.generate_narratives
"""
import json
import os
import time
import random
import pandas as pd
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DISTRIBUTIONS_PATH = Path("data/distributions/ghana_injury_distributions.json")
OUTPUT_PATH = Path("data/synthetic/narratives_en.csv")

GENERATION_PROMPT = """Generate a naturalistic spoken injury report (2-5 sentences) as if a community member in Ghana is reporting to a health hotline. The report should describe:
- Injury type: {injury_type}
- Victim: {age_group}, {sex}
- Severity: {severity}
- Body region affected: {body_region}
- Location: {location_type}

Write in simple, informal English as if the caller is not medically trained. Vary sentence structure. Do not use medical jargon. Do not include a greeting or sign-off — just the report itself."""

LOCATION_EXAMPLES = {
    "highway": ["Accra-Kumasi highway near Nsawam", "Tema Motorway near Ashaiman", "N1 highway near Kasoa", "Accra-Cape Coast road near Winneba"],
    "urban_road": ["Nima, Accra", "Adum, Kumasi", "Takoradi market area", "Tamale central", "Koforidua town centre"],
    "home": ["their home in Tesano", "a house in Suame", "home in Bolgatanga", "a compound house in Agbogbloshie"],
}


def _sample_row(distributions: dict, rng: random.Random) -> dict:
    def weighted_choice(d):
        keys, weights = zip(*d.items())
        return rng.choices(keys, weights=weights, k=1)[0]

    injury_type = weighted_choice(distributions["injury_type"])
    sex = weighted_choice(distributions["sex"])
    age_group = weighted_choice(distributions["age_group"])
    severity = weighted_choice(distributions["severity"])
    body_region = weighted_choice(distributions["body_region"])
    location_type = weighted_choice(distributions["location_type"])
    location = rng.choice(LOCATION_EXAMPLES[location_type])

    return {
        "injury_type": injury_type,
        "sex": sex,
        "age_group": age_group,
        "severity": severity,
        "body_region": body_region,
        "location_type": location_type,
        "location": location,
    }


def generate_narratives(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    distributions = json.loads(DISTRIBUTIONS_PATH.read_text())
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    rows = []
    for i in range(n):
        meta = _sample_row(distributions, rng)
        prompt = GENERATION_PROMPT.format(
            injury_type=meta["injury_type"].upper(),
            sex=meta["sex"],
            age_group=meta["age_group"],
            severity=meta["severity"],
            body_region=meta["body_region"].replace("_", "/"),
            location_type=f"{meta['location_type'].replace('_', ' ')} ({meta['location']})",
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            narrative = response.content[0].text.strip()
        except Exception as e:
            print(f"[generate] ERROR on row {i}: {e}")
            narrative = ""

        row = {"id": i, "narrative_en": narrative, **meta}
        rows.append(row)
        print(f"[{i+1}/{n}] {meta['injury_type']} | {meta['sex']} | {meta['age_group']}")
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} narratives to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    generate_narratives()
```

- [ ] **Step 2: Run to generate the dataset**

```bash
python -m data_gen.generate_narratives
```

Expected: `data/synthetic/narratives_en.csv` created with 120 rows and columns: `id, narrative_en, injury_type, sex, age_group, severity, body_region, location_type, location`

- [ ] **Step 3: Spot-check output**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/synthetic/narratives_en.csv'); print(df[['injury_type','narrative_en']].head(10).to_string())"
```

Verify narratives read as natural speech, not clinical notes.

- [ ] **Step 4: Commit**

```bash
git add data_gen/generate_narratives.py data/synthetic/narratives_en.csv
git commit -m "feat: generate 120 epidemiologically grounded English injury narratives"
```

---

## Task 7: Data Generation — Twi Translation and TTS

**Files:**
- Create: `data_gen/translate_to_twi.py`
- Create: `data_gen/tts.py`

- [ ] **Step 1: Create `data_gen/translate_to_twi.py`**

```python
"""
Translates English narratives to Twi using Khaya translation API.
Reads:  data/synthetic/narratives_en.csv
Writes: data/synthetic/narratives_twi.csv

Run: python -m data_gen.translate_to_twi
"""
import time
import pandas as pd
from pathlib import Path
from pipeline.translate import translate

INPUT_PATH = Path("data/synthetic/narratives_en.csv")
OUTPUT_PATH = Path("data/synthetic/narratives_twi.csv")


def translate_to_twi() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    twi_narratives = []

    for i, row in df.iterrows():
        en_text = row["narrative_en"]
        if not isinstance(en_text, str) or not en_text.strip():
            twi_narratives.append("")
            continue

        twi = translate(en_text, source_lang="en", target_lang="tw")
        twi_narratives.append(twi)
        print(f"[{i+1}/{len(df)}] translated")
        time.sleep(0.5)

    df["narrative_twi"] = twi_narratives
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    translate_to_twi()
```

- [ ] **Step 2: Create `data_gen/tts.py`**

```python
"""
Generates .wav audio files from Twi narratives using Khaya TTS.
Reads:  data/synthetic/narratives_twi.csv
Writes: data/synthetic/audio/narrative_{id:03d}.wav

Run: python -m data_gen.tts
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KHAYA_KEY = os.getenv("KHAYA_API_KEY")
TTS_URL = "https://tts.ghananlp.org/v1/tts"
HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": KHAYA_KEY,
}

INPUT_PATH = Path("data/synthetic/narratives_twi.csv")
AUDIO_DIR = Path("data/synthetic/audio")


def generate_audio() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    for i, row in df.iterrows():
        twi_text = row.get("narrative_twi", "")
        if not isinstance(twi_text, str) or not twi_text.strip():
            print(f"[{i+1}/{len(df)}] SKIP — empty Twi text")
            continue

        wav_path = AUDIO_DIR / f"narrative_{int(row['id']):03d}.wav"
        if wav_path.exists():
            print(f"[{i+1}/{len(df)}] SKIP — already exists")
            continue

        try:
            resp = requests.post(
                TTS_URL,
                headers=HEADERS,
                json={"text": twi_text, "language": "tw"},
                timeout=30,
            )
            resp.raise_for_status()
            wav_path.write_bytes(resp.content)
            print(f"[{i+1}/{len(df)}] saved {wav_path.name}")
        except Exception as e:
            print(f"[{i+1}/{len(df)}] ERROR: {e}")

        time.sleep(0.5)

    print(f"\nDone. Audio files in {AUDIO_DIR}")


if __name__ == "__main__":
    generate_audio()
```

- [ ] **Step 3: Run translation**

```bash
python -m data_gen.translate_to_twi
```

Expected: `data/synthetic/narratives_twi.csv` with `narrative_twi` column populated.

- [ ] **Step 4: Run TTS**

```bash
python -m data_gen.tts
```

Expected: 120 `.wav` files in `data/synthetic/audio/`. Check a few with any audio player to confirm they are audible Twi speech.

- [ ] **Step 5: Commit**

```bash
git add data_gen/translate_to_twi.py data_gen/tts.py data/synthetic/narratives_twi.csv
git commit -m "feat: Twi translation and TTS audio generation scripts"
```

---

## Task 8: Batch Pipeline Run

Run the full pipeline on all 120 audio files and save intermediate + final outputs.

**Files:**
- Create: `run_pipeline_batch.py`

- [ ] **Step 1: Create `run_pipeline_batch.py`**

```python
"""
Runs the VoiceTrace pipeline on all 120 synthetic audio files.
Saves intermediate and final results to data/synthetic/.

Run: python run_pipeline_batch.py
"""
import time
import pandas as pd
from pathlib import Path
from pipeline.asr import transcribe
from pipeline.translate import translate
from pipeline.extract import extract
from pipeline.geocode import geocode

AUDIO_DIR = Path("data/synthetic/audio")
NARRATIVES_PATH = Path("data/synthetic/narratives_en.csv")
ASR_OUTPUT = Path("data/synthetic/asr_transcripts.csv")
TRANSLATIONS_OUTPUT = Path("data/synthetic/translations_en.csv")
FINAL_OUTPUT = Path("data/synthetic/pipeline_results.csv")


def run_batch() -> None:
    df = pd.read_csv(NARRATIVES_PATH)
    asr_rows, translation_rows, result_rows = [], [], []

    for i, row in df.iterrows():
        narrative_id = int(row["id"])
        wav_path = AUDIO_DIR / f"narrative_{narrative_id:03d}.wav"

        if not wav_path.exists():
            print(f"[{i+1}] SKIP — no audio file: {wav_path}")
            continue

        print(f"\n[{i+1}/120] id={narrative_id}")

        # Stage 1: ASR
        asr_text = transcribe(str(wav_path))
        asr_rows.append({"id": narrative_id, "asr_transcript": asr_text})
        print(f"  ASR: {asr_text[:80]}...")
        time.sleep(0.5)

        # Stage 2: Translate ASR output back to English
        en_text = translate(asr_text, source_lang="tw", target_lang="en") if asr_text else ""
        translation_rows.append({"id": narrative_id, "translated_en": en_text})
        print(f"  Translated: {en_text[:80]}...")
        time.sleep(0.5)

        # Stage 3: Extract
        fields = extract(en_text) if en_text else {k: "unknown" for k in [
            "injury_type", "mechanism", "severity", "body_region",
            "victim_sex", "victim_age_group", "location_description"
        ]}

        # Stage 4: Geocode
        geo = geocode(fields.get("location_description", "unknown"))

        result_rows.append({
            "id": narrative_id,
            "asr_transcript": asr_text,
            "translated_en": en_text,
            **fields,
            "lat": geo["lat"],
            "lng": geo["lng"],
        })

    # Save intermediate outputs
    pd.DataFrame(asr_rows).to_csv(ASR_OUTPUT, index=False)
    pd.DataFrame(translation_rows).to_csv(TRANSLATIONS_OUTPUT, index=False)
    pd.DataFrame(result_rows).to_csv(FINAL_OUTPUT, index=False)

    print(f"\nDone.")
    print(f"  ASR transcripts → {ASR_OUTPUT}")
    print(f"  Translations    → {TRANSLATIONS_OUTPUT}")
    print(f"  Final results   → {FINAL_OUTPUT}")


if __name__ == "__main__":
    run_batch()
```

- [ ] **Step 2: Run the batch pipeline**

```bash
python run_pipeline_batch.py
```

Expected: three CSV files created in `data/synthetic/`. This will take ~10–20 minutes due to API rate limiting delays.

- [ ] **Step 3: Verify output shape**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/synthetic/pipeline_results.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(df[['id','injury_type','severity','victim_sex']].head(5).to_string())
"
```

Expected: 120 rows, 11 columns.

- [ ] **Step 4: Commit**

```bash
git add run_pipeline_batch.py data/synthetic/asr_transcripts.csv data/synthetic/translations_en.csv data/synthetic/pipeline_results.csv
git commit -m "feat: batch pipeline runner — processes all 120 audio files"
```

---

## Task 9: Gold Annotation

This task is manual — you annotate all 120 English narratives against the 7-field schema.

**Files:**
- Create: `data/synthetic/gold_annotations.csv` (manually)
- Create: `evaluation/annotator_guide.md`

- [ ] **Step 1: Create annotator guide**

```markdown
# VoiceTrace Annotation Guide

Annotate each row of `narratives_en.csv` against the schema below.
Read the `narrative_en` column. Fill in each field based ONLY on what is stated — do not infer.

## Schema

| Field | Valid values |
|---|---|
| injury_type | rta, fall, assault, burn, drowning, occupational, unknown |
| mechanism | free text, max 10 words, or "unknown" |
| severity | minor, moderate, severe, unknown |
| body_region | head_neck, upper_limb, lower_limb, trunk, multiple, unknown |
| victim_sex | male, female, unknown |
| victim_age_group | child (<15), youth (15–34), adult (35–54), elderly (55+), unknown |
| location_description | free text, max 15 words, or "unknown" |

## Rules
- If a field is ambiguous: write "unknown"
- Do not use the metadata columns (injury_type, sex etc.) — annotate from narrative_en only
- Mechanism = HOW the injury happened (e.g. "pedestrian struck by motorbike", "fell from ladder")
- Severity cues: minor = walk-in, no hospitalization implied; moderate = needs treatment; severe = life-threatening, critical, unconscious
```

- [ ] **Step 2: Create the annotation CSV template**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/synthetic/narratives_en.csv')[['id','narrative_en']]
for col in ['injury_type','mechanism','severity','body_region','victim_sex','victim_age_group','location_description']:
    df[col] = ''
df.to_csv('data/synthetic/gold_annotations.csv', index=False)
print(f'Template created: {len(df)} rows')
"
```

- [ ] **Step 3: First annotation pass**

Open `data/synthetic/gold_annotations.csv` in a spreadsheet editor (Excel, LibreOffice, or VS Code with CSV extension). Fill in all 7 fields for all 120 rows using `narrative_en` only. **Do not use the metadata columns.**

Estimated time: 2–3 hours.

- [ ] **Step 4: Second annotation pass (24 hours later)**

Close the file. Wait at least 24 hours. Open again. Re-annotate all 120 rows from scratch in a copy called `gold_annotations_pass2.csv`. Compare pass 1 vs pass 2 for each categorical field. Record consistency rate.

- [ ] **Step 5: Resolve disagreements and finalize**

Where pass 1 and pass 2 differ, re-read the narrative and pick the better answer. Save the resolved version as `gold_annotations.csv`. Record the pre-resolution consistency rate in your paper (Section 4).

- [ ] **Step 6: Commit**

```bash
git add data/synthetic/gold_annotations.csv evaluation/annotator_guide.md
git commit -m "data: gold annotations — 120 manually annotated injury narratives"
```

---

## Task 10: Evaluation

**Files:**
- Create: `evaluation/evaluate_asr.py`
- Create: `evaluation/evaluate_extraction.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py
import pytest
from evaluation.evaluate_asr import compute_wer
from evaluation.evaluate_extraction import compute_field_f1


class TestComputeWer:
    def test_identical_strings_give_zero_wer(self):
        refs = ["me ko fie", "ɔbarima no tee"]
        hyps = ["me ko fie", "ɔbarima no tee"]
        result = compute_wer(refs, hyps)
        assert result["overall_wer"] == pytest.approx(0.0)

    def test_completely_wrong_gives_high_wer(self):
        refs = ["me ko fie"]
        hyps = ["xyz abc def"]
        result = compute_wer(refs, hyps)
        assert result["overall_wer"] > 0.5

    def test_returns_expected_keys(self):
        result = compute_wer(["a b c"], ["a b d"])
        assert "overall_wer" in result
        assert "n_references" in result


class TestComputeFieldF1:
    def test_perfect_predictions_give_f1_one(self):
        gold = [{"injury_type": "rta", "severity": "minor"}]
        pred = [{"injury_type": "rta", "severity": "minor"}]
        result = compute_field_f1(gold, pred, fields=["injury_type", "severity"])
        assert result["injury_type"]["f1"] == pytest.approx(1.0)

    def test_wrong_predictions_give_lower_f1(self):
        gold = [{"injury_type": "rta"}, {"injury_type": "fall"}]
        pred = [{"injury_type": "fall"}, {"injury_type": "fall"}]
        result = compute_field_f1(gold, pred, fields=["injury_type"])
        assert result["injury_type"]["f1"] < 1.0

    def test_returns_macro_f1(self):
        gold = [{"injury_type": "rta", "severity": "minor"}]
        pred = [{"injury_type": "rta", "severity": "minor"}]
        result = compute_field_f1(gold, pred, fields=["injury_type", "severity"])
        assert "macro_f1" in result
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `evaluation/evaluate_asr.py`**

```python
"""
Computes WER for Khaya ASR output vs. original Twi narratives.
Reads: data/synthetic/narratives_twi.csv, data/synthetic/asr_transcripts.csv
Saves: evaluation/results/asr_results.csv

Run: python -m evaluation.evaluate_asr
"""
import pandas as pd
from jiwer import wer
from pathlib import Path

NARRATIVES_TWI = Path("data/synthetic/narratives_twi.csv")
ASR_TRANSCRIPTS = Path("data/synthetic/asr_transcripts.csv")
OUTPUT_PATH = Path("evaluation/results/asr_results.csv")


def compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    """Compute WER between reference and hypothesis string lists."""
    valid = [(r, h) for r, h in zip(references, hypotheses) if r and h]
    if not valid:
        return {"overall_wer": None, "n_references": 0}
    refs, hyps = zip(*valid)
    overall = wer(list(refs), list(hyps))
    return {"overall_wer": round(overall, 4), "n_references": len(valid)}


def run_asr_evaluation() -> dict:
    twi_df = pd.read_csv(NARRATIVES_TWI)
    asr_df = pd.read_csv(ASR_TRANSCRIPTS)
    merged = twi_df.merge(asr_df, on="id")

    overall = compute_wer(
        merged["narrative_twi"].tolist(),
        merged["asr_transcript"].tolist()
    )

    # WER broken down by injury type
    per_type = []
    for injury_type, group in merged.groupby("injury_type"):
        type_wer = compute_wer(
            group["narrative_twi"].tolist(),
            group["asr_transcript"].tolist()
        )
        per_type.append({"injury_type": injury_type, **type_wer})

    results_df = pd.DataFrame(per_type)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nOverall WER: {overall['overall_wer']:.1%} (n={overall['n_references']})")
    print(results_df.to_string(index=False))
    return overall


if __name__ == "__main__":
    run_asr_evaluation()
```

- [ ] **Step 4: Create `evaluation/evaluate_extraction.py`**

```python
"""
Computes precision/recall/F1 for Claude extraction vs. gold annotations.
Reads: data/synthetic/gold_annotations.csv, data/synthetic/pipeline_results.csv
Saves: evaluation/results/extraction_results.csv

Run: python -m evaluation.evaluate_extraction
"""
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
from typing import Optional
import numpy as np

GOLD_PATH = Path("data/synthetic/gold_annotations.csv")
PIPELINE_PATH = Path("data/synthetic/pipeline_results.csv")
OUTPUT_PATH = Path("evaluation/results/extraction_results.csv")

CATEGORICAL_FIELDS = ["injury_type", "severity", "body_region", "victim_sex", "victim_age_group"]


def compute_field_f1(
    gold: list[dict],
    pred: list[dict],
    fields: Optional[list[str]] = None,
) -> dict:
    """Compute per-field F1 and macro-averaged F1 for categorical fields."""
    if fields is None:
        fields = CATEGORICAL_FIELDS

    results = {}
    f1_scores = []

    for field in fields:
        y_true = [str(r.get(field, "unknown")).lower() for r in gold]
        y_pred = [str(r.get(field, "unknown")).lower() for r in pred]
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        field_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        results[field] = {
            "f1": round(field_f1, 4),
            "precision": round(report.get("macro avg", {}).get("precision", 0.0), 4),
            "recall": round(report.get("macro avg", {}).get("recall", 0.0), 4),
        }
        f1_scores.append(field_f1)

    results["macro_f1"] = round(float(np.mean(f1_scores)), 4)
    return results


def run_extraction_evaluation() -> dict:
    gold_df = pd.read_csv(GOLD_PATH)
    pred_df = pd.read_csv(PIPELINE_PATH)
    merged = gold_df.merge(pred_df, on="id", suffixes=("_gold", "_pred"))

    gold = [{f: merged.iloc[i][f"{f}_gold"] for f in CATEGORICAL_FIELDS} for i in range(len(merged))]
    pred = [{f: merged.iloc[i][f"{f}_pred"] for f in CATEGORICAL_FIELDS} for i in range(len(merged))]

    results = compute_field_f1(gold, pred)

    rows = [{"field": f, **v} for f, v in results.items() if f != "macro_f1"]
    rows.append({"field": "MACRO AVG", "f1": results["macro_f1"], "precision": None, "recall": None})
    results_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(results_df.to_string(index=False))
    print(f"\nHeadline macro-F1: {results['macro_f1']:.4f}")
    return results


if __name__ == "__main__":
    run_extraction_evaluation()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Run the evaluations (after Task 9 is done)**

```bash
python -m evaluation.evaluate_asr
python -m evaluation.evaluate_extraction
```

Record the headline numbers: Overall WER and Macro-F1. These fill the `[X]` placeholders in the abstract.

- [ ] **Step 7: Commit**

```bash
git add evaluation/evaluate_asr.py evaluation/evaluate_extraction.py tests/test_evaluate.py
git commit -m "feat: WER and F1 evaluation modules with tests"
```

---

## Task 11: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: Implement `dashboard/app.py`**

```python
"""
VoiceTrace Streamlit dashboard.
Displays pipeline results on a map and in a table.

Run: streamlit run dashboard/app.py
"""
import pandas as pd
import streamlit as st
from pathlib import Path

RESULTS_PATH = Path("data/synthetic/pipeline_results.csv")

st.set_page_config(page_title="VoiceTrace — Ghana Injury Dashboard", layout="wide")
st.title("VoiceTrace — Ghana Injury Surveillance Dashboard")
st.caption("Synthetic evaluation dataset | n=120 | Twi-language NLP pipeline")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter")
injury_types = ["All"] + sorted(df["injury_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Injury type", injury_types)
selected_severity = st.sidebar.multiselect(
    "Severity", df["severity"].dropna().unique().tolist(),
    default=df["severity"].dropna().unique().tolist()
)

filtered = df.copy()
if selected_type != "All":
    filtered = filtered[filtered["injury_type"] == selected_type]
if selected_severity:
    filtered = filtered[filtered["severity"].isin(selected_severity)]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records", len(filtered))
col2.metric("Road traffic (RTA)", len(filtered[filtered["injury_type"] == "rta"]))
col3.metric("Severe injuries", len(filtered[filtered["severity"] == "severe"]))
col4.metric("Geocoded", len(filtered[filtered["lat"].notna()]))

# Map
geo = filtered[filtered["lat"].notna() & filtered["lng"].notna()][["lat", "lng"]].copy()
geo.columns = ["lat", "lon"]
if not geo.empty:
    st.subheader("Injury locations")
    st.map(geo)
else:
    st.info("No geocoded records for current filter.")

# Table
st.subheader("Records")
display_cols = ["id", "injury_type", "mechanism", "severity", "body_region",
                "victim_sex", "victim_age_group", "location_description"]
st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

# Raw transcript expander
with st.expander("View ASR transcripts and translations"):
    st.dataframe(
        filtered[["id", "asr_transcript", "translated_en"]].reset_index(drop=True),
        use_container_width=True
    )
```

- [ ] **Step 2: Run the dashboard**

```bash
streamlit run dashboard/app.py
```

Expected: browser opens at `http://localhost:8501` showing the map, metrics, and table.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit dashboard with map, filters, and record table"
```

---

## Task 12: Run All Tests and Verify End-to-End

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS. Current tests cover:
- `test_extract.py` — 6 tests
- `test_pipeline.py` — 2 tests
- `test_evaluate.py` — 5 tests

- [ ] **Step 2: Verify evaluation results exist**

```bash
ls evaluation/results/
```

Expected: `asr_results.csv` and `extraction_results.csv` both present.

- [ ] **Step 3: Print headline numbers**

```bash
python -c "
import pandas as pd
asr = pd.read_csv('evaluation/results/asr_results.csv')
ext = pd.read_csv('evaluation/results/extraction_results.csv')
macro_row = ext[ext['field'] == 'MACRO AVG']
print('=== HEADLINE NUMBERS ===')
print(f'ASR overall WER: {asr.iloc[0][\"overall_wer\"]}')
print(f'Extraction macro-F1: {macro_row[\"f1\"].values[0]}')
"
```

These are your abstract `[X]` values.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: verify all tests pass, evaluation results confirmed"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `.env` with both API keys | Task 1 |
| `ghana_injury_distributions.json` locked | Task 1 |
| `prompts/extraction_prompt.txt` | Task 1 |
| Thin slice first / API verification | Task 3 |
| `pipeline/asr.py` — `transcribe()` | Task 4 |
| `pipeline/translate.py` — `translate()` | Task 4 |
| `pipeline/geocode.py` — `geocode()` | Task 4 |
| `pipeline/extract.py` — `extract()` | Task 2 |
| `pipeline/pipeline.py` — orchestrator | Task 5 |
| `data_gen/generate_narratives.py` | Task 6 |
| `data_gen/translate_to_twi.py` | Task 7 |
| `data_gen/tts.py` | Task 7 |
| Batch pipeline run | Task 8 |
| Gold annotation (manual) | Task 9 |
| `evaluation/evaluate_asr.py` — WER | Task 10 |
| `evaluation/evaluate_extraction.py` — F1 | Task 10 |
| `dashboard/app.py` — Streamlit | Task 11 |
| 0.5s delays between API calls | Tasks 7, 8 |
| Retry policy: one retry then log + continue | Tasks 4 (in module error handling) |
| `data/raw/` gitignored | Task 1 |
| Solo annotation with 2-pass consistency check | Task 9 |
| Definition of done: 120 narratives, 120 wav, gold annotations, eval results, dashboard | Task 12 |

No gaps found.
