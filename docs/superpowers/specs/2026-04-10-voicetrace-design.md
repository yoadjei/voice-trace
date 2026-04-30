# VoiceTrace — Design Specification
**Date:** 2026-04-10 (updated 2026-04-12)
**Author:** Yaw Osei
**Conference:** KNUST Centre for Injury Prevention and Research, May 5–6 2026
**Abstract deadline:** April 21, 2026

---

## 1. Objective

Build **VoiceTrace**: a voice-first multilingual NLP pipeline that receives an unstructured community injury voice report in a Ghanaian language, delivers immediate first aid guidance to the caller, identifies the nearest health facility via geocoded location, and generates a structured surveillance record automatically. Evaluate on 126 epidemiologically grounded synthetic narratives per language across three evaluation tracks. Submit abstract by April 21.

---

## 2. Scope & Cut Items

### What VoiceTrace IS (current contribution)
- A voice-first injury first-response and surveillance pipeline
- **Full voice end-to-end (ASR → translate → LLM extract → geocode)** for **Twi, Ga, Ewe, and Dagbani** using Khaya ASR v1 for cross-language parity
- **Six-language framework:** translation + TTS confirmed for all six (Twi, Ga, Ewe, Fante, Dagbani, Gurune); **Fante and Gurune** await Khaya ASR before full voice deployment
- Dual-output: spoken first aid guidance back to caller AND structured surveillance record as byproduct
- Nearest health facility identification via Nominatim geocoding + GHS facility list

### What VoiceTrace IS NOT claiming
- A live deployed system
- A real-time NAS dispatch system
- Full ASR-backed voice pipeline for Fante or Gurune today (translation/TTS-only until ASR ships)
- Fine-tuned ASR or NLP models
- ICD-10/ICD-11 automatic coding
- Real clinical data from KATH, Korle-Bu, or BRRI

### Permanently out of scope
- Fine-tuning any model
- Real patient data
- Production deployment with live users
- WhatsApp or app-based interface

---

## 3. Architecture

### 3.1 The Pipeline (Research Contribution)

```
[Voice input in Twi, Ga, Ewe, or Dagbani]
        ↓
   Khaya ASR v1       →  Local-language transcript
        ↓
Khaya Translate        →  English text
        ↓
 Claude (single call)  →  Structured JSON (6 fields) + First aid guidance text
        ↓                         ↓
Khaya Translate        →  Local response text      Geocode (Nominatim) → lat/lng
        ↓                                                    ↓
 Khaya TTS             →  Spoken response          GHS facility list → nearest facility
        ↓                         ↓
  Caller receives               Surveillance record logged (CSV/SQLite)
  spoken guidance               + facility name returned to caller
```

**ASR versioning (methodological):** Khaya exposes **ASR v1** for multiple languages (`tw`, `gaa`, `ee`, `dag`, …). A **higher-accuracy Twi-only v2** exists (punctuation and accuracy). This work uses **v1 for all four voice languages** so ASR WER and downstream extraction are comparable across languages; Twi-specific v2 is not mixed into the cross-language benchmark.

Two simultaneous outputs from one call:
- **Caller-facing:** Spoken first aid guidance in the caller’s language + nearest facility name
- **System-facing:** Structured injury record with geocoordinates

### 3.2 Six-language framework

- **Full voice (confirmed):** Twi, Ga, Ewe, Dagbani — ASR v1 + translate + extract + TTS.
- **Translation + TTS only (ASR pending):** Fante, Gurune — same downstream stages once reference text is available; synthetic evaluation uses text/translation tracks (round-trip, extraction from translated text, κ).
- Paper claim: *"The pipeline is fully voice-driven end-to-end across four Ghanaian languages — Twi, Ga, Ewe, and Dagbani — with translation and TTS confirmed for Fante and Gurune, establishing a six-language deployment-ready framework as ASR support for those languages matures."*

### 3.3 Geocoding Implementation

```
location_description (free text from Claude extraction)
        ↓
Nominatim: https://nominatim.openstreetmap.org/search?q={location}&countrycodes=gh&format=json
        ↓
lat/lng coordinates
        ↓
Haversine distance against GHS national facility list (download from HDX)
        ↓
Nearest facility: name, type, distance (km)
```

**Action item before writing accuracy claims:** Test Nominatim on 20 sample location strings from narratives. If failure rate > 30%, flag Ghana Post GPS as recommended replacement in future work section.

Paper claim (exact wording): *"Caller-reported location descriptions are geocoded via OpenStreetMap Nominatim and matched against the Ghana Health Service national facility database using Haversine distance to identify the nearest health facility, establishing the technical foundation for real-time dispatch integration."*

---

## 4. File Structure

```
voicetrace/
├── .env                          # KHAYA_API_KEY, ANTHROPIC_API_KEY (gitignored)
├── requirements.txt
├── data/
│   ├── raw/                      # GBD, WHO, DHS CSVs (gitignored)
│   ├── distributions/
│   │   └── ghana_injury_distributions.json   # Locked sampling weights
│   └── synthetic/
│       ├── narratives_en.csv          # 126 Claude-generated English narratives
│       ├── narratives_twi.csv         # Khaya-translated Twi
│       ├── narratives_{lang}.csv      # Per-language (ewe, ga, fante, dagbani, gurene)
│       ├── narratives_all_langs.csv   # Merged all-language file
│       ├── audio/{twi,ga,ewe,...}/    # 126 .wav per language (Khaya TTS)
│       ├── asr_transcripts.csv        # ASR v1 on Twi audio (legacy path)
│       ├── asr_transcripts_{lang}.csv # other languages (e.g. _ga)
│       ├── translations_en.csv        # Twi: ASR → English
│       ├── translations_en_{lang}.csv # same for other voice languages
│       └── gold_annotations.csv      # Ground truth (126 rows, 6 fields, two-pass)
├── pipeline/
│   ├── asr.py                    # wav → text (Khaya ASR v1)
│   ├── lang_config.py            # Khaya codes + full-voice language set
│   ├── translate.py              # local ↔ English (Khaya) + Claude en→local
│   ├── extract.py                # English → structured JSON + first aid (Claude)
│   ├── geocode.py                # location string → lat/lng → nearest GHS facility
│   └── pipeline.py               # Orchestrator: all stages end-to-end
├── data_gen/
│   ├── generate_narratives.py    # Claude: produces narratives_en.csv
│   ├── translate_all_langs.py    # Khaya: produces per-language CSVs
│   └── tts.py                    # Khaya TTS: produces audio/
├── evaluation/
│   ├── evaluate_asr.py           # Track 1: WER via jiwer
│   ├── evaluate_extraction.py    # Track 1+2: F1 per field via sklearn
│   ├── evaluate_translation.py   # Track 2: BERTScore + BLEU round-trip fidelity
│   ├── evaluate_consistency.py   # Track 3: Cohen's Kappa cross-language
│   └── results/                  # Output tables and logs
├── dashboard/
│   └── app.py                    # Streamlit: map + table view
├── prompts/
│   └── extraction_prompt.txt     # Claude extraction + first aid prompt (versioned)
└── paper/
    ├── voicetrace_paper.tex
    ├── abstract.docx
    └── figures/
        └── architecture_diagram.png
```

---

## 5. Data & Synthetic Generation

### 5.1 Corpus

- **126 English narratives** generated by Claude, stratified by injury distribution
- **6 language translations** via Khaya: Twi, Ga, Ewe, Fante, Dagbani, Gurene
- **126 .wav files per language** via Khaya TTS; **full ASR→extract evaluation** on Twi, Ga, Ewe, Dagbani
- **Total corpus:** 756 narratives across all languages

### 5.2 Sampling Distribution

Locked from published Ghana epidemiological data.

| Field | Distribution | Source |
|---|---|---|
| Injury type | RTA 39.1%, Fall 19.7%, Assault 12.0%, Burn 8.5%, Drowning 4.3%, Occupational 16.4% | Boateng et al. 2019 |
| Sex | Male 67.8%, Female 32.2% | DHS 2008/2022 |
| Age group | <15: 18%, 15–34: 45%, 35–54: 27%, 55+: 10% | DHS 2008/2022 |
| Severity | Minor 52%, Moderate 31%, Severe 17% | Opoku et al. 2025 |
| Body region | Head/neck 38%, Limb 44%, Trunk 18% | Boateng et al. 2019 |
| Location type | Highway 41%, Urban road 35%, Home 24% | Mesic et al. 2024 |

### 5.3 Gold Annotation Schema (6 fields)

```json
{
  "injury_type": "rta | fall | assault | burn | drowning | occupational | unknown",
  "severity": "minor | moderate | severe | unknown",
  "body_region": "head_neck | upper_limb | lower_limb | trunk | multiple | unknown",
  "victim_sex": "male | female | unknown",
  "victim_age_group": "child | youth | adult | elderly | unknown",
  "location_description": "free text, max 15 words"
}
```

Note: `mechanism` removed from evaluation schema — retained in Claude output but not in headline F1 (free-text, not scoreable categorically).

### 5.4 Annotation Protocol

Two independent passes on all 126 English source narratives with ≥24 hours between passes. Compute inter-annotator Cohen's Kappa per field. Resolve disagreements. This single gold standard applies across all six language evaluations since extraction is compared against English source labels.

---

## 6. Claude Extraction + First Aid Prompt

Stored in `prompts/extraction_prompt.txt`. Single call that does both tasks.

```
You are an injury surveillance and first response assistant for the Ghana health system.
You will receive a community injury report, translated to English from a Ghanaian language.
The report may be informal, incomplete, or imprecise.

Do two things:

1. Extract the following structured fields. Use "unknown" if a field cannot be determined.
Return as JSON under the key "extraction":
{
  "injury_type": one of [rta, fall, assault, burn, drowning, occupational, unknown],
  "severity": one of [minor, moderate, severe, unknown],
  "body_region": one of [head_neck, upper_limb, lower_limb, trunk, multiple, unknown],
  "victim_sex": one of [male, female, unknown],
  "victim_age_group": one of [child, youth, adult, elderly, unknown],
  "location_description": place name or description (max 15 words) or "unknown"
}

2. Generate a brief first aid guidance message (3–5 sentences) appropriate to the
injury type and severity. Frame as immediate support pending professional care.
Do not diagnose. Do not prescribe medication. Do not speculate beyond what is reported.
Every response must end with: "Go to the nearest health facility immediately."
Return as a string under the key "first_aid".

Return ONLY valid JSON: {"extraction": {...}, "first_aid": "..."}

Report: """
{TRANSLATED_TEXT}
"""
```

---

## 7. Evaluation — Three Tracks

### Track 1 — Full End-to-End Pipeline (Twi, Ga, Ewe, Dagbani)

| Step | Input | Output | Metric | Tool |
|---|---|---|---|---|
| ASR | Local .wav | Khaya ASR v1 transcript | WER overall + by injury type | jiwer |
| Extraction post-ASR | ASR transcript → English → Claude | Structured JSON | F1 per field | sklearn |
| End-to-end compounding | Clean text pipeline vs ASR pipeline | Final JSON vs gold | **Δ F1** (text input vs ASR input) per language | sklearn |

Highlight **Δ F1**: measures how much ASR degrades extraction vs gold relative to running extraction on clean English or round-trip text — comparable across languages and interpretable with WER (robustness vs proportional error).

### Track 2 — Translation Round-Trip Fidelity + Extraction (all 6 languages)

| Step | Input | Output | Metric | Tool |
|---|---|---|---|---|
| Translation fidelity | English → local lang → back to English | Round-trip English | BERTScore, BLEU vs original | bert-score, sacrebleu |
| Extraction from translated | English → local → back → Claude | Structured JSON vs gold | F1 per field | sklearn |

Research question: how much information survives the translation round-trip before Claude sees it?

### Track 3 — Cross-Language Extraction Consistency (all 6 languages)

| Step | Input | Output | Metric | Tool |
|---|---|---|---|---|
| Consistency | Same 126 scenarios through all 6 language pipelines | 6 extraction outputs per scenario | Cohen's Kappa across language pairs per field | sklearn |

Research question: does translation introduce systematic extraction bias, and is it language-specific or consistent?

### Full Evaluation Table

| Track | Languages | Metric | Tool |
|---|---|---|---|
| ASR quality | Twi, Ga, Ewe, Dagbani | Word Error Rate | jiwer |
| Extraction accuracy post-ASR | Twi, Ga, Ewe, Dagbani | Precision, Recall, F1 per field | sklearn |
| End-to-end compounding error | Twi, Ga, Ewe, Dagbani | Δ F1 (text vs ASR input) | sklearn |
| Translation round-trip fidelity | All 6 | BERTScore, BLEU | bert-score, sacrebleu |
| Extraction from translated text | All 6 | F1 per field | sklearn |
| Cross-language consistency | All 6 | Cohen's Kappa across language pairs | sklearn |

---

## 8. API Integration & Error Handling

### Khaya
- Rate limit: 10 requests/minute (free tier), 100 calls/month
- 6s delay between requests enforced in all scripts
- 429 handler: exponential backoff (30s × attempt), 3 retries
- 403 handler: stop immediately, log quota message
- All outputs saved to disk after each row — all jobs resumable

### Claude
- Model: `claude-sonnet-4-6`
- `max_tokens=600`
- Response must be valid JSON with both `extraction` and `first_aid` keys
- Invalid JSON or missing keys: log, record `"unknown"` for all extraction fields, continue

### Secrets
- `.env`: `KHAYA_API_KEY`, `ANTHROPIC_API_KEY`
- `.gitignore`: `.env`, `data/raw/`

---

## 9. Datasets in Hand

| Dataset | Purpose |
|---|---|
| IHME GBD Ghana (filtered) | Introduction burden statistics |
| WHO Ghana Road Safety CSV (HDX) | Road fatality figures |
| WHO Ghana GHE CSV (HDX) | Headline mortality |
| Ghana DHS 2022 | Demographic grounding |
| Ghana DHS 2008 | Injury prevalence module |
| Boateng et al. 2019 (Korle-Bu) | Primary sampling distribution source |
| Opoku et al. 2025 | Pooled prevalence and severity |
| Mesic et al. 2024 | Road crash hotspot analysis / location distribution |
| GHS National Facility List (HDX) | Geocoding nearest facility — **download this week** |

---

## 10. Known Risks

| Risk | Mitigation |
|---|---|
| Khaya ASR WER high on injury vocabulary | Expected 30–50% WER — frame as a finding, not a failure |
| Claude extraction inconsistent on ambiguous inputs | Add few-shot examples if zero-shot F1 < 0.70 on first 10 |
| TTS→ASR evaluation is circular | Acknowledge as lower bound; natural speech WER would be higher |
| Nominatim fails on informal location descriptions | Test on 20 samples first; flag Ghana Post GPS as backup in future work |
| Reviewer pushback on synthetic data | Cite Nyamawe & Shao 2026; grounding in Boateng distributions is the defence |
| Khaya API quota running low | 100 calls/month free tier; use separate keys per language |

---

## 11. Week-by-Week Timeline

| Week | Dates | Focus |
|---|---|---|
| 2 | Apr 14–21 | Complete audio for four voice langs; `run_pipeline_batch.py --lang …`; gold annotation; **submit abstract Apr 21** |
| 3 | Apr 21–27 | Complete annotation; run all 3 evaluation tracks; dashboard; architecture diagram |
| 4 | Apr 28–May 4 | Full paper draft by Apr 30; review; revisions by May 2; slides; practice talk |
| **May 5** | — | **Present** |

---

## 12. Definition of Done

1. 126 `.wav` files per voice language in `data/synthetic/audio/{twi,ga,ewe,dagbani}/`
2. `gold_annotations.csv` fully populated (126 rows, 6 fields, two-pass, kappa computed)
3. Track 1 evaluation complete: WER, post-ASR F1, and Δ F1 **for all four voice languages**
4. Track 2 evaluation complete: BERTScore/BLEU and F1 for all 6 languages
5. Track 3 evaluation complete: Kappa table across language pairs
6. GHS facility list downloaded; Nominatim tested on 20 location samples
7. `app.py` runs locally with map + table
8. Abstract [X] placeholders filled with real numbers
9. Full paper draft in `paper/voicetrace_paper.tex`
10. Architecture diagram in `paper/figures/architecture_diagram.png`
