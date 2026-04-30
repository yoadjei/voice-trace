# Ghana Injury NLP Paper — Complete Research Roadmap
**Conference:** KNUST Centre for Injury Prevention and Research 2-Day Conference  
**Date:** May 5–6, 2026 | Abstract deadline: April 21, 2026  
**Format:** Oral presentation (assume full paper required)  
**Team:** 4 authors  
**Status:** Pre-research — no clinical data, no prior codebase

---

## PART 0 — FEASIBILITY AUDIT (READ THIS FIRST)

Before touching anything else, these are the things that looked good on paper but are outrageous given your constraints. They are cut permanently.

### CUT — Fine-tuning any ASR or NLP model on Ghanaian injury text
**Why it's tempting:** Domain adaptation of Khaya ASR or ABENA/AfroXLMR to medical vocabulary would improve extraction accuracy.  
**Why it's cut:** Requires annotated Twi/Ewe injury speech or text corpora that do not exist. Creating them from scratch takes 3–6 months minimum. You have 6 weeks.

### CUT — Real clinical data from KATH, Korle-Bu, or BRRI
**Why it's tempting:** Grounds all claims in real Ghanaian patient data.  
**Why it's cut:** Data access agreements, institutional ethics approval, and administrative sign-off at these institutions take 8–12 weeks minimum. You are past that window.

### CUT — A working deployed system with real users
**Why it's tempting:** A live demo would be impressive for an oral presentation.  
**Why it's cut:** Building, testing, and stabilising a production system on top of everything else is too much. What you build is a working prototype pipeline that you can demo locally or on a controlled dataset. That is sufficient.

### RESCOPED — Full voice evaluation on four languages (Twi, Ga, Ewe, Dagbani)
**What changed:** Khaya ASR v1 supports these four; the contribution is now **parallel** Track 1 metrics (WER, post-ASR F1, Δ F1) across all four, using v1 everywhere for parity (optional Twi-only ASR v2 is out of scope for the cross-language benchmark). Fante and Gurune stay translation/TTS-only until ASR exists. Poor WER on a language is a **result**, not a reason to drop the language — frame alongside extraction robustness.

### CUT — ICD-10/ICD-11 automatic coding as a deliverable
**Why it's tempting:** Medical coding is a high-impact NLP task.  
**Why it's cut:** Accurate ICD coding from free-text requires either labelled training data or extensive prompt engineering validated against a gold standard. Without real clinical notes as a validation set, any claimed accuracy is unverifiable. Reframe to structured field extraction (injury type, mechanism, severity, body region, demographics, location) which is a defensible and original contribution on its own.

### CUT — Twi TTS for community outreach audio messages as a paper deliverable
**Why it's tempting:** Immediately practical.  
**Why it's cut:** It is a separate product, not a research contribution. Mention it in future work.

---

### WHAT REMAINS — THE DEFENSIBLE SCOPE

A proposed and demonstrated **end-to-end multilingual injury surveillance pipeline** (full voice on **Twi, Ga, Ewe, Dagbani**; text/translation tracks on six languages) that:
1. Accepts a voice report in a supported language (simulated via Khaya TTS from synthetic narratives)
2. Transcribes it via Khaya ASR v1
3. Translates to English via Khaya translation API
4. Extracts structured injury fields via Claude API (zero-shot prompt)
5. Outputs a structured record (JSON → CSV → dashboard)

**Validated on:** epidemiologically grounded synthetic Twi injury narratives (n ≈ 100–150), with extraction accuracy benchmarked against manual annotation.  
**Claims:** Architecture novelty (first multilingual NLP injury surveillance pipeline for any African language), feasibility demonstration (ASR transcription quality, extraction F1 per field), and clear deployment roadmap.

This is honest, original, technically grounded, and completable in 6 weeks with 4 people.

---

## PART 1 — THE PAPER

### Proposed title
**"VoiceTrace: A Voice-First Multilingual NLP Pipeline for Community Injury Surveillance in Ghana"**

Alternatives if the team prefers:
- *SentinelVoice: Multilingual Injury Surveillance via Speech and LLMs in Ghana*
- *InjuryLens: An NLP Framework for Community-Reported Injury Detection in Ghanaian Languages*
- *RoadEcho / CrashTrace / InjuryLog* — shorter, punchier options

Name criteria: easy to say, not culturally loaded, describes what it does, not an acronym. **VoiceTrace** is the recommendation.

---

### Abstract (submit by April 21)

> Injuries constitute a critical and under-surveilled public health crisis in Ghana, accounting for an 18% pooled prevalence of unintentional injuries and nearly 3,000 road traffic fatalities in 2025 alone. Existing surveillance infrastructure — fragmented across paper-based hospital records, police crash reports, and aggregate health information systems — captures fewer than 5% of community-level injuries, and no computational tool exists for injury surveillance in any Ghanaian language. We propose and demonstrate VoiceTrace, a voice-first multilingual natural language processing pipeline that transforms unstructured community voice reports in Ghanaian languages into structured, geocoded injury surveillance records. The pipeline integrates GhanaNLP's Khaya automatic speech recognition and translation API with Anthropic's Claude large language model for zero-shot structured extraction of injury type, mechanism, severity, body region, demographics, and location. We evaluate the pipeline on 120 epidemiologically grounded synthetic Twi-language injury narratives derived from published Ghanaian epidemiological distributions, achieving extraction F1 scores of [X] across six structured fields, with ASR word error rates of [X]% on injury-domain speech. VoiceTrace requires no domain-specific training data, operates via standard API calls feasible on low-resource hardware, and is architecturally extensible to Ewe, Ga, and Dagbani. To our knowledge, this is the first NLP pipeline for injury surveillance in any African language. We discuss deployment pathways through Ghana's National Ambulance Service, community health worker networks, and USSD/WhatsApp voice interfaces, and identify priority requirements for real-world validation.

**Note:** Fill [X] values after running experiments. Keep abstract under 300 words for submission. This draft is ~270 words.

---

### Paper structure (target: 6–8 pages, conference format)

**1. Introduction** (~600 words)
- Ghana's injury burden: statistics, economic cost, surveillance gap
- The language barrier in health reporting: why English-only systems exclude the majority
- GhanaNLP/Khaya as an enabling technology
- Paper contributions (list 3–4 bullet points explicitly)
- Paper organisation

**2. Related Work** (~700 words)
- Injury surveillance in Ghana: epidemiology and data systems
- ML/NLP for injury data globally (Vallmuur 2015, TraumaICDBERT, GPT-4 for ED injury notes)
- NLP for African public health (JMIR 2025 scoping review)
- GhanaNLP and Khaya: capabilities and prior applications
- Gap statement: no NLP for injury in any African language

**3. System Architecture** (~800 words + 1 architecture diagram)
- Overview of the 5-stage pipeline
- Stage 1: Voice input interface design (WhatsApp/USSD/call)
- Stage 2: Khaya ASR — model details, supported languages, limitations
- Stage 3: Khaya translation — English output
- Stage 4: Claude extraction — prompt design, output schema, zero-shot rationale
- Stage 5: Geocoding + dashboard
- Design decisions and tradeoffs

**4. Synthetic Data Generation** (~500 words)
- Rationale for synthetic validation
- Source distributions (IHME GBD, Mesic 2024, Korle-Bu BMC 2019, BRRI statistics)
- Generation procedure: structured sampling → narrative template → Twi translation
- Dataset statistics: n=120, injury type distribution, severity distribution, demographic distribution

**5. Evaluation** (~700 words)
- ASR evaluation: WER on injury-domain Twi speech (TTS-generated audio)
- Extraction evaluation: precision, recall, F1 per field
- End-to-end evaluation: final structured record accuracy
- Error analysis: common failure modes (ASR noise, translation drift, extraction hallucination)

**6. Discussion** (~500 words)
- What the results mean for deployment feasibility
- Limitations: synthetic data, Twi-only evaluation, no real-world validation
- Deployment pathways: NAS integration, CHW mobile tools, DHIMS-2 pipeline
- Ethical considerations: privacy, consent, bias in injury reporting

**7. Conclusion and Future Work** (~300 words)
- Summary of contributions
- Priority next steps: real data access via BRRI/KATH, multi-language evaluation, community pilot

**References** — target 25–35 citations

---

## PART 2 — EVERYTHING YOU NEED TO READ

Read in this order. Do not read all of them — read the ones marked REQUIRED in full. Skim the BACKGROUND ones. Note the CITE ONLY ones — you cite them for credibility without needing to read the full paper.

### REQUIRED (read in full, ~2–3 hours total)

1. **Opoku et al. (2025) — Examining the burden of unintentional injuries in Ghana: A systematic review and meta-analysis**  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12481059/  
   *Why:* Your primary epidemiological source. All your burden statistics come from here. Read every table.

2. **Mesic et al. (2024) — Identifying emerging hotspots of road traffic injury severity using spatiotemporal methods: longitudinal analyses on major roads in Ghana from 2005 to 2020**  
   BMC Public Health: https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-024-18915-x  
   *Why:* Best existing computational work on Ghana road injuries. You build directly on this gap.

3. **Omiye et al. (2025) — Natural Language Processing Technologies for Public Health in Africa: Scoping Review**  
   JMIR: https://www.jmir.org/2025/1/e68720  
   *Why:* Establishes that NLP for injury in Africa does not exist. Your gap statement depends on this.

4. **Lorenzoni et al. (2024) — Use of a Large Language Model to Identify and Classify Injuries With Free-Text Emergency Department Data**  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11134210/  
   *Why:* Closest global precedent to what you are doing. GPT-4 on 40,031 ED records. Read carefully — your methodology mirrors and extends this.

5. **GhanaNLP documentation and Khaya API reference**  
   https://ghananlp.org | https://ghananlp.org/tech/2021/07/23/speech-recognition.html  
   *Why:* You need to understand exactly what Khaya can and cannot do before you claim it in a paper. Check: supported languages, WER benchmarks they publish, API rate limits.

6. **Vallmuur (2015) — Machine learning approaches to analysing textual injury surveillance data: a systematic review**  
   Find via: QUT ePrints https://eprints.qut.edu.au/82722/  
   *Why:* Foundational paper for the NLP-injury surveillance field. Gives you the intellectual lineage for your approach.

### BACKGROUND (skim — abstract + introduction + results)

7. **Wahab & Jiang (2019) — A comparative study on machine learning based algorithms for prediction of motorcycle crash severity**  
   PLOS ONE: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0214966  
   *Why:* The only ML study on Ghana injury data. You position yourself as the next step beyond this.

8. **Yankson et al. (2012) — Reporting on road traffic injury: content analysis of injuries and prevention opportunities in Ghanaian newspapers**  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3271730/  
   *Why:* Manual newspaper analysis — exactly what you are automating. Perfect precedent to cite.

9. **Boateng et al. (2019) — Epidemiology of injuries presenting to the accident centre of Korle-Bu Teaching Hospital, Ghana**  
   BMC Emergency Medicine: https://bmcemergmed.biomedcentral.com/articles/10.1186/s12873-019-0252-3  
   *Why:* Source of your synthetic data distributions (injury type, age, sex, mechanism proportions).

10. **Nyamawe & Shao (2026) — On the use of synthetic data for healthcare AI in Africa**  
    SAGE Journals: https://journals.sagepub.com/doi/10.1177/20552076261418897  
    *Why:* Justifies your synthetic data approach. Cite this to pre-empt reviewer pushback.

11. **Choi et al. (2024) — TraumaICDBERT: A Natural Language Processing Model for Trauma Injury ICD-10 Coding**  
    Annals of Surgery: https://journals.lww.com/annalsofsurgery/abstract/9900/traumaicdbert  
    *Why:* State-of-the-art precedent for NLP+injury coding. Positions VoiceTrace in the right technical context.

12. **Nsubuga et al. (2025) — Enhancing trauma triage in low-resource settings using machine learning**  
    BMC Emergency Medicine: https://link.springer.com/article/10.1186/s12873-025-01175-2  
    *Why:* Closest African ML trauma paper. Uganda, not Ghana, but directly comparable context.

### CITE ONLY (abstract only, cite for credibility)

13. **IHME GBD 2023** — https://ghdx.healthdata.org/gbd-2023 — your Ghana injury burden figures
14. **WHO Ghana Road Safety Data** — https://data.who.int/countries/288
15. **Masakhane NLP** — https://www.masakhane.io — positions your work in the African NLP ecosystem
16. **Rwanda Nature Medicine LLM trial (2025)** — https://www.nature.com/articles/s41591-025-03815-3
17. **Boateng et al. (2020) — Community Causes of Death, Central Region Ghana** — Wiley, drowning data
18. **BRRI crash statistics** — cite NRSA annual reports for official crash numbers

---

## PART 3 — THE BUILD PLAN

### Tech stack

| Component | Tool | Access |
|-----------|------|--------|
| Voice recording simulation | Khaya TTS API | Existing Khaya API key |
| ASR transcription | Khaya ASR API | Same key |
| Translation (Twi → English) | Khaya translation API | Same key |
| Structured extraction | Claude API (claude-sonnet-4-6) | Existing Claude API key |
| Geocoding | OpenStreetMap Nominatim (free) | No key needed |
| Data storage | SQLite / CSV | No setup |
| Dashboard | Streamlit (Python) | pip install |
| Evaluation | Python: spaCy, pandas, sklearn | pip install |
| Version control | GitHub (private repo) | Existing |

**No paid infrastructure required beyond the APIs you already have.**

---

### Repository structure

```
voicetrace/
├── README.md
├── requirements.txt
├── data/
│   ├── synthetic/
│   │   ├── raw_narratives_en.csv        # English source narratives
│   │   ├── raw_narratives_twi.csv       # Twi translations
│   │   ├── audio/                       # TTS-generated .wav files
│   │   └── gold_annotations.csv         # Manual annotation for evaluation
│   └── distributions/
│       └── ghana_injury_distributions.json  # Source epidemiological stats
├── pipeline/
│   ├── tts.py                           # Khaya TTS: text → audio
│   ├── asr.py                           # Khaya ASR: audio → Twi text
│   ├── translate.py                     # Khaya: Twi → English
│   ├── extract.py                       # Claude API: English → structured JSON
│   ├── geocode.py                       # Location string → lat/lng
│   └── pipeline.py                      # End-to-end runner
├── evaluation/
│   ├── annotator_guide.md
│   ├── evaluate_asr.py                  # WER calculation
│   ├── evaluate_extraction.py           # Precision/recall/F1 per field
│   └── results/
├── dashboard/
│   └── app.py                           # Streamlit dashboard
├── prompts/
│   └── extraction_prompt.txt            # Claude prompt template
└── paper/
    ├── voicetrace_paper.tex (or .docx)
    └── figures/
        └── architecture_diagram.png
```

---

### Synthetic data generation procedure

**Step 1 — Define the distribution (from published Ghana data)**

Draw from these proportions for 120 narratives:

| Field | Distribution | Source |
|-------|-------------|--------|
| Injury type | RTA 39.1%, Fall 19.7%, Assault 12.0%, Burn 8.5%, Drowning 4.3%, Occupational 16.4% | Boateng et al. 2019 (Korle-Bu) |
| Sex | Male 67.8%, Female 32.2% | Boateng et al. 2019 |
| Age group | <15: 18%, 15–34: 45%, 35–54: 27%, 55+: 10% | Opoku et al. 2025 |
| Severity | Minor 52%, Moderate 31%, Severe 17% | Opoku et al. 2025 |
| Body region | Head/neck 38%, Limb 44%, Trunk 18% | Korle-Bu / Mesic 2024 |
| Location type | Highway 41%, Urban road 35%, Home 24% | BRRI / Opoku 2025 |

**Step 2 — Generate English narratives**

Use Claude to generate 120 naturalistic English injury report narratives following the above distribution. Each narrative should be 2–5 sentences, written as if spoken by a community member reporting an injury they witnessed or experienced. Vary vocabulary, specificity, and register. Save as `raw_narratives_en.csv`.

Prompt template for generation:
```
Generate a naturalistic spoken injury report (2-5 sentences) as if a community member in 
Ghana is reporting to a health hotline. The report should describe:
- Injury type: [RTA / fall / burn / etc.]
- Victim: [age group], [sex]
- Severity: [minor / moderate / severe]
- Body region affected: [head / limb / trunk]
- Location: [highway near X / home in Y / etc.]
Write in simple, informal English as if the caller is not medically trained. 
Vary sentence structure. Do not include medical jargon.
```

**Step 3 — Translate to Twi**

Use Khaya translation API to translate all 120 English narratives to Twi. Save as `raw_narratives_twi.csv`. Manually review 20–30 for gross translation errors. Note error patterns.

**Step 4 — Generate TTS audio**

Feed Twi narratives into Khaya TTS to produce .wav audio files. This simulates community voice input. Save to `data/synthetic/audio/`.

**Step 5 — Manual gold annotation**

Two annotators (from your team) independently annotate each English narrative against the 6-field schema:
- `injury_type`: {rta, fall, assault, burn, drowning, occupational, unknown}
- `mechanism`: free text (e.g. "pedestrian hit by vehicle", "fall from height")
- `severity`: {minor, moderate, severe, unknown}
- `body_region`: {head_neck, upper_limb, lower_limb, trunk, multiple, unknown}
- `victim_sex`: {male, female, unknown}
- `victim_age_group`: {child, youth, adult, elderly, unknown}
- `location_description`: free text

Compute inter-annotator agreement (Cohen's kappa). Resolve disagreements. Save as `gold_annotations.csv`.

---

### Extraction prompt (Claude)

This is the core of the system. Draft carefully — prompt quality determines extraction quality.

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

Test this prompt on 10 examples manually before running the full evaluation. Iterate. Document what breaks.

---

### Evaluation metrics

**ASR evaluation**
- Metric: Word Error Rate (WER) = (S + D + I) / N
- Compute on: Twi TTS audio → Khaya ASR transcription vs. original Twi text
- Tool: `jiwer` Python library (`pip install jiwer`)
- Report: overall WER, WER broken down by injury type (RTAs may have more vehicle-specific vocabulary)

**Extraction evaluation**
- Metric: Precision, Recall, F1 per field
- For categorical fields (injury_type, severity, etc.): exact match accuracy
- For free-text fields (mechanism, location_description): report separately, use qualitative analysis
- Tool: sklearn `classification_report`
- Report: macro-averaged F1 across categorical fields as headline number

**End-to-end evaluation**
- Run full pipeline: Twi text → TTS audio → ASR → translation → extraction
- Compare final structured output to gold annotation
- Report: how much ASR error propagates into extraction error (error compounding analysis)

---

### Division of labour (suggested)

| Person | Role | Deliverable |
|--------|------|-------------|
| You | Lead author, architecture, Claude extraction pipeline, paper writing | `extract.py`, `pipeline.py`, paper sections 1/3/6/7 |
| Co-author 2 | Data: synthetic generation + TTS/ASR pipeline | `tts.py`, `asr.py`, `translate.py`, `data/synthetic/` |
| Co-author 3 | Evaluation: annotation, WER/F1 computation, error analysis | `evaluate_asr.py`, `evaluate_extraction.py`, paper section 5 |
| Co-author 4 | Related work, dashboard, figures | `app.py`, architecture diagram, paper sections 2/4 |

---

## PART 4 — WEEK-BY-WEEK TIMELINE

You have from now (April 10) to May 5 — 25 days. The abstract is due April 21.

### Week 1 (April 10–16) — Reading + Setup + Abstract draft
- [ ] Read all REQUIRED papers (split among team)
- [ ] Set up GitHub repo with the folder structure above
- [ ] Verify Khaya API access: test TTS, ASR, and translation on 3–5 sample sentences
- [ ] Verify Claude API access: test extraction prompt on 5 hand-written English injury narratives
- [ ] Define exact Ghana injury distribution table (locked, no changes after this)
- [ ] Draft abstract — submit for team review by April 14
- [ ] Finalise paper title and author order

### Week 2 (April 14–20) — Data generation + Pipeline build
- [ ] Generate 120 English narratives using Claude (1–2 hours with good prompting)
- [ ] Translate to Twi via Khaya API
- [ ] Generate TTS audio for all 120
- [ ] Run Khaya ASR on all 120 audio files — record raw transcriptions
- [ ] Run translation on ASR output
- [ ] Run Claude extraction on translations
- [ ] Manual annotation: co-author 3 starts gold standard (annotate 60 each, cross-check)
- [ ] **Submit abstract by April 21**

### Week 3 (April 21–27) — Evaluation + Error analysis
- [ ] Complete gold annotation for all 120
- [ ] Run WER calculation
- [ ] Run extraction F1 calculation
- [ ] Error analysis: document top 5 failure modes
- [ ] Build Streamlit dashboard (basic: map + table view)
- [ ] Architecture diagram (draw.io or Lucidchart — export as PNG)

### Week 4 (April 28–May 4) — Paper writing + Polish
- [ ] Full paper draft complete by April 30
- [ ] Internal review: all 4 co-authors read and comment
- [ ] Revisions by May 2
- [ ] Prepare oral presentation slides (10–12 slides)
- [ ] Practice talk (aim for 15 minutes + 5 questions)
- [ ] Final checks: citations, figures, numbers consistent throughout

### May 5 — Present

---

## PART 5 — KNOWN RISKS AND MITIGATIONS

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Khaya ASR WER is too high on injury vocabulary | Medium | Expected WER 30–50% on domain-specific text — this is a RESULT not a failure. Frame as "baseline performance and identified limitation motivating domain adaptation." |
| Claude extraction is inconsistent on ambiguous narratives | Medium | Tighten the prompt. Add 3–5 few-shot examples in the prompt if zero-shot is unstable. |
| Khaya TTS → ASR evaluation is circular (we generated the audio ourselves) | High | Acknowledge this limitation explicitly in the paper. Argue it provides a lower bound on WER. Real-world WER would be higher due to speaker variation and background noise. |
| Translation introduces errors before extraction | Medium | Evaluate translation quality separately on 20 samples using back-translation. Report translation quality as a distinct pipeline stage. |
| Reviewer pushback on synthetic data | Medium | Cite Nyamawe & Shao 2026. Frame as standard practice for LMIC health AI where data access is structurally restricted. Emphasise the epidemiological grounding. |
| Team coordination breaks down under exam pressure | High | Lock the GitHub repo structure in Week 1. Each person owns their files. Do not do collaborative editing on paper sections — assign sections, merge at the end. |
| Conference asks for full paper and you only have a draft | Low-medium | Start writing in Week 2, not Week 4. The paper is not separate from the work — it runs in parallel. |

---

## PART 6 — WHAT TO SAY IN THE ORAL PRESENTATION

Structure the 15-minute talk as follows:

**0:00–2:00 — Open with the problem, not the solution**  
Show a single statistic: "2,949 Ghanaians died on roads in 2025. The system that is supposed to track this captures fewer than 5% of injuries. And it only works in English."

**2:00–5:00 — Why existing tools fail here**  
Two slides: (1) the surveillance gap, (2) the language barrier. Keep it tight. No tables — use icons and numbers.

**5:00–9:00 — VoiceTrace: what it is and how it works**  
Walk through the architecture diagram. Show each stage as a box with an arrow. On each box, say one sentence about what it does and one sentence about why you chose that tool. Play a 20-second audio clip of a TTS-generated Twi injury report, show the ASR transcription, show the Claude extraction output. This is your demo moment.

**9:00–12:00 — What we found**  
Three key numbers: ASR WER, extraction macro-F1, end-to-end accuracy. Show an error analysis table — what fails and why. Be honest. Audiences trust researchers who name their limitations before being asked.

**12:00–14:00 — Why this matters and what happens next**  
Show a map of Ghana with NAS coverage overlaid with injury hotspots. Say: this pipeline can run on a USSD interface with no smartphone required. It can feed directly into DHIMS-2. The next step is a community pilot with 20 health workers in [region].

**14:00–15:00 — Close**  
One sentence: "VoiceTrace is the first NLP system for injury surveillance in any African language. We built it here, at KNUST, with Ghanaian language tools. The data is local. The problem is local. The solution should be too."

---

## PART 7 — ADDITIONAL RESOURCES

### APIs and documentation
- Khaya API: https://ghananlp.org — get your key, check rate limits before batch processing 120 files
- Claude API: https://docs.anthropic.com — use `claude-sonnet-4-6`, set `max_tokens=500` for extraction
- jiwer (WER): `pip install jiwer` — https://github.com/jitsi/jiwer
- Nominatim geocoding: https://nominatim.org/release-docs/latest/api/Search/
- Streamlit: https://docs.streamlit.io

### Key datasets to download now
- IHME GBD Results Tool (Ghana, injuries, 1990–2023): https://www.healthdata.org/data-tools-practices/interactive-visuals/gbd-results
- Ghana Open Data — National Accident Statistics: https://data.gov.gh/dataset/national-accident-statistics-fatalities
- WHO Ghana indicators (HDX): https://data.humdata.org/dataset/who-data-for-gha
- Ghana DHS 2022 (register at dhsprogram.com for microdata)

### Communities and contacts
- GhanaNLP team: contact via https://ghananlp.org — ask if they have any unpublished WER benchmarks for injury vocabulary. Even an email exchange is citable.
- KNUST Kumasi Injury program (kumasiinjury.org): your institutional anchor. If any co-author has a contact there, use it. An acknowledgement from that program adds credibility.
- Masakhane Slack: https://www.masakhane.io — post your paper title, you will get feedback and potential reviewers

### If you want to go further (post-conference)
- Apply for BRRI data access now — takes 8–12 weeks, so if you start today, you could have it for a journal extension
- Contact INDEPTH Network for verbal autopsy data (has injury narratives in text form)
- Look at Afrispeech-200 (200 hours of African-accented English medical speech) — not Twi, but useful for pipeline testing

---

