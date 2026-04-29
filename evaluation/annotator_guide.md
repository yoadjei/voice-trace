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
