# khaya asr v1 `language` query param + translate source_language codes (same values).
# tts v2 uses slightly different codes in lang_prompts.LANG_CODES — use those for tts only.
#
# full voice pipeline (asr confirmed): twi, ga, ewe, dagbani.
# translation + tts only until asr ships: fante, gurene.
KHAYA_ASR_LANG = {
    "twi": "tw",
    "ga": "gaa",
    "ewe": "ee",
    "dagbani": "dag",
    "fante": "fat",
    "gurene": "gur",
}

FULL_VOICE_LANGS = frozenset({"twi", "ga", "ewe", "dagbani"})

# Paper / primary eval: four voice langs + Fante translation track (Gurene omitted)
EVAL_LANGUAGES_FIVE: tuple[str, ...] = ("twi", "ga", "ewe", "fante", "dagbani")

# ASR WER eval — same five (needs asr_transcripts* per language)
ASR_EVAL_LANGS = frozenset(EVAL_LANGUAGES_FIVE)
