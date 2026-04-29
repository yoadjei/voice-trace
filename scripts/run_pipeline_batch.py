# runs the voicetrace pipeline on synthetic audio for one language at a time.
# stages: asr -> translate to en -> extract -> geocode.
#
# Resume (no --limit): skips any id already in pipeline_results_*.csv; reuses ASR/trans
# rows from checkpoint CSVs so you do not redo upstream stages.
#
# Test run:  python run_pipeline_batch.py --lang twi --limit 3
# full voice e2e languages:  twi, ga, ewe, dagbani
import sys
import io
import argparse
import logging
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from pathlib import Path
from pipeline.asr import KhayaQuotaExceededError, transcribe
from pipeline.translate import translate_asr_to_english
from pipeline.extract import extract
from pipeline.geocode import geocode
from pipeline.eval_subset import get_eval_id_set
from pipeline.lang_config import FULL_VOICE_LANGS, KHAYA_ASR_LANG

DATA_SYN = Path("data/synthetic")

LOG = logging.getLogger("voicetrace.pipeline")


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging() -> None:
    root = LOG
    root.handlers.clear()
    root.setLevel(logging.INFO)
    h = _FlushStreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(h)
    root.propagate = False


def _paths_for_lang(lang: str) -> tuple[Path, Path, Path, Path, Path, str]:
    # returns narratives_csv, audio_dir, asr_out, trans_out, final_out, narrative_col
    col = f"narrative_{lang}"
    if lang == "twi":
        narratives = DATA_SYN / "narratives_twi.csv"
        # backward-compatible filenames for the original twi-only run
        asr_out = DATA_SYN / "asr_transcripts.csv"
        trans_out = DATA_SYN / "translations_en.csv"
        final_out = DATA_SYN / "pipeline_results.csv"
    else:
        narratives = DATA_SYN / f"narratives_{lang}.csv"
        asr_out = DATA_SYN / f"asr_transcripts_{lang}.csv"
        trans_out = DATA_SYN / f"translations_en_{lang}.csv"
        final_out = DATA_SYN / f"pipeline_results_{lang}.csv"
    audio_dir = DATA_SYN / "audio" / lang
    return narratives, audio_dir, asr_out, trans_out, final_out, col


def _load_done_ids(path: Path, id_col: str = "id") -> set:
    if path.exists():
        try:
            return set(pd.read_csv(path)[id_col].tolist())
        except Exception:
            pass
    return set()


def run_batch(lang: str, limit: int | None = None) -> None:
    if lang not in FULL_VOICE_LANGS:
        raise SystemExit(
            f"--lang must be one of {sorted(FULL_VOICE_LANGS)} "
            "(fante/gurene: translation+tts only until asr is available)."
        )
    khaya = KHAYA_ASR_LANG[lang]
    narratives_path, audio_dir, asr_out, trans_out, final_out, nar_col = _paths_for_lang(lang)

    if not narratives_path.exists():
        raise SystemExit(f"missing narratives file: {narratives_path}")

    df = pd.read_csv(narratives_path)
    if nar_col not in df.columns:
        raise SystemExit(f"column {nar_col!r} not in {narratives_path}")

    subset = get_eval_id_set()
    if subset is not None:
        df = df[df["id"].isin(subset)].copy().sort_values("id")
        LOG.info("evaluation subset active: %d rows (data/synthetic/evaluation_subset_ids.txt)", len(df))

    total = len(df)
    done_asr = _load_done_ids(asr_out)
    done_trans = _load_done_ids(trans_out)
    done_final = _load_done_ids(final_out)

    with_wav = 0
    pending = 0
    for _, row in df.iterrows():
        wid = int(row["id"])
        if (audio_dir / f"narrative_{wid:03d}.wav").exists():
            with_wav += 1
            if wid not in done_final:
                pending += 1

    lim_msg = f" | --limit {limit} (stop after that many new final rows)" if limit is not None else ""
    LOG.info(
        "start lang=%s khaya=%s | narratives=%s rows=%d | wav on disk=%d | "
        "checkpoint final=%d asr=%d trans=%d | pending (not in %s) ≈%d%s",
        lang,
        khaya,
        narratives_path,
        total,
        with_wav,
        len(done_final),
        len(done_asr),
        len(done_trans),
        final_out.name,
        pending,
        lim_msg,
    )
    if done_final:
        LOG.info(
            "resume: %d id(s) already in %s — those are skipped; run again to continue",
            len(done_final),
            final_out.name,
        )

    asr_rows, translation_rows, result_rows = [], [], []
    row_idx = 0
    processed = 0

    def _append_or_create(path: Path, new_rows: list) -> None:
        if not new_rows:
            return
        new_df = pd.DataFrame(new_rows)
        if path.exists():
            existing = pd.read_csv(path)
            new_df = new_df.reindex(columns=existing.columns)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["id"], keep="last").to_csv(path, index=False)
        else:
            new_df.to_csv(path, index=False)

    for _, row in df.iterrows():
        row_idx += 1
        narrative_id = int(row["id"])
        wav_path = audio_dir / f"narrative_{narrative_id:03d}.wav"

        if not wav_path.exists():
            LOG.info("[%d/%d] skip — no audio file=%s", row_idx, total, wav_path.name)
            continue

        if narrative_id in done_final:
            LOG.info("[%d/%d] skip — already in %s id=%d", row_idx, total, final_out.name, narrative_id)
            continue

        if limit is not None and processed >= limit:
            LOG.info(
                "[%d/%d] --limit %d reached (%d new row(s) written this run) — stopping. "
                "Rerun without --limit to process the rest.",
                row_idx,
                total,
                limit,
                processed,
            )
            break

        t_row = time.perf_counter()
        LOG.info("[%d/%d] ▶ id=%d — running pipeline", row_idx, total, narrative_id)

        if narrative_id not in done_asr:
            t0 = time.perf_counter()
            try:
                asr_text = transcribe(str(wav_path), language=khaya)
            except KhayaQuotaExceededError as e:
                LOG.error(
                    "Khaya quota exceeded — stopping batch before checkpoint (current id=%d). %s",
                    narrative_id,
                    e,
                )
                LOG.error(
                    "After quota resets: remove rows with empty ASR from %s / %s for ids you need to redo.",
                    asr_out,
                    final_out,
                )
                raise SystemExit(2) from e
            asr_rows.append({"id": narrative_id, "asr_transcript": asr_text})
            LOG.info(
                "  asr done in %.2fs | %s",
                time.perf_counter() - t0,
                (str(asr_text)[:120] + "…") if len(str(asr_text)) > 120 else str(asr_text),
            )
            time.sleep(0.5)
        else:
            existing = pd.read_csv(asr_out)
            asr_text = existing.loc[existing["id"] == narrative_id, "asr_transcript"].values[0]
            LOG.info("  asr (cached) | %s", (str(asr_text)[:100] + "…") if len(str(asr_text)) > 100 else str(asr_text))

        if narrative_id not in done_trans:
            t0 = time.perf_counter()
            en_text = translate_asr_to_english(asr_text, khaya, lang) if asr_text else ""
            translation_rows.append({"id": narrative_id, "translated_en": en_text})
            LOG.info(
                "  translate done in %.2fs | %s",
                time.perf_counter() - t0,
                (str(en_text)[:120] + "…") if len(str(en_text)) > 120 else str(en_text),
            )
            time.sleep(0.5)
        else:
            existing = pd.read_csv(trans_out)
            en_text = existing.loc[existing["id"] == narrative_id, "translated_en"].values[0]
            LOG.info("  translate (cached) | %s", (str(en_text)[:100] + "…") if len(str(en_text)) > 100 else str(en_text))

        if en_text and isinstance(en_text, str) and en_text.strip():
            t0 = time.perf_counter()
            result = extract(en_text)
            LOG.info("  extract+first_aid done in %.2fs", time.perf_counter() - t0)
            if result.get("fatal_billing"):
                LOG.error(
                    "Anthropic billing/credits error — add credits, then re-run. Stopping before writing this id."
                )
                raise SystemExit(3)
        else:
            result = {"extraction": {k: "unknown" for k in [
                "injury_type", "mechanism", "severity", "body_region",
                "victim_sex", "victim_age_group", "location_description"
            ]}, "first_aid": ""}
            LOG.info("  extract skipped (empty english)")
        time.sleep(0.5)

        extraction = result["extraction"]
        first_aid = result["first_aid"]
        location_desc = extraction.get("location_description", "unknown")

        try:
            t0 = time.perf_counter()
            geo = geocode(location_desc)
            lat = geo.get("lat")
            lng = geo.get("lng")
            facility_name = geo.get("facility_name")
            facility_dist_km = geo.get("facility_dist_km")
            LOG.info(
                "  geocode done in %.2fs | lat=%s lng=%s facility=%s dist_km=%s",
                time.perf_counter() - t0,
                lat,
                lng,
                facility_name,
                facility_dist_km,
            )
        except Exception as e:
            LOG.warning("  geocode ERROR: %s", e)
            lat, lng, facility_name, facility_dist_km = None, None, None, None

        processed += 1
        LOG.info(
            "  ✓ id=%d finished in %.2fs | injury_type=%s severity=%s",
            narrative_id,
            time.perf_counter() - t_row,
            extraction.get("injury_type"),
            extraction.get("severity"),
        )

        result_row = {
            "id": narrative_id,
            "asr_transcript": asr_text,
            "translated_en": en_text,
            **extraction,
            "first_aid": first_aid,
            "lat": lat,
            "lng": lng,
            "facility_name": facility_name,
            "facility_dist_km": facility_dist_km,
        }
        result_rows.append(result_row)
        # Per-row checkpoint: logs show progress but CSVs used to update only at loop exit.
        _append_or_create(asr_out, [{"id": narrative_id, "asr_transcript": asr_text}])
        _append_or_create(trans_out, [{"id": narrative_id, "translated_en": en_text}])
        _append_or_create(final_out, [result_row])
        LOG.info("  checkpoint → disk | id=%d", narrative_id)

    if processed:
        LOG.info(
            "checkpoint summary | +asr=%d +trans=%d +final=%d (written per row)",
            len(asr_rows),
            len(translation_rows),
            len(result_rows),
        )

    LOG.info("done lang=%s | new rows completed this run=%d | outputs:", lang, processed)
    LOG.info("  asr transcripts   → %s", asr_out)
    LOG.info("  translations      → %s", trans_out)
    LOG.info("  final results     → %s", final_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Run VoiceTrace batch ASR→translate→extract→geocode.")
    p.add_argument(
        "--lang",
        default="twi",
        choices=sorted(FULL_VOICE_LANGS),
        help="source language (must have .wav under data/synthetic/audio/<lang>/)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="warnings/errors only (less progress detail)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="process at most N new rows (ids not already in pipeline_results), then stop — for testing",
    )
    args = p.parse_args()
    _setup_logging()
    if args.quiet:
        LOG.setLevel(logging.WARNING)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    run_batch(args.lang, limit=args.limit)


if __name__ == "__main__":
    main()
