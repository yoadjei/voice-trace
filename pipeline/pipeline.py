import time
from pipeline.asr import transcribe
from pipeline.translate import translate
from pipeline.extract import extract
from pipeline.geocode import geocode


def run_pipeline(wav_path: str) -> dict:
    # run the full voicetrace pipeline on a single .wav file.
    # returns flat dict with all extracted fields + geocoordinates.
    # never raises — returns partial results if any stage fails.

    # stage 1: asr (twi audio → twi text)
    asr_transcript = transcribe(wav_path)
    time.sleep(0.5)

    # stage 2: translate (twi → english)
    translated_text = translate(asr_transcript, source_lang="tw", target_lang="en") if asr_transcript else ""
    time.sleep(0.5)

    # stage 3: extract (english → structured json)
    extracted = extract(translated_text) if translated_text else {
        "injury_type": "unknown", "mechanism": "unknown", "severity": "unknown",
        "body_region": "unknown", "victim_sex": "unknown",
        "victim_age_group": "unknown", "location_description": "unknown",
    }

    time.sleep(0.5)

    # stage 4: geocode
    try:
        geo = geocode(extracted.get("location_description", "unknown"))
        lat = geo.get("lat")
        lng = geo.get("lng")
    except Exception as e:
        print(f"[pipeline] geocode ERROR: {e}")
        lat, lng = None, None

    return {
        "asr_transcript": asr_transcript,
        "translated_text": translated_text,
        **extracted,
        "lat": lat,
        "lng": lng,
    }
