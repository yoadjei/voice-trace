# re-run geocode on existing pipeline_results*.csv — no ASR / translate / extract.
# use after fixing pipeline/geocode.py, adding facility geojson, or improving extraction.
#
#   python -m data_gen.regeocode_pipeline_results --lang twi
#   python -m data_gen.regeocode_pipeline_results --lang twi --dry-run
#   python -m data_gen.regeocode_pipeline_results --lang twi --clear-geo  # blank lat/lng/facility columns
#
# Full pipeline reset (re-run ASR→…→geocode for all ids): delete these for Twi:
#   data/synthetic/pipeline_results.csv
#   data/synthetic/asr_transcripts.csv
#   data/synthetic/translations_en.csv
import argparse
import time
from pathlib import Path

import pandas as pd

from pipeline.geocode import geocode
from pipeline.lang_config import FULL_VOICE_LANGS

DATA_SYN = Path("data/synthetic")


def _final_path(lang: str) -> Path:
    if lang == "twi":
        return DATA_SYN / "pipeline_results.csv"
    return DATA_SYN / f"pipeline_results_{lang}.csv"


def _lat_missing(val) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except TypeError:
        pass
    return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-apply geocode() to pipeline_results using location_description only."
    )
    p.add_argument("--lang", required=True, choices=sorted(FULL_VOICE_LANGS))
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="re-geocode every row (default: only rows with missing lat)",
    )
    p.add_argument("--dry-run", action="store_true", help="print planned updates, do not write")
    p.add_argument(
        "--clear-geo",
        action="store_true",
        help="set lat, lng, facility_name, facility_dist_km to empty (undo regeocode output)",
    )
    args = p.parse_args()

    path = _final_path(args.lang)
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "  Create it first by running the batch pipeline, e.g.\n"
            "    python run_pipeline_batch.py --lang twi\n"
            "  (regeocode only refreshes lat/lng/facility_* on existing rows.)"
        )

    df = pd.read_csv(path)
    if "location_description" not in df.columns:
        raise SystemExit(f"{path} has no location_description column")

    if args.clear_geo:
        for col in ("lat", "lng", "facility_name", "facility_dist_km"):
            if col in df.columns:
                df[col] = pd.NA
            else:
                df[col] = pd.NA
        df.to_csv(path, index=False)
        print(f"cleared geo columns in {path} | rows={len(df)}")
        return

    only_missing = not args.all_rows  # default: fill in rows where lat is null
    updates = 0
    for i, row in df.iterrows():
        lat = row.get("lat")
        if only_missing and not _lat_missing(lat):
            continue
        loc = row.get("location_description", "unknown")
        if not isinstance(loc, str):
            loc = str(loc) if loc is not None else "unknown"
        loc = loc.strip() or "unknown"

        if args.dry_run:
            print(f"id={row.get('id')} loc={loc!r} -> would geocode")
            updates += 1
            continue

        geo = geocode(loc)
        df.at[i, "lat"] = geo.get("lat")
        df.at[i, "lng"] = geo.get("lng")
        df.at[i, "facility_name"] = geo.get("facility_name")
        df.at[i, "facility_dist_km"] = geo.get("facility_dist_km")
        updates += 1
        print(
            f"id={row.get('id')} lat={geo.get('lat')} lng={geo.get('lng')} "
            f"facility={geo.get('facility_name')} dist_km={geo.get('facility_dist_km')}"
        )
        time.sleep(0.2)

    if args.dry_run:
        print(f"dry-run: {updates} row(s) would be updated → {path}")
        return

    df.to_csv(path, index=False)
    print(f"wrote {path} | geocode updates applied: {updates}")


if __name__ == "__main__":
    main()
