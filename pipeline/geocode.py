# geocodes a location string via nominatim, then finds the nearest facility
# via haversine distance against local GeoJSON (points or polygons → centroid).
#
# Loads and merges every file that exists from:
#   1) VOICETRACE_FACILITIES_GEOJSON — optional; comma- or semicolon-separated extra paths
#   2) data/raw/ghana_health_facilities.geojson (points, optional)
#   3) data/raw/hotosm_gha_health_facilities_polygons_geojson.geojson (OSM/HOT polygons)
#   4) data/raw/ghs_facilities.geojson — optional richer GHS export you add (FeatureCollection)
#
# To add GHS Facility Finder data: export or convert to GeoJSON, save as ghs_facilities.geojson.
import json
import math
import os
import re
import time
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def _default_facility_paths() -> list[Path]:
    return [
        Path("data/raw/ghana_health_facilities.geojson"),
        Path("data/raw/hotosm_gha_health_facilities_polygons_geojson.geojson"),
        Path("data/raw/ghs_facilities.geojson"),
    ]


def _all_facility_paths() -> list[Path]:
    # merge order: env extras first, then defaults (skip missing paths; de-dupe by resolved path)
    seen: set[str] = set()
    out: list[Path] = []
    env = os.environ.get("VOICETRACE_FACILITIES_GEOJSON", "").strip()
    if env:
        for part in re.split(r"[;,]", env):
            p = Path(part.strip())
            if not str(p):
                continue
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if p.exists() and key not in seen:
                seen.add(key)
                out.append(p)
    for d in _default_facility_paths():
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if d.exists() and key not in seen:
            seen.add(key)
            out.append(d)
    return out


_geolocator = Nominatim(user_agent="voicetrace-ghana-injury-surveillance")
_facilities: list[dict] | None = None  # lazy-loaded
_facilities_source: str | None = None  # description for logging


def _ring_centroid(ring: list) -> tuple[float, float] | None:
    # simple mean of vertices — good enough for building footprints
    if not ring or len(ring) < 1:
        return None
    lng_sum, lat_sum, n = 0.0, 0.0, 0
    for pt in ring:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        lng_sum += float(pt[0])
        lat_sum += float(pt[1])
        n += 1
    if n == 0:
        return None
    return lng_sum / n, lat_sum / n


def _geometry_lat_lng(geom: dict) -> tuple[float, float] | None:
    if not geom:
        return None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    if gtype == "Point":
        if len(coords) < 2:
            return None
        return float(coords[0]), float(coords[1])
    if gtype == "Polygon":
        ring = coords[0] if coords else None
        return _ring_centroid(ring) if ring else None
    if gtype == "MultiPolygon":
        first = coords[0] if coords else None
        ring = first[0] if first else None
        return _ring_centroid(ring) if ring else None
    return None


def _facility_label(props: dict) -> str:
    return (
        props.get("name")
        or props.get("name:en")
        or props.get("facility_name")
        or props.get("FacilityName")
        or props.get("facility")
        or props.get("title")
        or props.get("addr:city")
        or props.get("amenity")
        or props.get("healthcare")
        or "unnamed facility"
    )


def _load_facilities() -> list[dict]:
    global _facilities, _facilities_source
    if _facilities is not None:
        return _facilities
    paths = _all_facility_paths()
    if not paths:
        print("[geocode] no facility geojson found under data/raw/ — nearest facility lookup disabled")
        _facilities = []
        return _facilities
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for feature in data.get("features", []):
            geom = feature.get("geometry") or {}
            ll = _geometry_lat_lng(geom)
            if not ll:
                continue
            lng, lat = ll
            props = feature.get("properties") or {}
            name = _facility_label(props)
            rows.append({"name": name, "lat": float(lat), "lng": float(lng)})
    _facilities = rows
    _facilities_source = ", ".join(p.name for p in paths)
    print(f"[geocode] loaded {len(_facilities)} facility point(s) from {len(paths)} file(s): {_facilities_source}")
    return _facilities


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    # haversine formula — returns distance in km
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def _nearest_facility(lat: float, lng: float) -> tuple[str | None, float | None]:
    # returns (facility_name, distance_km) for the closest loaded facility
    facilities = _load_facilities()
    if not facilities:
        return None, None
    best_name, best_dist = None, float("inf")
    for f in facilities:
        d = _haversine_km(lat, lng, f["lat"], f["lng"])
        if d < best_dist:
            best_dist = d
            best_name = f["name"]
    return best_name, round(best_dist, 2)


def _query_variants(location_str: str) -> list[str]:
    # nominatim works best on short placenames; try fallbacks if the full line fails
    s = location_str.strip()
    if not s:
        return []
    variants: list[str] = [s]
    low = s.lower()
    for prefix in ("near ", "around ", "at ", "close to ", "in "):
        if low.startswith(prefix):
            rest = s[len(prefix) :].strip()
            if rest:
                variants.append(rest)
            break
    if "," in s:
        for part in s.split(","):
            part = part.strip()
            if part and part.lower() not in {v.lower() for v in variants}:
                variants.append(part)
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def geocode(location_str: str) -> dict:
    # geocode location string → lat/lng, then find nearest health facility.
    # returns dict with lat, lng, display_name, facility_name, facility_dist_km.
    null_result = {"lat": None, "lng": None, "display_name": None, "facility_name": None, "facility_dist_km": None}
    if not location_str or location_str.lower() == "unknown":
        return null_result

    for variant in _query_variants(location_str):
        query = f"{variant}, Ghana"
        try:
            time.sleep(1)  # nominatim rate limit: 1 req/sec per attempt
            location = _geolocator.geocode(query, timeout=10, country_codes="gh")
            if not location:
                continue
            lat, lng = location.latitude, location.longitude
            facility_name, facility_dist_km = _nearest_facility(lat, lng)
            return {
                "lat": lat,
                "lng": lng,
                "display_name": location.address,
                "facility_name": facility_name,
                "facility_dist_km": facility_dist_km,
            }
        except GeocoderTimedOut:
            print(f"[geocode] timeout: {query}")
        except Exception as e:
            print(f"[geocode] error for '{query}': {e}")

    return null_result
