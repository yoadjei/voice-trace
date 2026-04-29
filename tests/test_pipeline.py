import pytest
from unittest.mock import patch


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
