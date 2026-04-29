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
