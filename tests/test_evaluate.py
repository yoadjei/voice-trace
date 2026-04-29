# tests for wer and extraction f1 evaluation functions
import pytest
from evaluation.evaluate_asr import compute_wer
from evaluation.evaluate_extraction import compute_field_f1


class TestComputeWer:
    def test_identical_strings_give_zero_wer(self):
        refs = ["me ko fie", "obarima no tee"]
        hyps = ["me ko fie", "obarima no tee"]
        result = compute_wer(refs, hyps)
        assert result["overall_wer"] == pytest.approx(0.0)

    def test_completely_wrong_gives_high_wer(self):
        refs = ["me ko fie"]
        hyps = ["xyz abc def"]
        result = compute_wer(refs, hyps)
        assert result["overall_wer"] > 0.5

    def test_returns_expected_keys(self):
        result = compute_wer(["a b c"], ["a b d"])
        assert "overall_wer" in result
        assert "n_references" in result


class TestComputeFieldF1:
    def test_perfect_predictions_give_f1_one(self):
        gold = [{"injury_type": "rta", "severity": "minor"}]
        pred = [{"injury_type": "rta", "severity": "minor"}]
        result = compute_field_f1(gold, pred, fields=["injury_type", "severity"])
        assert result["injury_type"]["f1"] == pytest.approx(1.0)

    def test_wrong_predictions_give_lower_f1(self):
        gold = [{"injury_type": "rta"}, {"injury_type": "fall"}]
        pred = [{"injury_type": "fall"}, {"injury_type": "fall"}]
        result = compute_field_f1(gold, pred, fields=["injury_type"])
        assert result["injury_type"]["f1"] < 1.0

    def test_returns_macro_f1(self):
        gold = [{"injury_type": "rta", "severity": "minor"}]
        pred = [{"injury_type": "rta", "severity": "minor"}]
        result = compute_field_f1(gold, pred, fields=["injury_type", "severity"])
        assert "macro_f1" in result
