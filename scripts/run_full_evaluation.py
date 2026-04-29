#!/usr/bin/env python3
"""
VoiceTrace: Full Evaluation Pipeline

Single entry point to reproduce all evaluation results from the paper.

Usage:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --config configs/default.yaml
    python scripts/run_full_evaluation.py --track 1  # Track 1 only
    python scripts/run_full_evaluation.py --track 2  # Track 2 only
    python scripts/run_full_evaluation.py --track 3  # Track 3 only

Output:
    results/evaluation/
        track1_asr_wer.csv
        track1_extraction_f1.csv
        track2_translation_fidelity.csv
        track2_extraction_f1.csv
        track3_cross_language_kappa.csv
        track3_kappa_vs_gold.csv
"""
import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seeds


def run_track1(config):
    """Track 1: Full end-to-end pipeline (ASR → MT → Extract)."""
    print("\n" + "=" * 60)
    print("TRACK 1: Full End-to-End Pipeline")
    print("=" * 60)
    
    from evaluation.evaluate_asr import main as eval_asr
    from evaluation.evaluate_extraction import run_extraction_evaluation
    
    # ASR evaluation
    print("\n[Track 1] Evaluating ASR (WER)...")
    eval_asr()
    
    # Post-ASR extraction evaluation
    print("\n[Track 1] Evaluating post-ASR extraction (F1)...")
    for lang in config.languages.evaluated:
        run_extraction_evaluation(lang)


def run_track2(config):
    """Track 2: Translation round-trip fidelity and extraction on clean text."""
    print("\n" + "=" * 60)
    print("TRACK 2: Translation Round-Trip Fidelity")
    print("=" * 60)
    
    from evaluation.evaluate_translation import main as eval_translation
    from evaluation.run_extraction_all_langs import main as eval_extraction_all
    
    # Translation fidelity (BERTScore, BLEU)
    print("\n[Track 2] Evaluating translation round-trip fidelity...")
    eval_translation()
    
    # Extraction on round-tripped text
    print("\n[Track 2] Evaluating extraction on round-tripped text...")
    eval_extraction_all()


def run_track3(config):
    """Track 3: Cross-language extraction consistency."""
    print("\n" + "=" * 60)
    print("TRACK 3: Cross-Language Extraction Consistency"
          )
    print("=" * 60)
    
    from evaluation.evaluate_consistency import main as eval_consistency
    
    print("\n[Track 3] Computing cross-language Cohen's kappa...")
    eval_consistency()


def main():
    parser = argparse.ArgumentParser(
        description="VoiceTrace Full Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--track",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run specific track only (default: run all)"
    )
    args = parser.parse_args()
    
    # Load config and set seeds
    config = load_config(args.config)
    set_seeds(config)
    
    print("=" * 60)
    print("VoiceTrace Evaluation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Languages: {config.languages.evaluated}")
    print(f"Seeds: numpy={config.seeds.numpy}, python={config.seeds.python}")
    
    # Create output directory
    output_dir = REPO_ROOT / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tracks
    if args.track is None or args.track == 1:
        run_track1(config)
    if args.track is None or args.track == 2:
        run_track2(config)
    if args.track is None or args.track == 3:
        run_track3(config)
    
    print("\n" + "=" * 60)
    print("Evaluation complete. Results saved to results/evaluation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
