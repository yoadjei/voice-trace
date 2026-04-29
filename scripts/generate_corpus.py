#!/usr/bin/env python3
"""
VoiceTrace: Corpus Generation Pipeline

Generates the synthetic evaluation corpus:
1. Generate English narratives from epidemiological distributions
2. Translate to target languages via Khaya MT
3. Synthesize audio via Khaya TTS
4. Create gold annotations

Usage:
    python scripts/generate_corpus.py --step narratives
    python scripts/generate_corpus.py --step translate --lang twi
    python scripts/generate_corpus.py --step translate --all
    python scripts/generate_corpus.py --step tts --lang twi
    python scripts/generate_corpus.py --step tts --all
    python scripts/generate_corpus.py --step annotate
    python scripts/generate_corpus.py --all  # Run full pipeline

Prerequisites:
    - ANTHROPIC_API_KEY in .env (for narrative generation)
    - KHAYA_API_KEY in .env (for translation and TTS)
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seeds


def generate_narratives(config):
    """Step 1: Generate English injury narratives."""
    print("\n[Step 1] Generating English narratives...")
    from data_gen.generate_narratives import main as gen_narratives
    gen_narratives()


def translate_narratives(config, lang: str | None = None):
    """Step 2: Translate narratives to target languages."""
    from data_gen.translate_all_langs import translate_language
    
    languages = [lang] if lang else config.languages.evaluated
    for l in languages:
        print(f"\n[Step 2] Translating to {l}...")
        translate_language(l)


def synthesize_audio(config, lang: str | None = None):
    """Step 3: Synthesize audio via TTS."""
    from data_gen.tts import synthesize_language
    
    languages = [lang] if lang else config.languages.evaluated
    for l in languages:
        print(f"\n[Step 3] Synthesizing audio for {l}...")
        synthesize_language(l)


def create_annotations(config):
    """Step 4: Create/verify gold annotations."""
    print("\n[Step 4] Gold annotations...")
    from data_gen.annotate_gold import main as annotate
    annotate()


def main():
    parser = argparse.ArgumentParser(
        description="VoiceTrace Corpus Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--step",
        choices=["narratives", "translate", "tts", "annotate"],
        help="Run specific step only"
    )
    parser.add_argument(
        "--lang",
        choices=["twi", "fante", "ewe", "ga", "dagbani"],
        help="Target language (for translate/tts steps)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full corpus generation pipeline"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seeds(config)
    
    print("=" * 60)
    print("VoiceTrace Corpus Generation")
    print("=" * 60)
    
    if args.all:
        generate_narratives(config)
        translate_narratives(config)
        synthesize_audio(config)
        create_annotations(config)
    elif args.step == "narratives":
        generate_narratives(config)
    elif args.step == "translate":
        translate_narratives(config, args.lang)
    elif args.step == "tts":
        synthesize_audio(config, args.lang)
    elif args.step == "annotate":
        create_annotations(config)
    else:
        parser.print_help()
        return
    
    print("\n" + "=" * 60)
    print("Corpus generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
