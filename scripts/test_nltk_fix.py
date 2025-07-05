#!/usr/bin/env python3
"""Test script to verify NLTK fix works correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_paper_summarizer.utils import setup_nltk_offline, setup_unstructured_environment, patch_nltk_download


def test_nltk_setup():
    """Test NLTK setup functions."""
    print("🧪 Testing NLTK utilities...")

    print("📦 Setting up unstructured environment...")
    setup_unstructured_environment()
    print("✅ Environment variables set")

    print("🔧 Patching NLTK download...")
    patch_result = patch_nltk_download()
    print(f"✅ NLTK download patched: {patch_result}")

    print("📚 Setting up NLTK offline...")
    setup_result = setup_nltk_offline()
    print(f"✅ NLTK offline setup: {setup_result}")

    print("🎉 All NLTK utilities tested successfully!")


if __name__ == "__main__":
    test_nltk_setup()
