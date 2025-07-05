"""Tests for NLTK utilities in utils.py."""

import os
import sys
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_paper_summarizer.utils import patch_nltk_download, setup_nltk_offline, setup_unstructured_environment


class TestNLTKUtils:
    """Test cases for NLTK utility functions."""

    def test_setup_unstructured_environment(self):
        """Test that environment variables are set correctly."""
        # Store original environment variables
        original_nltk_data = os.environ.get("NLTK_DATA")
        original_disable_download = os.environ.get("UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD")

        try:
            # Call the function
            setup_unstructured_environment()

            # Check that environment variables are set
            assert "NLTK_DATA" in os.environ
            assert "UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD" in os.environ
            assert os.environ["UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD"] == "1"

            # Check that NLTK_DATA contains expected paths
            nltk_data_paths = os.environ["NLTK_DATA"].split(os.pathsep)
            expected_paths = [
                os.path.expanduser("~/.local/share/nltk_data"),
                os.path.expanduser("~/nltk_data"),
                "/usr/local/share/nltk_data",
                "/usr/share/nltk_data",
            ]

            for expected_path in expected_paths:
                assert expected_path in nltk_data_paths

        finally:
            # Restore original environment variables
            if original_nltk_data is not None:
                os.environ["NLTK_DATA"] = original_nltk_data
            elif "NLTK_DATA" in os.environ:
                del os.environ["NLTK_DATA"]

            if original_disable_download is not None:
                os.environ["UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD"] = original_disable_download
            elif "UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD" in os.environ:
                del os.environ["UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD"]

    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_setup_nltk_offline_import_error(self, mock_warn):
        """Test setup_nltk_offline when nltk import fails."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'nltk'")):
            result = setup_nltk_offline()

            assert result is False
            mock_warn.assert_called_once()
            assert "Failed to setup NLTK" in str(mock_warn.call_args[0][0])

    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_patch_nltk_download_import_error(self, mock_warn):
        """Test patch_nltk_download when nltk import fails."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'nltk'")):
            result = patch_nltk_download()

            assert result is False
            mock_warn.assert_called_once()
            assert "Failed to patch NLTK download" in str(mock_warn.call_args[0][0])

    @patch("arxiv_paper_summarizer.utils.ssl")
    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_setup_nltk_offline_success(self, mock_warn, mock_ssl):
        """Test successful NLTK offline setup."""
        # Mock nltk module
        mock_nltk = Mock()
        mock_nltk.data.path = []
        mock_nltk.data.find = Mock(side_effect=LookupError("Not found"))
        mock_nltk.download = Mock(return_value=True)

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            with patch("os.path.exists", return_value=True):
                result = setup_nltk_offline()

                assert result is True
                # Check that SSL context was set
                mock_ssl._create_default_https_context = mock_ssl._create_unverified_context

                # Check that NLTK paths were added
                assert len(mock_nltk.data.path) > 0

    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_patch_nltk_download_success(self, mock_warn):
        """Test successful NLTK download patching."""
        # Mock nltk module
        mock_nltk = Mock()
        original_download = Mock()
        mock_nltk.download = original_download

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            result = patch_nltk_download()

            assert result is True
            # Check that download function was replaced
            assert mock_nltk.download != original_download

    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_patched_download_handles_403_error(self, mock_warn):
        """Test that patched download function handles 403 errors gracefully."""
        # Mock nltk module
        mock_nltk = Mock()
        original_download = Mock(side_effect=Exception("HTTP Error 403: Forbidden"))
        mock_nltk.download = original_download

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            patch_nltk_download()

            # Test the patched download function
            result = mock_nltk.download("punkt")

            assert result is False
            mock_warn.assert_called()
            assert "NLTK download blocked" in str(mock_warn.call_args[0][0])

    @patch("arxiv_paper_summarizer.utils.warnings.warn")
    def test_patched_download_reraises_other_errors(self, mock_warn):
        """Test that patched download function reraises non-HTTP errors."""
        # Mock nltk module
        mock_nltk = Mock()
        original_download = Mock(side_effect=ValueError("Some other error"))
        mock_nltk.download = original_download

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            patch_nltk_download()

            # Test the patched download function
            with pytest.raises(ValueError, match="Some other error"):
                mock_nltk.download("punkt")


if __name__ == "__main__":
    # Run a simple smoke test
    print("🧪 Running NLTK utilities smoke test...")

    try:
        setup_unstructured_environment()
        print("✅ setup_unstructured_environment() - OK")

        result1 = setup_nltk_offline()
        print(f"✅ setup_nltk_offline() - {result1}")

        result2 = patch_nltk_download()
        print(f"✅ patch_nltk_download() - {result2}")

        print("🎉 All functions executed without fatal errors!")

    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        sys.exit(1)
