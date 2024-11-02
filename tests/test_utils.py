import pytest
import base64

from unittest.mock import patch, mock_open

from arxiv_paper_summarizer.utils import get_env_var, compute_token, encode_image


def test_get_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/home/runner")
    monkeypatch.setenv("USER", "runner")
    monkeypatch.setenv("SHELL", "/bin/bash")

    assert get_env_var("HOME") == "/home/runner"
    assert get_env_var("USER") == "runner"
    assert get_env_var("SHELL") == "/bin/bash"

    with pytest.raises(ValueError):
        get_env_var("NOT_EXISTING")


def test_compute_token():
    sample_text = "Hello, world!"
    expected_token_count = 3  # Assuming the text splits into 3 tokens

    with patch("arxiv_paper_summarizer.utils.tiktoken") as mock_tiktoken:
        mock_encoder = mock_tiktoken.encoding_for_model.return_value
        mock_encoder.encode.return_value = ["Hello", ",", "world!"]

        token_count = compute_token(sample_text)

        assert token_count == expected_token_count
        mock_tiktoken.encoding_for_model.assert_called_with("gpt-4o")
        mock_encoder.encode.assert_called_with(sample_text)


def test_encode_image():
    sample_image_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    expected_base64_encoded = base64.b64encode(sample_image_content).decode("utf-8")

    with patch("builtins.open", mock_open(read_data=sample_image_content)) as mock_file:
        result = encode_image("path/to/image.png")

        assert result == expected_base64_encoded
        mock_file.assert_called_once_with("path/to/image.png", "rb")
