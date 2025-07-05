"""Utilities helper functions."""

import base64
import json
import os
import re
import ssl
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, ParamSpec, Sequence, TypeVar, Union, get_args, get_origin

import tiktoken

P = ParamSpec("P")
R = TypeVar("R")
TypeT = type[R]


def get_env_var(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    raise ValueError(f"Environment variable '{name}' is not set.")


def compute_token(text: str, model_name: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_union_type(type_: type) -> bool:
    """Return True if the type is a union type."""
    type_ = get_origin(type_) or type_
    return type_ is Union or type_ is types.UnionType


def split_union_type(type_: TypeT) -> Sequence[TypeT]:
    """Split a union type into its constituent types."""
    return get_args(type_) if is_union_type(type_) else [type_]


def _extract_json_block(text: str) -> str | None:
    pattern = re.compile(r"```json\s*(\[[\s\S]*?\])\s*```", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1) if match else None


def _clean_json_text(json_text: str) -> str:
    string_pattern = re.compile(r"\"(.*?)(?<!\\)\"", re.DOTALL)

    def replace_newlines(match: re.Match) -> str:
        string_content = match.group(1).replace("\n", " ")
        return f'"{string_content}"'

    return string_pattern.sub(replace_newlines, json_text)


def extract_json_content(text: str) -> list[dict[str, str]] | None | Any:
    """
    Extract JSON content enclosed within ```json ... ``` from the given text.

    This function searches for a JSON block in the provided text that is enclosed within triple backticks
    and labeled as 'json'. It cleans the JSON text by removing unescaped newline characters within strings,
    which are invalid in JSON. The cleaned JSON text is then parsed and returned as a list of dictionaries.

    Args:
        text (str): The input text containing the JSON block.

    Returns:
        Optional[List[Dict[str, str]]]: The parsed JSON content as a list of dictionaries,
        or None if no valid JSON block is found.
    """
    json_text = _extract_json_block(text)
    if json_text is None:
        return None

    cleaned_json_text = _clean_json_text(json_text)

    try:
        return json.loads(cleaned_json_text)
    except json.JSONDecodeError:
        return None


def normalize_image_filename(filename: str) -> str:
    """
    Normalize image filenames by removing unnecessary parts.
    """
    path = Path(filename)
    stem, suffix = path.stem, path.suffix

    if stem.startswith(("figure", "table")) and stem.count("-") > 1:
        parts = stem.split("-")
        return f"{parts[0]}-{parts[-1]}{suffix}"

    return filename


def is_first_figure(path: str) -> bool:
    return bool(re.match(r"^figure-\d+-1$", Path(path).stem))


def get_publication_week_folder(published_date: str | None, base_dir: str | Path = "papers") -> Path:
    """
    Generate folder path based on paper's publication date.

    Args:
        published_date: ISO format date string from paper.published
        base_dir: Base directory for papers (default: "papers")

    Returns:
        Path object for the week folder (e.g., papers/2024/October/week_42)
        Falls back to current date if published_date is None or invalid
    """
    base_path = Path(base_dir)

    try:
        if published_date:
            # Parse ISO format date string (e.g., "2024-10-15T10:30:00Z")
            pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            year = pub_date.year
            month_name = pub_date.strftime("%B")  # Full month name (e.g., "October")
            week_num = pub_date.isocalendar()[1]
        else:
            # Fallback to current date if no publication date
            current_date = datetime.now()
            year = current_date.year
            month_name = current_date.strftime("%B")
            week_num = current_date.isocalendar()[1]
    except (ValueError, AttributeError):
        # Fallback to current date if parsing fails
        current_date = datetime.now()
        year = current_date.year
        month_name = current_date.strftime("%B")
        week_num = current_date.isocalendar()[1]

    return base_path / f"{year}/{month_name}/week_{week_num:02d}"


# NLTK utilities for handling package downloads and initialization
def setup_nltk_offline():
    """Setup NLTK for offline use by ensuring required packages are available."""
    try:
        import nltk

        # Set up SSL context for downloads (if needed)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Set NLTK data path to common locations
        nltk_data_paths = [
            os.path.expanduser("~/.local/share/nltk_data"),
            os.path.expanduser("~/nltk_data"),
            "/usr/local/share/nltk_data",
            "/usr/share/nltk_data",
        ]

        for path in nltk_data_paths:
            if os.path.exists(path):
                if path not in nltk.data.path:
                    nltk.data.path.append(path)

        # Try to download required packages if not available and if possible
        required_packages = ["punkt", "punkt_tab", "averaged_perceptron_tagger", "stopwords"]

        for package in required_packages:
            try:
                nltk.data.find(f"tokenizers/{package}")
            except LookupError:
                try:
                    # Try to download if not found
                    nltk.download(package, quiet=True)
                except Exception as e:
                    warnings.warn(f"Could not download NLTK package '{package}': {e}")

        return True

    except Exception as e:
        warnings.warn(f"Failed to setup NLTK: {e}")
        return False


def setup_unstructured_environment():
    """Setup environment variables to prevent unstructured from downloading NLTK packages."""
    # Set environment variables to prevent unstructured from trying to download NLTK data
    os.environ["NLTK_DATA"] = os.pathsep.join(
        [
            os.path.expanduser("~/.local/share/nltk_data"),
            os.path.expanduser("~/nltk_data"),
            "/usr/local/share/nltk_data",
            "/usr/share/nltk_data",
        ]
    )

    # Disable automatic NLTK downloads
    os.environ["UNSTRUCTURED_DISABLE_NLTK_DOWNLOAD"] = "1"


def patch_nltk_download():
    """Patch NLTK download function to prevent network calls in restricted environments."""
    try:
        import nltk

        original_download = nltk.download

        def safe_download(*args, **kwargs):
            """Safe download that doesn't fail on network errors."""
            try:
                return original_download(*args, **kwargs)
            except Exception as e:
                if "403" in str(e) or "Forbidden" in str(e) or "HTTP" in str(e):
                    warnings.warn(f"NLTK download blocked (likely in CI environment): {e}")
                    return False
                else:
                    raise

        nltk.download = safe_download
        return True

    except Exception as e:
        warnings.warn(f"Failed to patch NLTK download: {e}")
        return False
