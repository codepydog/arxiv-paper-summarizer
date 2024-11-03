"""Utilities helper functions."""

import base64
import json
import os
import re
import types
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
