"""Utilities helper functions."""

import base64
import os

import tiktoken


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
