"""Utilities helper functions."""

import base64
import os
import types
from typing import ParamSpec, Sequence, TypeVar, Union, get_args, get_origin

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
