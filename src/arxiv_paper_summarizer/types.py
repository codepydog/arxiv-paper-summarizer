"""Schema types."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias, Union

from openai import AzureOpenAI, OpenAI
from pydantic import AnyHttpUrl, BaseModel, Field

SUPPORTED_MODEL_TYPE: TypeAlias = Union[OpenAI, AzureOpenAI]
LLM_TYPE: TypeAlias = SUPPORTED_MODEL_TYPE | Any


class Language(str, Enum):
    """Language enum."""

    ENGLISH = "English"
    TRADITIONAL_CHINESE = "Traditional Chinese"


class Paper(BaseModel):
    title: str
    text: str
    url: AnyHttpUrl
    authors: list[str] = Field(default_factory=list, description="List of paper authors")
    published: str | None = Field(default=None, description="Publication date")
    references: list[Paper] | None = None  # type: ignore

    def flatten(self) -> list[Paper]:
        papers = [self]
        if self.references:
            for paper in self.references:
                papers.extend(paper.flatten())
        return papers


class ExtractedSectionResult(BaseModel):
    section: str = Field(..., description="Section title or heading.")
    content: str = Field(..., description="Full section text without summarization.")
    ref_fig: list[str] = Field(..., description="List of figure references in 'Figure-<page>' format.")
    ref_tb: list[str] = Field(..., description="List of table references in 'Table-<page>' format.")


class SectionInfo(BaseModel):
    title: str
    content: str
    image_encoding_str_list: list[str] = Field(
        default_factory=list,
        description="List of image paths.",
        repr=False,
    )
    image_paths: list[str] = Field(default_factory=list, description="List of image paths.")
    table_paths: list[str] = Field(default_factory=list, description="List of table paths.")


class SectionNote(BaseModel):
    header: str
    summary_content: str
    quotes: str
    image_paths: list[str] = Field(default_factory=list, description="List of image paths.")
    table_paths: list[str] = Field(default_factory=list, description="List of table paths.")


class ImagePath(BaseModel):
    path: str
    filename: str


class SummaryResult(BaseModel):
    keynote: str = Field(..., description="Keynote summary.")
    section_notes: list[SectionNote] = Field(default_factory=list, description="List of section notes.")
