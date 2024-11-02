"""AI Summarizer for arXiv papers."""

import tempfile
import warnings
from pathlib import Path

from pydantic import TypeAdapter
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

from arxiv_paper_summarizer.arxiv import fetch_papers_by_url, load_paper_as_file_by_url
from arxiv_paper_summarizer.prompt_function import openai_prompt
from arxiv_paper_summarizer.types import (
    LLM_TYPE,
    ExtractedSectionResult,
    ImagePath,
    Paper,
    SectionInfo,
)
from arxiv_paper_summarizer.utils import encode_image, extract_json_content, get_env_var

try:
    from langfuse.openai import OpenAI
except ImportError:
    from openai import OpenAI


class ArxivPaperSummarizer:
    """Summarizer for arXiv papers."""

    def __init__(self, arxiv_url: str, llm: LLM_TYPE | None = None) -> None:
        """Initialize the summarizer."""
        self._arxiv_url = arxiv_url
        self._llm = llm

        self._paper: Paper | None = None
        self._elements: list[Element] | None = None
        self._image_path_list: list[ImagePath] = []

        self._image_output_dir = tempfile.mkdtemp()

        self._post_init()

    def _post_init(self):
        self._paper = self.get_paper()
        self._elements = self.get_partition_elements()
        self._image_path_list = self.get_image_path_list()

    def summarize(self):
        """Summarize the arXiv paper."""
        section_list = self.extract_section_list(self.paper.text)
        section_info_list = self.get_section_info_list(section_list)
        print(section_info_list)

    @property
    def arxiv_url(self) -> str:
        """Return the arXiv URL."""
        if not self._arxiv_url:
            raise ValueError("arXiv URL is required.")
        return self._arxiv_url

    @property
    def llm(self) -> LLM_TYPE:
        """Return the language model."""
        return self._llm or self.get_llm()

    @property
    def paper(self) -> Paper:
        """Return the arXiv paper."""
        if self._paper is None:
            self._paper = self.get_paper()
        return self._paper

    @property
    def image_path_list(self) -> list[ImagePath]:
        """Return the list of image paths."""
        return self._image_path_list

    @property
    def elements(self) -> list[Element]:
        """Return the partition elements of the paper."""
        if self._elements is None:
            self._elements = self.get_partition_elements()
        return self._elements

    def get_llm(self) -> LLM_TYPE:
        """Get the language model."""
        return OpenAI(
            api_key=get_env_var("OPENAI_API_KEY"),
            base_url=get_env_var("OPENAI_BASE_URL"),
        )

    def get_paper(self) -> Paper:
        """Get the arXiv paper."""
        papers = fetch_papers_by_url(self.arxiv_url)
        if not papers:
            raise ValueError(f"No paper found for URL '{self.arxiv_url}'.")

        if len(papers) > 1:
            warnings.warn(f"Multiple papers found for URL '{self.arxiv_url}'. Using the first one. ")

        return papers[0]

    def get_partition_elements(self) -> list[Element]:
        """Get the partition elements of the paper."""
        file = load_paper_as_file_by_url(self.arxiv_url)
        elements = partition_pdf(
            file=file,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self._image_output_dir,
        )
        return elements

    def get_image_path_list(self) -> list[ImagePath]:
        """Get the list of image paths."""
        image_path_list: list[ImagePath] = []
        for element in self.elements:
            element_dict = element.to_dict()
            if element_dict["type"] == "Image" or element_dict["type"] == "Table":
                img_path = element_dict["metadata"]["image_path"]
                filename = Path(img_path).stem
                image_path_list.append(ImagePath(path=str(img_path), filename=str(filename)))
        return image_path_list

    def get_image_filename_set(self) -> set[str]:
        """Get the set of image paths."""
        return set([image_path.filename for image_path in self.image_path_list])

    def get_section_info_list(self, section_list: list[ExtractedSectionResult]) -> list[SectionInfo]:
        """Get the section info of the paper."""
        result = []
        for section in section_list:
            if section.content == "":
                continue

            image_content = []
            for filename in section.ref_fig + section.ref_tb:
                if filename not in self.get_image_filename_set():
                    continue
                image_path = Path(self._image_output_dir) / f"{filename}.jpg"
                image_content.append(encode_image(str(image_path)))

            result.append(
                SectionInfo(
                    title=section.section,
                    content=section.content,
                    image_content=image_content,
                )
            )
        return TypeAdapter(list[SectionInfo]).validate_python(result)

    def extract_section_list(self, text: str) -> list[ExtractedSectionResult]:
        """Extract the sections from the content of paper."""
        json_str = self._extract_paper_sections(text)  # type: ignore
        json_obj = extract_json_content(json_str)
        return TypeAdapter(list[ExtractedSectionResult]).validate_python(json_obj)

    @openai_prompt(
        ("system", "You are a helpful AI assistant."),
        (
            "user",
            "## Task:\n"
            "Given the content of a paper in triple backticks, organize each section into a JSON object format where each element contains:\n"
            "- `section`: The name of the section, capturing the main topic or heading of the section.\n"
            "- `content`: The full content of the section, presented exactly as in the paper without reduction or summarization.\n"
            "- `ref_fig`: A list of figure references in this section. "
            "Each reference must follow the format 'figure-<page>-<number>' (e.g., 'figure-20-7' for the seventh figure on page 20).\n"
            "- `ref_tb`: A list of table references in this section. "
            "Each reference must follow the format 'table-<page>-<number>' (e.g., 'table-15-3' for the third table on page 15).\n\n"
            "Ensure that each section is represented as a structured JSON object, without reducing or summarizing the content. "
            "Do not include the references section.\n",
        ),
        (
            "user",
            '### Response Format:\n```json\n[{{"section": "str", "content": "str", "ref_fig": ["str"], "ref_tb": ["str"]}}]\n```',
        ),
        ("user", "Paper content: ```{text}```"),
        model_name="gpt-4o-mini",
    )
    def _extract_paper_sections(self, text: str) -> str: ...  # type: ignore[empty-body]
