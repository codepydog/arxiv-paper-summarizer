"""AI Summarizer for arXiv papers."""

import json
import tempfile
import warnings
from pathlib import Path

from pydantic import TypeAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt
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
    SectionNote,
    SummaryResult,
)
from arxiv_paper_summarizer.utils import (
    encode_image,
    extract_json_content,
    get_env_var,
    is_first_figure,
    normalize_image_filename,
)

try:
    from langfuse.openai import OpenAI
except ImportError:
    from openai import OpenAI


class ArxivPaperSummarizer:
    """Summarizer for arXiv papers."""

    def __init__(
        self,
        arxiv_url: str,
        llm: LLM_TYPE | None = None,
        extract_section_notes: bool = False,
        verbose: bool = False,  # TODO: Implement verbose mode
    ) -> None:
        """Initialize the summarizer."""
        self._arxiv_url = arxiv_url
        self._llm = llm
        self._extract_section_notes = extract_section_notes
        self._verbose = verbose

        self._paper: Paper | None = None
        self._elements: list[Element] | None = None

        self._img_path_map: dict[str, str] = {}
        self._image_path_list: list[ImagePath] = []
        self._image_output_dir = tempfile.mkdtemp()

        self._post_init()

    def _post_init(self):
        self._paper = self.get_paper()
        self._elements = self.get_partition_elements()
        self._image_path_list = self.get_image_path_list()

    def summarize(self) -> SummaryResult:
        """Summarize the arXiv paper."""
        section_list = self.extract_section_list(self.paper.text)
        section_info_list = self.get_section_info_list(section_list)
        keynote = self.extract_keynote()

        if self._extract_section_notes:
            section_notes = self.extract_section_note_list(section_info_list)
        else:
            section_notes = []

        return SummaryResult(keynote=keynote, section_notes=section_notes)

    def extract_keynote(self) -> str:
        """Extract the keynote of the paper."""
        try:
            return self._extract_keynote(self.paper.text)  # type: ignore
        except Exception as e:
            warnings.warn(f"Error in extracting keynote: {e}")
            return ""

    def extract_section_note_list(self, section_info_list: list[SectionInfo]) -> list[SectionNote]:
        """Extract the section notes of the paper."""
        try:
            section_note_list = [self._write_section_note(section) for section in section_info_list]
            return TypeAdapter(list[SectionNote]).validate_python(section_note_list)
        except Exception as e:
            warnings.warn(f"Error in writing comprehensive analysis note: {e}")
            return []

    def _write_section_note(self, section: SectionInfo) -> SectionNote:
        """Summarize the section."""
        summary = self.summarize_section(text=section.content, title=section.title)  # type: ignore
        img_summaries = [self.summarize_images(img_enc, summary) for img_enc in section.image_encoding_str_list]  # type: ignore
        structured_section_summary = self.organize_section_summary(summary, img_summaries)  # type: ignore
        quotes = self.extract_section_quotes(section.content, title=section.title)  # type: ignore
        return SectionNote(
            header=section.title,
            summary_content=structured_section_summary,
            quotes=quotes,
            image_paths=section.image_paths,
            table_paths=section.table_paths,
        )

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
                filename = str(Path(img_path).stem)
                image_path_list.append(ImagePath(path=str(img_path), filename=filename))
        return image_path_list

    def get_image_filename_set(self) -> set[str]:
        """Get the set of image paths."""
        filename_list: list[str] = []
        for image_path in self.image_path_list:
            norm_filename = normalize_image_filename(image_path.filename)
            self._img_path_map[norm_filename] = image_path.filename  # Need this map to get the original filename
            filename_list.append(norm_filename)
        return set(filename_list)

    def get_section_info_list(self, section_list: list[ExtractedSectionResult]) -> list[SectionInfo]:
        """Get the section info of the paper."""
        result = []
        for section in section_list:
            if section.content == "":
                continue

            image_paths = []
            table_paths = []
            image_encoding_str_list = []
            for filename in section.ref_fig + section.ref_tb:
                if filename not in self.get_image_filename_set():
                    continue

                image_path = str(Path(self._image_output_dir) / f"{self.get_image_filename(filename)}.jpg")
                if filename.startswith("figure"):
                    image_paths.append(image_path)
                elif filename.startswith("table"):
                    table_paths.append(image_path)

                img_enc = encode_image(image_path)
                image_encoding_str_list.append(img_enc)

            result.append(
                SectionInfo(
                    title=section.section,
                    content=section.content,
                    image_encoding_str_list=image_encoding_str_list,
                    image_paths=image_paths,
                    table_paths=table_paths,
                )
            )
        return TypeAdapter(list[SectionInfo]).validate_python(result)

    def get_image_filename(self, norm_filename: str) -> str:
        """Get the original image filename."""
        try:
            return self._img_path_map[norm_filename]
        except KeyError as e:
            warnings.warn(f"No image found for filename '{norm_filename}': {e}")
            return ""

    def get_cover_image_path(self) -> str | None:
        """Get the default cover image path."""
        for img_path in self.image_path_list:
            if is_first_figure(img_path.path):
                return img_path.path
        return None

    @retry(
        stop=stop_after_attempt(3),
        retry=(retry_if_exception_type(json.JSONDecodeError) | retry_if_exception_type(ValueError)),
    )
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
            "Each reference must follow the format 'figure-<number>' (e.g., 'figure-1' for the first figure).\n"
            "- `ref_tb`: A list of table references in this section. "
            "Each reference must follow the format 'table-<number>' (e.g., 'table-3' for the third table).\n\n"
            "Ensure that each section is represented as a structured JSON object, without reducing or summarizing the content. "
            "Do not include the references section.\n"
            "Do not duplicate the ref_fig and ref_tb to each section. Include them only once.",
        ),
        (
            "user",
            '### Response Format:\n```json\n[{{"section": "str", "content": "str", "ref_fig": ["str"], "ref_tb": ["str"]}}]\n```',
        ),
        ("user", "Paper content: ```{text}```"),
        model_name="gpt-4o-mini",
    )
    def _extract_paper_sections(self, text: str) -> str: ...  # type: ignore[empty-body]

    @openai_prompt(
        ("system", "You are a AI Research."),
        (
            "user",
            "I am reading a machine learning and deep learning paper and will provide you with a section of its content. "
            "Provide a brief summary of the section.",
        ),
        ("user", "## Response Format\n## {title}\n```{text}```"),
        ("user", "## Title: {title}\nContent:\n```\n{text}\n```"),
        model_name="claude-4-sonnet",  # type: ignore[arg-type]
    )
    def summarize_section(self, text: str, title: str) -> str: ...  # type: ignore[empty-body]

    @openai_prompt(
        ("system", "You are an AI research assistant."),
        (
            "user",
            "## Task\n"
            "I am reading a machine learning and deep learning paper and will provide you with a section of its content. "
            "Extract only the essential quotes that capture the key information from this section, as follows:\n"
            "- Include quotes that highlight the primary problem or question addressed.\n"
            "- Add quotes describing any proposed methods or solutions, along with theoretical foundations or significant insights.\n"
            "- Provide quotes on any major findings or important points emphasized by the author.\n"
            "## Response Format\n"
            "> 'Quote text here'\n\n"
            "## Requirements\n"
            "- Limit to ONLY three critical quotes.\n"
            "- Quotes should be very critical or insightful or innovative.\n"
            "- If the title is related to Abstract, References or Conclusion, return 'NO_QUOTES'.\n"
            "- If the entire section is unimportant, return 'NO_QUOTES'",
        ),
        ("user", "## Title: {title}\nContent:\n```\n{text}\n```"),
        model_name="claude-4-sonnet",  # type: ignore[arg-type]
    )
    def extract_section_quotes(self, text: str, title: str) -> str: ...  # type: ignore[empty-body]

    def summarize_images(self, base64_img: str, section_summary: str) -> str:
        content = []
        content += [
            {
                "type": "text",
                "text": "Given the image of an paper, explain the key insights and findings from the image.",
            }
        ]
        content += [{"type": "text", "text": section_summary}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]  # type: ignore

        response = self.llm.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {  # type: ignore
                    "role": "user",
                    "content": content,
                }
            ],
        )  # type: ignore
        return response.choices[0].message.content  # type: ignore

    @openai_prompt(
        ("system", "You are an AI research assistant."),
        (
            "user",
            "## Task\n"
            "Organize the provided section summary, and image summaries into a structured format with bullet points:\n\n"
            "- Display each line of the summary as a bullet point.\n\n"
            "- Include the image summary as a bullet point if it not empty.\n\n",
        ),
        ("user", "Section Summary: ```{summary}```"),
        ("user", "Image Summary: ```{image_summary}```"),
        model_name="gpt-4o",
    )
    def organize_section_summary(self, summary: str, image_summary: list[str]): ...  # type: ignore[empty-body]

    @openai_prompt(
        ("system", "You are an AI research assistant."),
        (
            "user",
            "I am reading deep learning and AI research papers and need structured notes based on specific sections of the content. "
            "For each section provided, focus on the following points:\n\n"
            "### Problem\n"
            "- What problem does this paper aim to solve?\n"
            "- What are the existing methods, and what limitations do they have?\n\n"
            "### Solution\n"
            "- What solution does the paper propose?\n"
            "- What inspired this idea? Was it influenced by other papers?\n"
            "- What theoretical basis supports this method?\n\n"
            "### Experiment\n"
            "- How well does the experiment perform?\n"
            "- What limitations or assumptions are associated with this method?\n\n"
            "### Innovation\n"
            "- What important or novel discoveries does this paper make?\n\n"
            "### Comments / Critique\n"
            "- Are there any limitations in this paper?\n"
            "- Does the paper substantiate its claims effectively?\n\n"
            "Do NOT include any content outside this format.\n",
        ),
        ("user", "Section Content: ```{text}```"),
        model_name="claude-4-sonnet",  # type: ignore[arg-type]
    )
    def _extract_keynote(self, text: str) -> str: ...  # type: ignore[empty-body]
