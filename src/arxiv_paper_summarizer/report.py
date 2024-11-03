"""arXiv report generated module."""

from pathlib import Path
from typing import cast

import markdown2
from weasyprint import HTML

from arxiv_paper_summarizer.translation import translate_quote, translate_text
from arxiv_paper_summarizer.types import Language, Paper, SectionNote
from arxiv_paper_summarizer.utils import encode_image


class ReportGenerator:
    """Arxiv Paper Report Generator."""

    _QUOTES_FLAG = "NO_QUOTES"

    def __init__(
        self,
        paper: Paper,
        keynote: str | None = None,
        section_notes: list[SectionNote] | None = None,
        language: str | Language | None = None,
        cover_path: str | Path | None = None,
    ) -> None:
        self._paper = paper
        self._keynote = keynote
        self._section_notes = section_notes
        self._language = language
        self._cover_path = cover_path

        self._image_path_set: set[str] = set()

    @property
    def paper(self) -> Paper:
        return self._paper

    @property
    def keynote(self) -> str:
        if self._keynote is None:
            raise ValueError("Keynote is not set.")
        return self._keynote

    @property
    def section_notes(self) -> list[SectionNote]:
        if self._section_notes is None:
            raise ValueError("Section notes are not set.")
        return self._section_notes

    @property
    def language(self) -> str:
        if self._language is None:
            self._language = Language.ENGLISH.value
        elif isinstance(self._language, Language):
            self._language = self._language.value
        elif isinstance(self._language, str):
            try:
                self._language = Language(self._language).value
            except ValueError as e:
                raise ValueError(f"Language '{self._language}' is not supported. Please use one of {Language.__members__}") from e
        return self._language

    @property
    def cover_path(self) -> str | None:
        if self._cover_path is None:
            return None
        return str(self._cover_path)

    def generate_pdf_report(self, output_path: str) -> None:
        report_content = self.generate_report_content()
        html_content = self.generate_html(report_content)
        html = HTML(string=html_content)
        html.write_pdf(str(output_path))

    def generate_report_content(self) -> str:
        report_content = ""
        title = self.paper.title
        arxiv_url = self.paper.url
        cover_content = self._generate_image_content([self.cover_path]) if self.cover_path else ""
        comprehensive_analysis_content = self.generate_comprehensive_analysis_content() if self.section_notes else "No section notes."

        report_content += (
            f"# {title} ([arxiv]({arxiv_url}))\n\n"
            f"## Key Highlights\n"
            f"{cover_content}\n"
            f"{self.generate_keynote_content()}\n\n"
            f"## Comprehensive Analysis\n"
            f"{comprehensive_analysis_content}\n\n"
            f"## References\n"
            f"{self.generate_reference_content()}\n\n"
        )
        return report_content

    def generate_keynote_content(self) -> str:
        return self.keynote if self.language == Language.ENGLISH.value else translate_text(self.keynote, self.language)

    def generate_comprehensive_analysis_content(self) -> str:
        content = ""
        for section in self.section_notes:
            immediate_content = self._generate_comprehensive_analysis_content(section)
            content += immediate_content
        return content

    def _generate_comprehensive_analysis_content(self, section: SectionNote) -> str:
        content = ""
        content += f"### {section.header}\n"
        content += self._generate_image_content(section.image_paths)
        content += f"{self._translate_text(section.summary_content, self.language)}\n\n"
        if section.quotes != self._QUOTES_FLAG:
            content += f"{self._translate_quote(section.quotes, self.language)}\n\n"
        content += self._generate_image_content(section.table_paths)
        return content

    def generate_reference_content(self) -> str:
        if self.paper.references is None:
            return "No references found."

        content = ""
        for reference in self.paper.references:
            content += f"- [{reference.title}]({reference.url})\n"
        return content

    def _generate_image_content(self, paths: list[str]) -> str:
        content = ""
        for path in paths:
            # ensure no duplicate image or table
            if path in self._image_path_set:
                continue
            encoded_image = encode_image(path)
            content += f'<img src="data:image/jpeg;base64,{encoded_image}" ' f'style="max-width:100%; height:auto;" alt="Image"/>\n\n'
            self._image_path_set.add(path)
        return content

    def _translate_text(self, text: str, language: str) -> str:
        if language == Language.ENGLISH.value:
            return text
        return cast(str, translate_text(text, language))

    def _translate_quote(self, text: str, language: str) -> str:
        if language == Language.ENGLISH.value:
            return text
        return cast(str, translate_quote(text, language))

    def generate_html(self, text: str):
        css = """
        blockquote {
            font-style: italic;
            color: #555555;
            padding: 10px 20px;
            margin: 20px 0;
            border-left: 4px solid #cccccc;
            background-color: #f9f9f9;
        }
        """
        html_body = markdown2.markdown(text)

        html_content = f"""
        <html>
        <head>
            <style>{css}</style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        return html_content
