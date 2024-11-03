"""Example of generating a arXiv paper report."""

from dotenv import load_dotenv
from arxiv_paper_summarizer import ArxivPaperSummarizer
from arxiv_paper_summarizer.report import ReportGenerator

load_dotenv()

ARXIV_URL = "https://arxiv.org/abs/2401.18059v1"
# ARXIV_URL = "https://arxiv.org/abs/2410.23123"
LANGUAGE = "Traditional Chinese"
# LANGUAGE = "English"


def main():
    print("Starting to summarize the paper")
    paper_summarizer = ArxivPaperSummarizer(
        arxiv_url=ARXIV_URL,
        extract_section_notes=True,
    )
    summary_result = paper_summarizer.summarize()

    print("Starting to generate the report for the paper")
    report_generator = ReportGenerator(
        paper=paper_summarizer.paper,
        keynote=summary_result.keynote,
        section_notes=summary_result.section_notes,
        language=LANGUAGE,
        cover_path=paper_summarizer.get_cover_image_path(),
    )

    paper_title = paper_summarizer.paper.title
    report_generator.generate_pdf_report(output_path=f"{paper_title} ({LANGUAGE}).pdf")
    print(f"Report successfully generated at {paper_title}.pdf")


if __name__ == "__main__":
    main()
