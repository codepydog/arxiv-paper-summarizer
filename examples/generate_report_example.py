"""Example of generating a arXiv paper report."""

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from arxiv_paper_summarizer import ArxivPaperSummarizer
from arxiv_paper_summarizer.report import ReportGenerator

load_dotenv()

ARXIV_URL_LIST = [
    # "https://arxiv.org/abs/2410.20672",
    # "https://arxiv.org/abs/2410.10762",
    # "https://arxiv.org/abs/2410.21943",
    # "https://arxiv.org/pdf/2410.22071",
    # "https://arxiv.org/abs/2410.18982",
    # "https://arxiv.org/abs/2410.19750",
    # "https://arxiv.org/abs/2410.19385",
    "https://arxiv.org/abs/2506.21495",
]
LANGUAGE = "Traditional Chinese"
PAPER_SAVE_DIR = Path("papers")
EXTRACT_SECTION_NOTES = True


def generate_report(arxiv_url, language, save_dir):
    print(f"Starting to summarize the paper {arxiv_url}")
    summarizer = ArxivPaperSummarizer(
        arxiv_url=arxiv_url,
        extract_section_notes=EXTRACT_SECTION_NOTES,
    )
    summary_result = summarizer.summarize()

    print("Starting to generate the report for the paper")
    report = ReportGenerator(
        paper=summarizer.paper,
        keynote=summary_result.keynote,
        section_notes=summary_result.section_notes,
        language=language,
        cover_path=summarizer.get_cover_image_path(),
    )

    output_path = save_dir / f"{summarizer.paper.title} ({language}).pdf"
    report.generate_pdf_report(output_path=output_path)
    print(f"Report successfully generated at {output_path}")


def main():
    weekly_dir = PAPER_SAVE_DIR / f"{datetime.now():%Y}/week_{datetime.now().isocalendar()[1]:02}"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    for url in ARXIV_URL_LIST:
        try:
            generate_report(url, LANGUAGE, save_dir=weekly_dir)
        except Exception as e:
            print(f"Failed to generate report for {url}: {e}")
    print("All reports generated successfully!")


if __name__ == "__main__":

    main()
