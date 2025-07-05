#!/usr/bin/env python3
"""Script for generating a single arXiv paper report via GitHub Actions."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from arxiv_paper_summarizer import ArxivPaperSummarizer
from arxiv_paper_summarizer.report import ReportGenerator


def generate_report(arxiv_url: str, language: str = "English", mode: str = "simple") -> Path:
    """Generate report for a single arXiv paper.

    Args:
        arxiv_url: The arXiv paper URL
        language: Report language ("English" or "Traditional Chinese")
        mode: Report mode ("simple" or "detailed")

    Returns:
        Path to the generated PDF report
    """
    print(f"Starting to summarize the paper: {arxiv_url}")
    print(f"Language: {language}")
    print(f"Mode: {mode}")

    # Convert mode to extract_section_notes boolean
    extract_section_notes = mode == "detailed"
    print(f"Extract section notes: {extract_section_notes}")

    # Create papers directory with year/week structure like the example
    papers_base_dir = Path("papers")
    weekly_dir = papers_base_dir / f"{datetime.now():%Y}/week_{datetime.now().isocalendar()[1]:02}"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize summarizer
        summarizer = ArxivPaperSummarizer(
            arxiv_url=arxiv_url,
            extract_section_notes=extract_section_notes,
        )

        # Generate summary
        summary_result = summarizer.summarize()
        print("Summary generated successfully")

        # Generate report
        print("Starting to generate the report")
        report = ReportGenerator(
            paper=summarizer.paper,
            keynote=summary_result.keynote,
            section_notes=summary_result.section_notes,
            language=language,
            cover_path=summarizer.get_cover_image_path(),
        )

        # Create output filename with mode indicator
        safe_title = "".join(c for c in summarizer.paper.title if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_title = safe_title.replace(" ", "_")[:80]  # Limit length

        language_code = "TC" if language == "Traditional Chinese" else "EN"
        mode_code = "detailed" if mode == "detailed" else "simple"
        output_filename = f"{safe_title}_{language_code}_{mode_code}.pdf"
        output_path = weekly_dir / output_filename

        # Generate PDF report
        report.generate_pdf_report(output_path=output_path)
        print(f"Report successfully generated at {output_path}")

        return output_path

    except Exception as e:
        print(f"Error generating report: {e}")
        raise


def main():
    """Main function to parse arguments and generate report."""
    parser = argparse.ArgumentParser(
        description="Generate arXiv paper report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_single_report.py https://arxiv.org/abs/2410.20672
  python scripts/generate_single_report.py https://arxiv.org/abs/2410.20672 --language "Traditional Chinese" --mode detailed
        """,
    )

    parser.add_argument("url", help="ArXiv paper URL (e.g., https://arxiv.org/abs/2410.20672)")

    parser.add_argument("--language", choices=["English", "Traditional Chinese"], default="English", help="Report language (default: English)")

    parser.add_argument(
        "--mode",
        choices=["simple", "detailed"],
        default="simple",
        help="Report mode: simple (keynote only) or detailed (with section notes) (default: simple)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key in the environment or .env file")
        sys.exit(1)

    try:
        output_path = generate_report(args.url, args.language, args.mode)

        # Output information for GitHub Actions
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"report_path={output_path}\n")
                f.write(f"report_filename={output_path.name}\n")

        print(f"\n✅ Report generation completed successfully!")
        print(f"📄 Generated file: {output_path}")

    except Exception as e:
        print(f"\n❌ Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
