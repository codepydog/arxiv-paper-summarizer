#!/usr/bin/env python3
"""Script for generating a single arXiv paper report via GitHub Actions."""

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from arxiv_paper_summarizer import ArxivPaperSummarizer
from arxiv_paper_summarizer.report import ReportGenerator
from arxiv_paper_summarizer.utils import get_publication_week_folder


def validate_environment():
    """Validate that all required environment variables and dependencies are available."""
    print("🔍 Validating environment...")

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    print(f"✅ OpenAI API key found (length: {len(api_key)})")

    # Check OpenAI base URL (optional)
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        print(f"✅ OpenAI base URL: {base_url}")
    else:
        print("ℹ️  Using default OpenAI base URL")

    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    print("✅ Environment validation completed")


def generate_report(arxiv_url: str, language: str = "English", mode: str = "simple") -> Path:
    """Generate report for a single arXiv paper.

    Args:
        arxiv_url: The arXiv paper URL
        language: Report language ("English" or "Traditional Chinese")
        mode: Report mode ("simple" or "detailed")

    Returns:
        Path to the generated PDF report
    """
    print(f"🚀 Starting to summarize the paper: {arxiv_url}")
    print(f"📝 Language: {language}")
    print(f"⚙️  Mode: {mode}")

    # Convert mode to extract_section_notes boolean
    extract_section_notes = mode == "detailed"
    print(f"🔧 Extract section notes: {extract_section_notes}")

    # Base papers directory
    papers_base_dir = Path("papers")

    summarizer = None
    summary_result = None
    report = None

    try:
        # Step 1: Initialize summarizer
        print("\n📋 Step 1: Initializing ArXiv paper summarizer...")
        try:
            summarizer = ArxivPaperSummarizer(
                arxiv_url=arxiv_url,
                extract_section_notes=extract_section_notes,
            )
            print(f"✅ Summarizer initialized successfully")
            print(f"📄 Paper title: {summarizer.paper.title}")
            print(f"👥 Authors: {', '.join(summarizer.paper.authors)}")
            print(f"📅 Published: {summarizer.paper.published}")

            # Use paper's publication date to determine folder structure
            weekly_dir = get_publication_week_folder(summarizer.paper.published, papers_base_dir)
            weekly_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory created: {weekly_dir}")

        except Exception as e:
            print(f"❌ ERROR in Step 1 - Summarizer initialization failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize summarizer: {e}") from e

        # Step 2: Generate summary
        print("\n📋 Step 2: Generating paper summary...")
        try:
            summary_result = summarizer.summarize()
            print(f"✅ Summary generated successfully")
            print(f"📝 Keynote length: {len(summary_result.keynote)} characters")
            print(f"📊 Section notes count: {len(summary_result.section_notes)}")
        except Exception as e:
            print(f"❌ ERROR in Step 2 - Summary generation failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate summary: {e}") from e

        # Step 3: Initialize report generator
        print("\n📋 Step 3: Initializing report generator...")
        try:
            cover_path = summarizer.get_cover_image_path()
            print(f"🖼️  Cover image path: {cover_path}")

            report = ReportGenerator(
                paper=summarizer.paper,
                keynote=summary_result.keynote,
                section_notes=summary_result.section_notes,
                language=language,
                cover_path=cover_path,
            )
            print(f"✅ Report generator initialized successfully")
        except Exception as e:
            print(f"❌ ERROR in Step 3 - Report generator initialization failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize report generator: {e}") from e

        # Step 4: Create output filename
        print("\n📋 Step 4: Creating output filename...")
        try:
            safe_title = "".join(c for c in summarizer.paper.title if c.isalnum() or c in (" ", "-", "_")).strip()
            safe_title = safe_title.replace(" ", "_")[:80]  # Limit length

            language_code = "TC" if language == "Traditional Chinese" else "EN"
            mode_code = "detailed" if mode == "detailed" else "simple"
            output_filename = f"{safe_title}_{language_code}_{mode_code}.pdf"
            output_path = weekly_dir / output_filename

            print(f"✅ Output filename created: {output_filename}")
            print(f"📄 Full output path: {output_path}")
        except Exception as e:
            print(f"❌ ERROR in Step 4 - Filename creation failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to create output filename: {e}") from e

        # Step 5: Generate PDF report
        print("\n📋 Step 5: Generating PDF report...")
        try:
            report.generate_pdf_report(output_path=output_path)
            print(f"✅ PDF report generated successfully")

            # Verify file was created
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"📊 Generated file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            else:
                raise FileNotFoundError(f"Output file was not created: {output_path}")

        except Exception as e:
            print(f"❌ ERROR in Step 5 - PDF generation failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate PDF report: {e}") from e

        print(f"\n🎉 Report successfully generated at {output_path}")
        return output_path

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in generate_report function:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   ArXiv URL: {arxiv_url}")
        print(f"   Language: {language}")
        print(f"   Mode: {mode}")

        # Add debugging information about the current state
        if summarizer:
            try:
                print(f"   Paper title: {summarizer.paper.title}")
            except:
                print("   Could not access paper title")

        if summary_result:
            print(f"   Summary keynote length: {len(summary_result.keynote) if summary_result.keynote else 0}")
            print(f"   Section notes count: {len(summary_result.section_notes) if summary_result.section_notes else 0}")

        print(f"\n🔍 Full error traceback:")
        traceback.print_exc()
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

    print("=" * 80)
    print("🚀 ArXiv Paper Report Generator")
    print("=" * 80)

    try:
        # Validate environment before processing
        validate_environment()

        print(f"\n📋 Processing Parameters:")
        print(f"   📄 ArXiv URL: {args.url}")
        print(f"   🌍 Language: {args.language}")
        print(f"   ⚙️  Mode: {args.mode}")

        # Generate the report
        output_path = generate_report(args.url, args.language, args.mode)

        # Output information for GitHub Actions
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"report_path={output_path}\n")
                f.write(f"report_filename={output_path.name}\n")
            print(f"✅ GitHub Actions output variables set")

        print(f"\n" + "=" * 80)
        print(f"🎉 SUCCESS: Report generation completed!")
        print(f"📄 Generated file: {output_path}")
        print("=" * 80)

    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"💥 FATAL ERROR: Report generation failed!")
        print(f"=" * 80)
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Error message: {str(e)}")
        print(f"📋 Input parameters:")
        print(f"   📄 ArXiv URL: {args.url}")
        print(f"   🌍 Language: {args.language}")
        print(f"   ⚙️  Mode: {args.mode}")

        print(f"\n🔍 Full error details:")
        traceback.print_exc()

        print(f"\n💡 Troubleshooting tips:")
        print(f"   1. Verify the ArXiv URL is valid and accessible")
        print(f"   2. Check your internet connection")
        print(f"   3. Ensure OpenAI API key is valid and has sufficient credits")
        print(f"   4. Try running with a different paper URL")
        print("=" * 80)

        sys.exit(1)


if __name__ == "__main__":
    main()
