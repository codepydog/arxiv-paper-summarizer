"""Windows setup helper for arxiv-paper-summarizer."""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path


def check_weasyprint_installation():
    """Check if WeasyPrint is properly installed."""
    try:
        import weasyprint

        print("✓ WeasyPrint is installed")
        return True
    except ImportError:
        print("✗ WeasyPrint is not installed")
        return False


def check_pango_installation():
    """Check if Pango is available for WeasyPrint."""
    try:
        from weasyprint import HTML

        # Try to create a simple HTML document
        html = HTML(string="<html><body><p>Test</p></body></html>")

        # Try to render to PDF in memory
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            html.write_pdf(temp_file.name)

        print("✓ Pango dependencies are properly configured")
        return True
    except Exception as e:
        print(f"✗ Pango dependencies issue: {e}")
        return False


def clean_temp_directories():
    """Clean up temporary directories that might be locked."""
    temp_dir = Path(tempfile.gettempdir())
    cleaned = 0

    for item in temp_dir.glob("tmp*"):
        if item.is_dir():
            try:
                shutil.rmtree(item, ignore_errors=True)
                cleaned += 1
            except Exception:
                pass

    print(f"✓ Cleaned {cleaned} temporary directories")


def test_pdf_generation():
    """Test PDF generation with a simple example."""
    try:
        from weasyprint import HTML

        html_content = """
        <html>
        <head><title>Test PDF</title></head>
        <body>
            <h1>Test PDF Generation</h1>
            <p>This is a test to verify WeasyPrint works correctly on Windows.</p>
        </body>
        </html>
        """

        html = HTML(string=html_content)
        test_file = Path("test_output.pdf")

        # Try multiple approaches
        max_retries = 3
        for attempt in range(max_retries):
            try:
                html.write_pdf(str(test_file))
                if test_file.exists():
                    print(f"✓ PDF generation test successful (attempt {attempt + 1})")
                    test_file.unlink()  # Clean up
                    return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    print(f"✗ PDF generation test failed after {max_retries} attempts: {e}")
                    return False

        return False
    except Exception as e:
        print(f"✗ PDF generation test failed: {e}")
        return False


def check_msys2_installation():
    """Check if MSYS2 is installed and Pango is available."""
    msys2_paths = [
        "C:\\msys64\\mingw64\\bin",
        "C:\\msys2\\mingw64\\bin",
    ]

    for path in msys2_paths:
        if Path(path).exists():
            print(f"✓ MSYS2 found at {path}")
            # Check if pango DLLs are present
            pango_libs = ["libpango-1.0-0.dll", "libpangocairo-1.0-0.dll"]
            all_found = True
            for lib in pango_libs:
                if not (Path(path) / lib).exists():
                    print(f"  ✗ Missing: {lib}")
                    all_found = False
                else:
                    print(f"  ✓ Found: {lib}")

            if all_found:
                return True

    print("✗ MSYS2 or Pango libraries not found")
    return False


def check_poppler_installation():
    """Check if Poppler is installed for PDF processing."""
    try:
        result = subprocess.run(["pdftoppm", "-h"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Poppler is installed and accessible")
            return True
    except FileNotFoundError:
        pass

    # Check common Poppler installation paths
    poppler_paths = [
        "C:\\Program Files\\poppler\\bin",
        "C:\\poppler\\bin",
        "C:\\msys64\\mingw64\\bin",
        "C:\\msys2\\mingw64\\bin",
    ]

    for path in poppler_paths:
        pdftoppm_path = Path(path) / "pdftoppm.exe"
        if pdftoppm_path.exists():
            print(f"✓ Poppler found at {path}")
            return True

    print("✗ Poppler not found. This is required for PDF processing.")
    return False


def print_installation_instructions():
    """Print installation instructions for Windows."""
    print("\n" + "=" * 60)
    print("WINDOWS SETUP INSTRUCTIONS FOR WEASYPRINT")
    print("=" * 60)
    print(
        """
If you're experiencing issues, please follow these steps:

1. Install MSYS2:
   - Download from: https://www.msys2.org/
   - Run installer with default options
   
2. Install Pango via MSYS2:
   - Open MSYS2 terminal
   - Run: pacman -S mingw-w64-x86_64-pango
   
3. Add MSYS2 to PATH:
   - Add C:\\msys64\\mingw64\\bin to your system PATH
   - Or add C:\\msys2\\mingw64\\bin (depending on install location)
   
4. Restart your terminal/IDE

5. Verify installation:
   - Run: python -m weasyprint --info
   
6. Alternative: Use Windows executable:
   - Download WeasyPrint executable from official releases
   - This avoids Python library dependencies

For more details, see: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows
"""
    )


def main():
    """Main diagnostic function."""
    print("Windows ArXiv Paper Summarizer Diagnostic Tool")
    print("=" * 50)

    # Clean temp directories first
    clean_temp_directories()

    # Check installations
    weasyprint_ok = check_weasyprint_installation()
    msys2_ok = check_msys2_installation() if weasyprint_ok else False
    pango_ok = check_pango_installation() if weasyprint_ok else False
    poppler_ok = check_poppler_installation()

    # Test PDF generation
    pdf_test_ok = test_pdf_generation() if weasyprint_ok and pango_ok else False

    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"WeasyPrint installed: {'✓' if weasyprint_ok else '✗'}")
    print(f"MSYS2/Pango setup: {'✓' if msys2_ok else '✗'}")
    print(f"Pango dependencies: {'✓' if pango_ok else '✗'}")
    print(f"Poppler installed: {'✓' if poppler_ok else '✗'}")
    print(f"PDF generation test: {'✓' if pdf_test_ok else '✗'}")

    if not all([weasyprint_ok, msys2_ok, pango_ok, poppler_ok, pdf_test_ok]):
        print_installation_instructions()
        if not poppler_ok:
            print("\nADDITIONAL: Install Poppler for PDF processing:")
            print("- Via MSYS2: pacman -S mingw-w64-x86_64-poppler")
            print("- Or download from: https://github.com/oschwartz10612/poppler-windows")
        return 1
    else:
        print("\n✓ All checks passed! ArXiv Paper Summarizer should work correctly.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
