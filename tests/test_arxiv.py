import pytest
from unittest.mock import patch, MagicMock, mock_open
from arxiv_paper_summarizer.error import InvalidArxivURLException
from arxiv_paper_summarizer.arxiv import (
    ArxivClient,
    fetch_papers_by_url,
    fetch_papers_by_query,
    load_papers_by_url,
    load_papers_by_query,
    load_paper_as_file_by_url,
    extract_refs,
    Paper,
)


def test_parse_arxiv_id():
    client = ArxivClient()

    # Valid cases
    assert client.parse_arxiv_id("https://arxiv.org/abs/2404.01475") == "2404.01475"
    assert client.parse_arxiv_id("https://arxiv.org/abs/2404.01475v1") == "2404.01475"

    # Invalid case
    with pytest.raises(InvalidArxivURLException):
        client.parse_arxiv_id("https://invalid.url/not_an_arxiv_id")


def test_parse_references():
    client = ArxivClient()
    text = "Refer to https://arxiv.org/abs/1234.56789 and https://arxiv.org/abs/9876.54321v2."
    expected = ["https://arxiv.org/abs/1234.56789", "https://arxiv.org/abs/9876.54321v2"]
    assert client.parse_references(text) == expected


def test_fetch_papers_by_url_empty():
    client = ArxivClient()
    assert client.fetch_papers_by_url([]) == []


def test_fetch_papers_by_url():
    client = ArxivClient()
    urls = ["https://arxiv.org/abs/1234.56789"]
    mock_paper = MagicMock()
    mock_paper.title = "Sample Paper"
    mock_paper.entry_id = "https://arxiv.org/abs/1234.56789"
    mock_paper.download_pdf.return_value = "/tmp/sample.pdf"

    with patch.object(client, "search_by_url", return_value=[mock_paper]):
        with patch("fitz.open", return_value=[MagicMock(get_text=lambda: "Paper content")]):
            papers = client.fetch_papers_by_url(urls)
            assert len(papers) == 1
            assert papers[0].title == "Sample Paper"


def test_fetch_papers_with_references_by_url():
    client = ArxivClient()
    urls = ["https://arxiv.org/abs/1234.56789"]
    main_paper = MagicMock()
    main_paper.title = "Main Paper"
    main_paper.entry_id = "https://arxiv.org/abs/1234.56789"
    main_paper.download_pdf.return_value = "/tmp/main.pdf"

    ref_paper = MagicMock()
    ref_paper.title = "Reference Paper"
    ref_paper.entry_id = "https://arxiv.org/abs/9876.54321"
    ref_paper.download_pdf.return_value = "/tmp/ref.pdf"

    with patch.object(client, "search_by_url", side_effect=[[main_paper], [ref_paper]]):
        with patch(
            "fitz.open", return_value=[MagicMock(get_text=lambda: "Content with https://arxiv.org/abs/9876.54321")]
        ):
            papers = client.fetch_papers_with_references_by_url(urls)
            assert len(papers) == 1
            assert papers[0].references is not None
            assert len(papers[0].references) == 1


def test_fetch_papers_by_query():
    client = ArxivClient()
    queries = ["Sample Query"]
    mock_paper = MagicMock()
    mock_paper.title = "Sample Query"
    mock_paper.entry_id = "https://arxiv.org/abs/1234.56789"
    mock_paper.download_pdf.return_value = "/tmp/sample.pdf"

    with patch.object(client, "search_by_query", return_value=[mock_paper]):
        with patch("fitz.open", return_value=[MagicMock(get_text=lambda: "Paper content")]):
            papers = client.fetch_papers_by_query(queries)
            assert len(papers) == 1
            assert papers[0].title == "Sample Query"


def test_fetch_papers_with_references_by_query():
    client = ArxivClient()
    queries = ["Sample Query"]
    main_paper = MagicMock()
    main_paper.title = "Sample Query"
    main_paper.entry_id = "https://arxiv.org/abs/1234.56789"
    main_paper.download_pdf.return_value = "/tmp/main.pdf"

    ref_paper = MagicMock()
    ref_paper.title = "Reference Paper"
    ref_paper.entry_id = "https://arxiv.org/abs/9876.54321"
    ref_paper.download_pdf.return_value = "/tmp/ref.pdf"

    with patch.object(client, "search_by_query", return_value=[main_paper]):
        with patch.object(
            client,
            "fetch_papers_by_url",
            return_value=[Paper(title="Reference Paper", text="Content", url="https://arxiv.org/abs/9876.54321")],
        ):
            with patch(
                "fitz.open", return_value=[MagicMock(get_text=lambda: "Content with https://arxiv.org/abs/9876.54321")]
            ):
                papers = client.fetch_papers_with_references_by_query(queries)
                assert len(papers) == 1
                assert papers[0].references is not None
                assert len(papers[0].references) == 1


def test_download_papers_by_url():
    client = ArxivClient()
    urls = ["https://arxiv.org/abs/1234.56789"]
    mock_paper = MagicMock()
    mock_paper.title = "Sample Paper"
    mock_paper.download_pdf = MagicMock()

    with patch.object(client, "search_by_url", return_value=[mock_paper]):
        client.download_papers_by_url(urls, save_dir="/tmp")
        mock_paper.download_pdf.assert_called_once()


def test_download_papers_by_query():
    client = ArxivClient()
    queries = ["Sample Query"]
    mock_paper = MagicMock()
    mock_paper.title = "Sample Query"
    mock_paper.download_pdf = MagicMock()

    with patch.object(client, "search_by_query", return_value=[mock_paper]):
        client.download_papers_by_query(queries, save_dir="/tmp")
        mock_paper.download_pdf.assert_called_once()


def test_load_paper_as_file_by_url():
    client = ArxivClient()
    url = "https://arxiv.org/abs/1234.56789"
    mock_paper = MagicMock()
    mock_paper.download_pdf.return_value = "/tmp/sample.pdf"

    with patch.object(client, "search_by_url", return_value=[mock_paper]):
        m_open = mock_open(read_data=b"PDF content")
        with patch("builtins.open", m_open):
            file_obj = client.load_paper_as_file_by_url(url)
            assert file_obj.read() == b"PDF content"


def test_fetch_papers_by_url_function():
    urls = ["https://arxiv.org/abs/1234.56789"]
    with patch("arxiv_paper_summarizer.arxiv.arxiv_client") as mock_client:
        fetch_papers_by_url(urls)
        mock_client.fetch_papers_by_url.assert_called_once_with(urls)


def test_fetch_papers_by_query_function():
    queries = ["Sample Query"]
    with patch("arxiv_paper_summarizer.arxiv.arxiv_client") as mock_client:
        fetch_papers_by_query(queries)
        mock_client.fetch_papers_by_query.assert_called_once_with(queries)


def test_load_papers_by_url_function():
    urls = ["https://arxiv.org/abs/1234.56789"]
    with patch("arxiv_paper_summarizer.arxiv.arxiv_client") as mock_client:
        load_papers_by_url(urls, save_dir="/tmp")
        mock_client.download_papers_by_url.assert_called_once_with(urls, "/tmp")


def test_load_papers_by_query_function():
    queries = ["Sample Query"]
    with patch("arxiv_paper_summarizer.arxiv.arxiv_client") as mock_client:
        load_papers_by_query(queries, save_dir="/tmp")
        mock_client.download_papers_by_query.assert_called_once_with(queries, "/tmp")


def test_load_paper_as_file_by_url_function():
    url = "https://arxiv.org/abs/1234.56789"
    with patch("arxiv_paper_summarizer.arxiv.arxiv_client") as mock_client:
        load_paper_as_file_by_url(url)
        mock_client.load_paper_as_file_by_url.assert_called_once_with(url)


def test_extract_refs():
    paper1 = Paper(title="Paper 1", text="Text 1", url="https://example.com/url1")
    paper2 = Paper(title="Paper 2", text="Text 2", url="https://example.com/url2", references=[paper1])
    papers = [paper2]
    refs = extract_refs(papers)
    assert len(refs) == 2
    assert paper1 in refs
    assert paper2 in refs
