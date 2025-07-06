"""Arxiv client tools."""

from __future__ import annotations

import io
import re
import ssl
import tempfile
from typing import Iterable, cast

import fitz
from arxiv import Client as _ArxivClient
from arxiv import Result as ArxivResult
from arxiv import Search as ArxivSearch
from more_itertools import flatten, unique_everseen

from arxiv_paper_summarizer.error import InvalidArxivURLException
from arxiv_paper_summarizer.types import Paper


class ArxivClient:

    def __init__(self) -> None:
        self.client = _ArxivClient()
        self.ensure_ssl_verified()

    @staticmethod
    def ensure_ssl_verified() -> None:
        ssl._create_default_https_context = ssl._create_unverified_context

    def search_by_url(self, id_list: list[str]) -> Iterable[ArxivResult]:
        search = ArxivSearch(id_list=id_list)
        yield from self.client.results(search=search)

    def search_by_query(self, queries: Iterable[str]) -> Iterable[ArxivResult]:
        for query in queries:
            search = ArxivSearch(query=query, max_results=1)
            yield from self.client.results(search=search)

    @staticmethod
    def parse_arxiv_id(url: str) -> str:
        """
        Extracts the Arxiv ID from a given URL.

        Args:
            url (str): The URL containing the Arxiv ID.

        Returns:
            str: The extracted Arxiv ID.

        Raises:
            InvalidArxivURLException: If the URL does not contain an Arxiv ID.

        Example:
            >>> arxiv_client = ArxivClient()
            >>> arxiv_client.parse_arxiv_id("https://huggingface.co/papers/2404.01475")
            '2404.01475'
        """
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        if match:
            return match.group(1)
        raise InvalidArxivURLException(f"Invalid Arxiv URL: {url}. Expected url should contain Arxiv ID.")

    @staticmethod
    def extract_id(url: str) -> str | None:
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        return match.group(1) if match else None

    @staticmethod
    def parse_references(text: str) -> list[str]:
        arxiv_urls = re.findall(r"(https?://arxiv\.org/abs/\d{4}\.\d{4,5}(v\d+)?)", text)
        return [match[0] for match in arxiv_urls]

    def fetch_papers_by_url(self, urls: Iterable[str]) -> list[Paper]:
        id_list = list(filter(None, (self.parse_arxiv_id(url) for url in urls)))

        papers = []
        for paper in self.search_by_url(id_list):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = paper.download_pdf(dirpath=temp_dir)
                doc = cast(fitz.Document, fitz.open(pdf_path))
                try:
                    text = "".join([page.get_text() for page in doc])  # type: ignore
                finally:
                    doc.close()  # Ensure document is closed before temp dir cleanup

            papers.append(
                Paper(
                    title=paper.title,
                    text=text,
                    url=paper.entry_id,
                    authors=[author.name for author in paper.authors],
                    published=paper.published.isoformat() if paper.published else None,
                )
            )  # type: ignore

        return papers

    def fetch_papers_with_references_by_url(self, urls: Iterable[str]) -> list[Paper]:
        parent_papers = self.fetch_papers_by_url(urls)

        for paper in parent_papers:
            reference_urls = self.parse_references(paper.text)
            if reference_urls:
                referenced_papers = self.fetch_papers_by_url(reference_urls)
                paper.references = referenced_papers

        return parent_papers

    def fetch_papers_by_query(self, queries: Iterable[str]) -> list[Paper]:
        papers = []
        for requested_query in queries:
            results = self.search_by_query([requested_query])
            for paper in results:
                if paper.title.lower() != requested_query.lower():
                    continue
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = paper.download_pdf(dirpath=temp_dir)
                    doc = cast(fitz.Document, fitz.open(pdf_path))
                    try:
                        text = "".join([page.get_text() for page in doc])  # type: ignore
                    finally:
                        doc.close()  # Ensure document is closed before temp dir cleanup
                papers.append(
                    Paper(
                        title=paper.title,
                        text=text,
                        url=paper.entry_id,
                        authors=[author.name for author in paper.authors],
                        published=paper.published.isoformat() if paper.published else None,
                    )
                )  # type: ignore

        return papers

    def fetch_papers_with_references_by_query(self, queries: Iterable[str]) -> list[Paper]:
        parent_papers = self.fetch_papers_by_query(queries)

        for paper in parent_papers:
            reference_urls = self.parse_references(paper.text)
            if reference_urls:
                referenced_papers = self.fetch_papers_by_url(reference_urls)
                paper.references = referenced_papers

        return parent_papers

    def download_papers_by_url(self, urls: str | Iterable[str], save_dir: str) -> None:
        id_list = list(filter(None, (self.parse_arxiv_id(url) for url in urls)))

        for paper in self.search_by_url(id_list):
            filename = re.sub(r"\W+", "_", paper.title) + ".pdf"
            paper.download_pdf(dirpath=save_dir, filename=filename)

    def download_papers_by_query(self, queries: str | Iterable[str], save_dir: str) -> None:
        for requested_query in queries:
            results = self.search_by_query([requested_query])
            for paper in results:
                if paper.title.lower() != requested_query.lower():
                    continue
                filename = re.sub(r"\W+", "_", paper.title) + ".pdf"
                paper.download_pdf(dirpath=save_dir, filename=filename)

    def load_paper_as_file_by_url(self, url: str) -> io.BytesIO | None:
        if not isinstance(url, str):
            raise ValueError("Only one URL is allowed.")

        id_ = self.parse_arxiv_id(url)
        if id_ is None:
            return None

        for paper in self.search_by_url([id_]):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = paper.download_pdf(dirpath=temp_dir)
                with open(pdf_path, "rb") as f:
                    return io.BytesIO(f.read())
        return None


arxiv_client = ArxivClient()


def fetch_papers_by_url(urls: str | Iterable[str], parse_reference: bool = False) -> list[Paper]:
    if isinstance(urls, str):
        urls = [urls]

    urls = list(flatten([urls]))

    if parse_reference:
        return arxiv_client.fetch_papers_with_references_by_url(urls)

    return arxiv_client.fetch_papers_by_url(urls)


def fetch_papers_by_query(queries: str | Iterable[str], parse_reference: bool = False) -> list[Paper]:
    if isinstance(queries, str):
        queries = [queries]

    queries = list(flatten([queries]))

    if parse_reference:
        return arxiv_client.fetch_papers_with_references_by_query(queries)

    return arxiv_client.fetch_papers_by_query(queries)


def load_papers_by_url(urls: str | Iterable[str], save_dir: str = "./") -> None:
    if isinstance(urls, str):
        urls = [urls]

    urls = list(flatten([urls]))
    arxiv_client.download_papers_by_url(urls, save_dir)


def load_papers_by_query(queries: str | Iterable[str], save_dir: str = "./") -> None:
    if isinstance(queries, str):
        queries = [queries]

    queries = list(flatten([queries]))
    arxiv_client.download_papers_by_query(queries, save_dir)


def load_paper_as_file_by_url(urls: str) -> io.BytesIO | None:
    return arxiv_client.load_paper_as_file_by_url(urls)


def extract_refs(papers: Iterable[Paper]) -> list[Paper]:
    return list(unique_everseen(flatten([paper.flatten() for paper in papers])))
