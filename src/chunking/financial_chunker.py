"""Structure-aware chunking for financial documents.

Splits filing sections into chunks that respect paragraph boundaries,
section headers, and financial table boundaries to preserve context.
Uses tiktoken (cl100k_base) for accurate token counting.
"""

import re
import uuid

import tiktoken
from bs4 import BeautifulSoup

from src.chunking.models import ChunkMetadata, DocumentChunk
from src.chunking.table_handler import html_table_to_markdown
from src.ingestion.models import FilingSection

CHARS_PER_PAGE = 3000


class FinancialChunker:
    """Chunker that respects the structure of SEC filings.

    Splits by section boundaries first, then by paragraphs within sections.
    Tables are always kept as atomic chunks. Overlap never crosses section
    boundaries.

    Args:
        chunk_size: Target token count per chunk (default 768).
        chunk_overlap: Number of overlapping tokens between chunks (default 128).
    """

    def __init__(self, chunk_size: int = 768, chunk_overlap: int = 128) -> None:
        """Initialize the chunker with size parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self._encoding.encode(text))

    def chunk_filing(self, sections: list[FilingSection]) -> list[DocumentChunk]:
        """Chunk all sections of a filing.

        Processes each section independently so that overlap never crosses
        section boundaries.

        Args:
            sections: All parsed sections from a single filing.

        Returns:
            Complete list of DocumentChunk objects for the filing.
        """
        all_chunks: list[DocumentChunk] = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section))
        return all_chunks

    def chunk_section(self, section: FilingSection) -> list[DocumentChunk]:
        """Split a filing section into document chunks.

        Args:
            section: A parsed filing section.

        Returns:
            List of DocumentChunk objects preserving structural boundaries.
        """
        ticker = section.metadata.ticker
        year = section.metadata.filing_date.year
        filing_type = section.metadata.filing_type
        section_name = section.section_name

        prefix = self._build_metadata_prefix(ticker, year, section_name, filing_type)
        prefix_tokens = self.count_tokens(prefix + "\n\n")
        effective_size = max(self.chunk_size - prefix_tokens, 1)

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        char_offset = 0

        # --- text chunks ---
        text_content = self._strip_html_tables(section.content)
        paragraphs = self._split_into_paragraphs(text_content)

        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens > effective_size:
                # Flush accumulated text first
                if current_parts:
                    chunk, char_offset = self._flush_text(
                        current_parts, prefix, ticker, year, filing_type,
                        section_name, chunk_index, char_offset,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_parts = []
                    current_tokens = 0

                # Split oversized paragraph by sentences
                sent_chunks, char_offset = self._split_large_paragraph(
                    para, effective_size, prefix, ticker, year, filing_type,
                    section_name, chunk_index, char_offset,
                )
                chunks.extend(sent_chunks)
                chunk_index += len(sent_chunks)

            elif current_tokens + para_tokens > effective_size:
                # Flush current chunk
                chunk, char_offset = self._flush_text(
                    current_parts, prefix, ticker, year, filing_type,
                    section_name, chunk_index, char_offset,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start next chunk with overlap from previous
                overlap_parts = self._get_overlap_paragraphs(current_parts)
                overlap_tokens = sum(self.count_tokens(p) for p in overlap_parts)
                current_parts = overlap_parts + [para]
                current_tokens = overlap_tokens + para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunk, char_offset = self._flush_text(
                current_parts, prefix, ticker, year, filing_type,
                section_name, chunk_index, char_offset,
            )
            chunks.append(chunk)
            chunk_index += 1

        # --- table chunks (always atomic, never split) ---
        for table_raw in section.tables:
            if table_raw.strip().startswith("<"):
                table_md = html_table_to_markdown(table_raw)
            else:
                table_md = table_raw

            page = self._estimate_page(char_offset)
            chunks.append(self._make_chunk(
                table_md, prefix, ticker, year, filing_type,
                section_name, chunk_index, True, page,
            ))
            chunk_index += 1
            char_offset += len(table_md)

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata_prefix(
        ticker: str, year: int, section_name: str, filing_type: str,
    ) -> str:
        return (
            f"[Ticker: {ticker}] [Year: {year}] "
            f"[Section: {section_name}] [Filing: {filing_type}]"
        )

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _estimate_page(char_offset: int) -> int:
        return (char_offset // CHARS_PER_PAGE) + 1

    @staticmethod
    def _strip_html_tables(content: str) -> str:
        if "<table" not in content.lower():
            return content
        soup = BeautifulSoup(content, "html.parser")
        for table in soup.find_all("table"):
            table.decompose()
        return soup.get_text()

    def _make_chunk(
        self,
        body: str,
        prefix: str,
        ticker: str,
        year: int,
        filing_type: str,
        section_name: str,
        chunk_index: int,
        is_table: bool,
        page_estimate: int,
    ) -> DocumentChunk:
        full_content = f"{prefix}\n\n{body}"
        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            content=full_content,
            metadata=ChunkMetadata(
                ticker=ticker,
                year=year,
                filing_type=filing_type,
                section_name=section_name,
                chunk_index=chunk_index,
                is_table=is_table,
                page_estimate=page_estimate,
            ),
            metadata_prefix=prefix,
            token_count=self.count_tokens(full_content),
        )

    def _flush_text(
        self,
        parts: list[str],
        prefix: str,
        ticker: str,
        year: int,
        filing_type: str,
        section_name: str,
        chunk_index: int,
        char_offset: int,
    ) -> tuple[DocumentChunk, int]:
        body = "\n\n".join(parts)
        page = self._estimate_page(char_offset)
        chunk = self._make_chunk(
            body, prefix, ticker, year, filing_type,
            section_name, chunk_index, False, page,
        )
        return chunk, char_offset + len(body)

    def _get_overlap_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Return trailing paragraphs that fit within the overlap budget."""
        if not paragraphs:
            return []
        result: list[str] = []
        tokens = 0
        for para in reversed(paragraphs):
            t = self.count_tokens(para)
            if tokens + t > self.chunk_overlap:
                break
            result.insert(0, para)
            tokens += t
        return result

    def _split_large_paragraph(
        self,
        paragraph: str,
        effective_size: int,
        prefix: str,
        ticker: str,
        year: int,
        filing_type: str,
        section_name: str,
        start_index: int,
        char_offset: int,
    ) -> tuple[list[DocumentChunk], int]:
        """Split an oversized paragraph at sentence boundaries."""
        sentences = self._split_into_sentences(paragraph)
        chunks: list[DocumentChunk] = []
        idx = start_index

        sent_parts: list[str] = []
        sent_tokens = 0

        for sent in sentences:
            st = self.count_tokens(sent)
            if sent_tokens + st > effective_size and sent_parts:
                body = " ".join(sent_parts)
                page = self._estimate_page(char_offset)
                chunks.append(self._make_chunk(
                    body, prefix, ticker, year, filing_type,
                    section_name, idx, False, page,
                ))
                idx += 1
                char_offset += len(body)

                # Sentence-level overlap
                overlap_sents: list[str] = []
                overlap_tok = 0
                for s in reversed(sent_parts):
                    s_tok = self.count_tokens(s)
                    if overlap_tok + s_tok > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok

                sent_parts = overlap_sents + [sent]
                sent_tokens = overlap_tok + st
            else:
                sent_parts.append(sent)
                sent_tokens += st

        if sent_parts:
            body = " ".join(sent_parts)
            page = self._estimate_page(char_offset)
            chunks.append(self._make_chunk(
                body, prefix, ticker, year, filing_type,
                section_name, idx, False, page,
            ))
            char_offset += len(body)

        return chunks, char_offset
