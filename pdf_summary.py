#!/usr/bin/env python3
"""PDF Summary Assistant.

A CLI tool to read text from PDF files and summarize the content,
with an optional agent-like map-reduce workflow.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List



STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "with",
    "at",
    "by",
    "is",
    "it",
    "this",
    "that",
    "as",
    "are",
    "be",
    "from",
    "was",
    "were",
    "can",
    "will",
    "which",
    "into",
    "their",
    "than",
    "about",
}


@dataclass
class SummaryConfig:
    mode: str = "direct"
    backend: str = "heuristic"
    chunk_size: int = 450
    max_bullets: int = 6


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from each page of a PDF."""
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required to read PDF files. Install requirements.txt first.") from exc

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove repeated blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter."""
    # Split on punctuation followed by whitespace and uppercase/number.
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in pieces if s.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]{2,}", text.lower())


def chunk_by_words(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def score_sentences(sentences: Iterable[str]) -> List[tuple[float, str]]:
    """Rank sentences using word frequency scoring."""
    sents = list(sentences)
    token_counts = Counter(
        t for sentence in sents for t in tokenize(sentence) if t not in STOPWORDS
    )

    if not token_counts:
        return [(0.0, s) for s in sents]

    max_freq = max(token_counts.values())
    norm = {token: freq / max_freq for token, freq in token_counts.items()}

    scored = []
    for s in sents:
        toks = [t for t in tokenize(s) if t not in STOPWORDS]
        if not toks:
            scored.append((0.0, s))
            continue
        score = sum(norm.get(t, 0.0) for t in toks) / math.sqrt(len(toks))
        scored.append((score, s))
    return scored


def heuristic_summary(text: str, max_bullets: int = 6) -> str:
    """Extractive summary from top-ranked sentences."""
    sentences = split_sentences(text)
    if not sentences:
        return "No extractable text found in the document."

    ranked = sorted(
        enumerate(score_sentences(sentences)),
        key=lambda x: x[1][0],
        reverse=True,
    )
    picked_idx = sorted(idx for idx, _ in ranked[:max_bullets])
    selected = [sentences[i] for i in picked_idx]

    if not selected:
        return "No meaningful summary could be generated."

    bullets = [f"- {s}" for s in selected]
    return "\n".join(bullets)


def openai_summary(text: str, max_bullets: int = 6) -> str:
    """Abstractive summary via OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is not installed.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai backend.")

    client = OpenAI(api_key=api_key)

    prompt = (
        "Summarize the following document into concise bullet points. "
        f"Return up to {max_bullets} bullets focusing on key findings, decisions, "
        "and action items.\n\n"
        f"DOCUMENT:\n{text[:14000]}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You produce clear and factual document summaries.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def summarize_text(text: str, backend: str, max_bullets: int) -> str:
    if backend == "heuristic":
        return heuristic_summary(text, max_bullets=max_bullets)
    if backend == "openai":
        return openai_summary(text, max_bullets=max_bullets)
    raise ValueError(f"Unsupported backend: {backend}")


def agentic_summary(text: str, config: SummaryConfig) -> str:
    """Map-reduce style summarization agent.

    1) Split text into chunks.
    2) Summarize each chunk.
    3) Summarize the combined chunk summaries.
    """
    chunks = chunk_by_words(text, config.chunk_size)
    if not chunks:
        return "No extractable text found in the document."

    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        partial = summarize_text(
            chunk,
            backend=config.backend,
            max_bullets=max(3, config.max_bullets // 2),
        )
        chunk_summaries.append(f"Chunk {i}/{len(chunks)}\n{partial}")

    combined = "\n\n".join(chunk_summaries)
    final_prompt_text = (
        "Synthesize the partial summaries into a final concise summary:\n\n"
        + combined
    )
    return summarize_text(
        final_prompt_text,
        backend=config.backend,
        max_bullets=config.max_bullets,
    )


def run(pdf_path: Path, config: SummaryConfig) -> str:
    raw_text = extract_pdf_text(pdf_path)
    text = clean_text(raw_text)

    if not text:
        return "No extractable text found in the document."

    if config.mode == "direct":
        return summarize_text(text, config.backend, config.max_bullets)
    if config.mode == "agent":
        return agentic_summary(text, config)
    raise ValueError(f"Unsupported mode: {config.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a PDF document.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to PDF file")
    parser.add_argument(
        "--mode",
        choices=["direct", "agent"],
        default="direct",
        help="Summarization mode",
    )
    parser.add_argument(
        "--backend",
        choices=["heuristic", "openai"],
        default="heuristic",
        help="Summary backend",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="Approximate words per chunk for agent mode",
    )
    parser.add_argument(
        "--max-bullets",
        type=int,
        default=6,
        help="Maximum bullets in final summary",
    )
    parser.add_argument("--output", type=Path, help="Optional output file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SummaryConfig(
        mode=args.mode,
        backend=args.backend,
        chunk_size=args.chunk_size,
        max_bullets=args.max_bullets,
    )

    result = run(args.pdf, config)

    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"Summary written to: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
