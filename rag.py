"""Small, in-memory RAG pipeline for PDF and text documents."""

from __future__ import annotations

import math
from typing import Any

from pypdf import PdfReader


def extract_pages(filename: str, content: bytes) -> list[dict[str, Any]]:
    """Extract page-level text and source metadata from a PDF or TXT upload."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(content)
        return [
            {"text": page.extract_text() or "", "source": filename, "page": page_number}
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    if filename.lower().endswith(".txt"):
        return [{"text": content.decode("utf-8-sig"), "source": filename, "page": None}]
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into bounded, overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def embed_texts(client: Any, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Create embeddings in bounded batches, preserving input order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    embeddings = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[start : start + batch_size],
        )
        embeddings.extend(
            item.embedding for item in sorted(response.data, key=lambda item: item.index)
        )
    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def retrieve_chunks(
    question_embedding: list[float],
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Return the highest-similarity chunks, annotated with their scores."""
    if len(chunks) != len(embeddings):
        raise ValueError("There must be one embedding for each chunk")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    ranked = sorted(
        (
            (cosine_similarity(question_embedding, embedding), index)
            for index, embedding in enumerate(embeddings)
        ),
        reverse=True,
    )
    return [
        {**chunks[index], "score": score}
        for score, index in ranked[:top_k]
    ]


def generate_answer(client: Any, question: str, matches: list[dict[str, Any]]) -> str:
    """Generate an answer grounded in retrieved passages."""
    context = "\n\n".join(
        f"[{match['source']}"
        f"{f', page {match[\"page\"]}' if match.get('page') is not None else ''}]\n"
        f"{match['text']}"
        for match in matches
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the supplied document "
                    "passages. If the passages do not contain the answer, say so. "
                    "Cite supporting passages using their bracketed source labels."
                ),
            },
            {
                "role": "user",
                "content": f"Document passages:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty answer")
    return answer
