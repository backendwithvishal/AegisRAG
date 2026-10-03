import re
from typing import List
import logfire


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly using regex lookarounds."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """
    Splits text by paragraphs (\n\n) while respecting chunk_size and sentence boundaries.
    Adds optional configurable overlap between contiguous chunks.

    Args:
        text: Raw document text to be chunked.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between sequential chunks.

    Returns:
        List of cleaned chunk strings.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text or "")):
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for p in paragraphs:
            # If paragraph itself exceeds chunk_size, split by sentences
            if len(p) > chunk_size:
                sentences = _split_into_sentences(p)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                            # Apply overlap if requested
                            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                                current_chunk = current_chunk[-chunk_overlap:].strip() + " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            chunks.append(sentence[:chunk_size])
                            current_chunk = sentence[chunk_size:].strip()
            else:
                if len(current_chunk) + len(p) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{p}".strip()
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                            overlap_text = current_chunk[-chunk_overlap:].strip()
                            current_chunk = f"{overlap_text}\n\n{p}".strip()
                        else:
                            current_chunk = p
                    else:
                        current_chunk = p

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
        return valid_chunks
