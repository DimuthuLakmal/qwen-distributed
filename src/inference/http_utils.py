"""
Small HTTP-layer helpers shared by the public-facing apps
(`src/inference/transformers/router/routes/inference.py`, `src/inference/vllm/routes/inference.py`).
Kept separate from `src/inference/schema.py`, which is schema definitions
only.
"""

from __future__ import annotations

from fastapi import UploadFile

from src.inference.exception import PayloadTooLargeError

_READ_CHUNK_BYTES = 1024 * 1024


async def read_bounded_upload(upload_file: UploadFile, max_bytes: int) -> bytes:
    """
    Read an upload incrementally and abort as soon as it exceeds `max_bytes`,
    instead of buffering an arbitrarily large upload fully into memory first.
    """
    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = await upload_file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise PayloadTooLargeError(f"image payload exceeds the configured limit of {max_bytes} bytes.")

        chunks.append(chunk)

    return b"".join(chunks)