import os
import sys
import uuid
import json
import time
import hashlib
import argparse
from typing import Dict, Any, List, Optional
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"
MANIFEST_FILE = os.path.join(PROCESSED_DATA_DIR, "ingestion_manifest.json")


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash of a file for incremental ingestion change detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: Dict[str, Any]) -> None:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(
    file_path: str,
    filename: str,
    source_type: str,
    incremental: bool = False,
    manifest: Optional[Dict[str, Any]] = None,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """Parse → chunk → save locally → embed → index in Qdrant with content hashing."""
    client = client or _get_qdrant_client()
    manifest = manifest if manifest is not None else load_manifest()
    
    file_hash = compute_file_hash(file_path)
    file_key = f"{source_type}/{filename}"

    if incremental and file_key in manifest and manifest[file_key].get("hash") == file_hash:
        logfire.info(f"⏭️ Skipping unchanged file: {filename} (hash match)")
        return {
            "status": "skipped",
            "filename": filename,
            "chunks_count": manifest[file_key].get("chunks_count", 0),
            "source_type": source_type
        }

    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext in ("txt", "md"):
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return {"status": "unsupported", "filename": filename, "error": "Unsupported extension"}

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return {"status": "empty", "filename": filename}

            # 2. Chunk text
            chunks = chunk_text(full_text, chunk_size=1500, chunk_overlap=150)
            if not chunks:
                return {"status": "no_chunks", "filename": filename}

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "file_hash": file_hash,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data snapshot → {local_path}")

            # 4. Embed and index in Qdrant
            target_collection = collection_name or settings.get_collection_name(get_embedding_dim())
            
            # If incremental and file was modified, delete previous points with matching source
            if incremental and file_key in manifest:
                try:
                    client.delete(
                        collection_name=target_collection,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="source",
                                        match=models.MatchValue(value=filename)
                                    )
                                ]
                            )
                        )
                    )
                    logfire.info(f"Purged stale points for updated file: {filename}")
                except Exception as del_err:
                    logfire.warning(f"Could not purge old points: {del_err}")

            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{idx}_{chunk[:30]}")),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "chunk_id": idx,
                            "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        },
                    )
                    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                client.upsert(
                    collection_name=target_collection,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant ({target_collection}) from {filename}.")

            # Update manifest
            manifest[file_key] = {
                "hash": file_hash,
                "chunks_count": len(chunks),
                "indexed_at": time.time(),
                "source_type": source_type
            }

            return {
                "status": "indexed",
                "filename": filename,
                "chunks_count": len(chunks),
                "source_type": source_type
            }

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return {"status": "error", "filename": filename, "error": str(e)}


def process_directory(
    dir_path: str,
    source_type: str,
    incremental: bool = False,
    manifest: Optional[Dict[str, Any]] = None,
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Process every file in a directory."""
    results = []
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            res = process_file(
                os.path.join(dir_path, filename),
                filename,
                source_type,
                incremental=incremental,
                manifest=manifest,
                client=client,
                collection_name=collection_name
            )
            results.append(res)
    return results


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: Optional[str] = None,
    wipe: bool = False,
    incremental: bool = False
) -> Dict[str, Any]:
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass wipe=True to drop and recreate the Qdrant collection before ingestion.
    Returns a comprehensive ingestion report.
    """
    start_time = time.time()
    client = _get_qdrant_client()
    manifest = {} if wipe else load_manifest()

    dim = get_embedding_dim()
    collection_name = settings.get_collection_name(dim)

    with logfire.span("Universal Ingestion Started", base_directory=base_dir, collection=collection_name):
        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if client.collection_exists(collection_name):
                    client.delete_collection(collection_name)
                    logfire.info(f"Collection '{collection_name}' deleted.")
                manifest = {}

        # Recreate collection if missing
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(f"Created collection '{collection_name}' ({dim}-dim, Cosine).")

        all_results: List[Dict[str, Any]] = []

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            res = process_directory(
                base_dir,
                source_type,
                incremental=incremental,
                manifest=manifest,
                client=client,
                collection_name=collection_name
            )
            all_results.extend(res)
        else:
            for subdir in sorted(subdirs):
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                res = process_directory(
                    os.path.join(base_dir, subdir),
                    source_type,
                    incremental=incremental,
                    manifest=manifest,
                    client=client,
                    collection_name=collection_name
                )
                all_results.extend(res)

        save_manifest(manifest)

        duration = time.time() - start_time
        indexed_files = [r for r in all_results if r.get("status") == "indexed"]
        skipped_files = [r for r in all_results if r.get("status") == "skipped"]
        failed_files = [r for r in all_results if r.get("status") == "error"]
        total_chunks = sum(r.get("chunks_count", 0) for r in indexed_files)

        report = {
            "collection_name": collection_name,
            "embedding_dim": dim,
            "files_scanned": len(all_results),
            "files_indexed": len(indexed_files),
            "files_skipped_unchanged": len(skipped_files),
            "files_failed": len(failed_files),
            "total_chunks_indexed": total_chunks,
            "duration_seconds": round(duration, 3),
            "failures": failed_files
        }

        logfire.info(
            f"🚀 Universal Ingestion Complete | indexed={len(indexed_files)}, "
            f"skipped={len(skipped_files)}, chunks={total_chunks}, duration={duration:.2f}s"
        )
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisRAG Universal Document Ingestion Processor")
    parser.add_argument("--dir", default="DATA", help="Path to documents root directory (default: DATA)")
    parser.add_argument("--source-type", default=None, help="Explicit source type tag (e.g. true, noisy)")
    parser.add_argument("--wipe", action="store_true", help="Wipe and recreate Qdrant collection before ingestion")
    parser.add_argument("--incremental", action="store_true", help="Enable SHA-256 incremental ingestion")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: path '{args.dir}' does not exist.")
        sys.exit(1)

    report = run_universal_ingestion(
        base_dir=args.dir,
        explicit_source_type=args.source_type,
        wipe=args.wipe,
        incremental=args.incremental
    )
    print("\n" + "=" * 50)
    print("INGESTION REPORT:")
    print(json.dumps(report, indent=2))
    print("=" * 50)
