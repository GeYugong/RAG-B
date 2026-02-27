import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = OUT_DIR / "meta.json"
PARAMS_PATH = OUT_DIR / "index_params.json"
EMBED_MATRIX_PATH = OUT_DIR / "embed_matrix.npy"


def _post_json(url: str, headers: Dict[str, str], payload: Dict, timeout: float):
    session = requests.Session()
    session.trust_env = False
    return session.post(url, headers=headers, json=payload, timeout=timeout)


def read_all_docs(doc_dir: Path) -> List[Dict]:
    docs = []
    for p in sorted(doc_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"path": str(p.relative_to(ROOT)), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def embed_texts_via_api(texts: List[str], batch_size: int = 10) -> np.ndarray:
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY.")

    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.ppio.com/openai/v1")
    model = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

    if "api.deepseek.com" in base_url:
        raise RuntimeError(
            "DeepSeek 当前不提供 embeddings API。请设置 EMBEDDING_BASE_URL/EMBEDDING_MODEL 到支持 embedding 的 OpenAI 兼容服务。"
        )

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _post_json(url, headers, {"model": model, "input": batch}, timeout)
        if resp.status_code >= 300:
            raise RuntimeError(f"Embedding API failed: {resp.status_code} {resp.text}")
        payload = resp.json()
        data = sorted(payload.get("data", []), key=lambda x: x.get("index", 0))
        if len(data) != len(batch):
            raise RuntimeError("Embedding API returned unexpected vector count.")
        vectors.extend([item["embedding"] for item in data])

    arr = np.asarray(vectors, dtype=np.float32)
    return _normalize_rows(arr)


def main():
    docs = read_all_docs(DOC_DIR)
    if not docs:
        raise RuntimeError(f"No docs found in {DOC_DIR}. Put some .txt/.md files there.")

    chunks = []
    meta = []
    for d in docs:
        cs = chunk_text(d["text"])
        for i, c in enumerate(cs):
            chunks.append(c)
            meta.append({"source": d["path"], "chunk_id": i, "text": c})

    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    embed_matrix = embed_texts_via_api(chunks, batch_size=batch_size)
    np.save(str(EMBED_MATRIX_PATH), embed_matrix)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    PARAMS_PATH.write_text(
        json.dumps(
            {
                "kind": "embedding_api",
                "rows": int(embed_matrix.shape[0]),
                "cols": int(embed_matrix.shape[1]),
                "embed_matrix_path": str(EMBED_MATRIX_PATH.relative_to(ROOT)),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "baai/bge-m3"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Ingest done.")
    print(f"Embedding matrix: {EMBED_MATRIX_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"Chunks: {len(meta)}")


if __name__ == "__main__":
    main()

