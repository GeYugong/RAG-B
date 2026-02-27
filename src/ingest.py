import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = OUT_DIR / "tfidf_matrix.npz"
VOCAB_PATH = OUT_DIR / "tfidf_vocab.json"
IDF_PATH = OUT_DIR / "tfidf_idf.npy"
META_PATH = OUT_DIR / "meta.json"
PARAMS_PATH = OUT_DIR / "index_params.json"
VECTORIZER_PATH = OUT_DIR / "tfidf_vectorizer.pkl"
EMBED_MATRIX_PATH = OUT_DIR / "embed_matrix.npy"


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


def embed_texts_via_api(texts: List[str], batch_size: int = 64) -> np.ndarray:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY (or DEEPSEEK_API_KEY).")

    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("EMBEDDING_MODEL", "deepseek-embedding")
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            url,
            headers=headers,
            json={"model": model, "input": batch},
            timeout=timeout,
        )
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

    use_embedding_api = os.getenv("USE_EMBEDDING_API", "1") == "1"
    index_kind = "embedding_api" if use_embedding_api else "tfidf"

    rows = len(meta)
    cols = 0
    vectorizer_rel = None

    if use_embedding_api:
        embed_matrix = embed_texts_via_api(chunks)
        np.save(str(EMBED_MATRIX_PATH), embed_matrix)
        rows, cols = int(embed_matrix.shape[0]), int(embed_matrix.shape[1])
    else:
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)
        matrix = vectorizer.fit_transform(chunks)

        from scipy.sparse import save_npz

        save_npz(str(MATRIX_PATH), matrix)
        vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        np.save(str(IDF_PATH), vectorizer.idf_.astype("float32"))
        with VECTORIZER_PATH.open("wb") as f:
            pickle.dump(vectorizer, f)

        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        vectorizer_rel = str(VECTORIZER_PATH.relative_to(ROOT))

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    PARAMS_PATH.write_text(
        json.dumps(
            {
                "kind": index_kind,
                "rows": rows,
                "cols": cols,
                "vectorizer_path": vectorizer_rel,
                "embed_matrix_path": (
                    str(EMBED_MATRIX_PATH.relative_to(ROOT)) if use_embedding_api else None
                ),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "deepseek-embedding")
                if use_embedding_api
                else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Ingest done.")
    if use_embedding_api:
        print(f"Embedding matrix: {EMBED_MATRIX_PATH}")
    else:
        print(f"Matrix: {MATRIX_PATH}")
    print(f"Meta: {META_PATH}")
    if not use_embedding_api:
        print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Chunks: {len(meta)}")


if __name__ == "__main__":
    main()
