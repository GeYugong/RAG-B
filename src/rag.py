import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "data" / "index" / "tfidf_matrix.npz"
META_PATH = ROOT / "data" / "index" / "meta.json"
PARAMS_PATH = ROOT / "data" / "index" / "index_params.json"
VECTORIZER_PATH = ROOT / "data" / "index" / "tfidf_vectorizer.pkl"
EMBED_MATRIX_PATH = ROOT / "data" / "index" / "embed_matrix.npy"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def load_index_and_meta():
    if not META_PATH.exists() or not PARAMS_PATH.exists():
        raise RuntimeError("Index not found or incomplete. Run: python src/ingest.py")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    kind = params.get("kind", "tfidf")

    if kind == "embedding_api":
        if not EMBED_MATRIX_PATH.exists():
            raise RuntimeError("Embedding index missing. Run: python src/ingest.py")
        matrix = np.load(str(EMBED_MATRIX_PATH))
        return kind, matrix, None, meta

    if not MATRIX_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError("TF-IDF index missing. Run: python src/ingest.py")

    matrix = load_npz(str(MATRIX_PATH)).tocsr()
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return "tfidf", matrix, vectorizer, meta


def embed_query(query: str) -> np.ndarray:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY (or DEEPSEEK_API_KEY).")

    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("EMBEDDING_MODEL", "deepseek-embedding")
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        url,
        headers=headers,
        json={"model": model, "input": [query]},
        timeout=timeout,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Embedding API failed: {resp.status_code} {resp.text}")

    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("Embedding API returned empty result.")

    vec = np.asarray(data[0]["embedding"], dtype=np.float32)
    return _normalize(vec)


def retrieve(query: str, k: int = 4, min_score: float = 1e-9) -> List[Dict]:
    kind, matrix, vectorizer, meta = load_index_and_meta()

    if kind == "embedding_api":
        q = embed_query(query)
        scores = matrix @ q
    else:
        q = vectorizer.transform([query])
        scores = (matrix @ q.T).toarray().ravel()

    positive_idx = np.where(scores > min_score)[0]
    if positive_idx.size == 0:
        return []

    k = max(1, min(k, positive_idx.size))
    candidate_scores = scores[positive_idx]
    top_local = np.argpartition(-candidate_scores, k - 1)[:k]
    top_idx = positive_idx[top_local]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        m = meta[int(idx)]
        results.append({"score": float(scores[int(idx)]), **m})
    return results


def generate_answer(query: str, contexts: List[Dict]) -> str:
    ctx_text = "\n\n".join(
        [
            f"[{i+1}] {c['source']}#chunk{c['chunk_id']} (score={c['score']:.3f})\n{c['text']}"
            for i, c in enumerate(contexts)
        ]
    )

    api_key = os.getenv("GENERATE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return (
            "未配置 GENERATE_API_KEY/DEEPSEEK_API_KEY，返回检索结果：\n\n"
            f"问题：{query}\n\n上下文：\n{ctx_text}"
        )

    base_url = os.getenv("GENERATE_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("GENERATE_MODEL", "deepseek-chat")
    timeout = float(os.getenv("GENERATE_TIMEOUT", "120"))

    system_prompt = (
        "你是一个RAG问答助手。请严格基于给定上下文回答。"
        "如果上下文不足以回答，明确说不知道并说明缺失信息。"
    )
    user_prompt = (
        f"问题：{query}\n\n"
        "以下是检索到的上下文，请优先引用其中信息回答：\n"
        f"{ctx_text}\n\n"
        "请用简洁中文回答。"
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(os.getenv("GENERATE_TEMPERATURE", "0.2")),
        },
        timeout=timeout,
    )

    if resp.status_code >= 300:
        raise RuntimeError(f"Generate API failed: {resp.status_code} {resp.text}")

    payload = resp.json()
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("Generate API returned empty choices.")

    return choices[0]["message"]["content"].strip()


def main():
    query = input("请输入问题：").strip()
    if not query:
        print("空问题，退出。")
        return

    contexts = retrieve(query, k=4)
    if not contexts:
        print("没有检索到内容。")
        return

    answer = generate_answer(query, contexts)

    print("\n" + "=" * 80)
    print(answer)
    print("\n--- 检索片段 ---")
    for i, c in enumerate(contexts, start=1):
        print(f"[{i}] {c['source']}#chunk{c['chunk_id']} score={c['score']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
