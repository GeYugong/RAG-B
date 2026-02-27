import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "data" / "index" / "meta.json"
PARAMS_PATH = ROOT / "data" / "index" / "index_params.json"
EMBED_MATRIX_PATH = ROOT / "data" / "index" / "embed_matrix.npy"


def _post_json(url: str, headers: Dict[str, str], payload: Dict, timeout: float):
    session = requests.Session()
    session.trust_env = False
    return session.post(url, headers=headers, json=payload, timeout=timeout)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def load_index_and_meta():
    if not META_PATH.exists() or not PARAMS_PATH.exists() or not EMBED_MATRIX_PATH.exists():
        raise RuntimeError("Embedding index not found. Run: python src/ingest.py")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    if params.get("kind") != "embedding_api":
        raise RuntimeError("Current index is not embedding_api. Rebuild: python src/ingest.py")

    matrix = np.load(str(EMBED_MATRIX_PATH))
    return matrix, meta


def embed_query(query: str) -> np.ndarray:
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
    resp = _post_json(url, headers, {"model": model, "input": [query]}, timeout)
    if resp.status_code >= 300:
        raise RuntimeError(f"Embedding API failed: {resp.status_code} {resp.text}")

    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("Embedding API returned empty result.")

    vec = np.asarray(data[0]["embedding"], dtype=np.float32)
    return _normalize(vec)


def retrieve(query: str, k: int = 4, min_score: float = 1e-9) -> List[Dict]:
    matrix, meta = load_index_and_meta()
    q = embed_query(query)
    scores = matrix @ q

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
        raise RuntimeError("Missing GENERATE_API_KEY (or DEEPSEEK_API_KEY).")

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
    resp = _post_json(
        url,
        headers,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(os.getenv("GENERATE_TEMPERATURE", "0.2")),
        },
        timeout,
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

