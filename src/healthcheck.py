import os

import requests


def _post_json(url: str, headers: dict, payload: dict, timeout: float):
    session = requests.Session()
    session.trust_env = False
    return session.post(url, headers=headers, json=payload, timeout=timeout)


def check_embedding():
    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.ppio.com/openai/v1")
    model = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
    if not api_key:
        return False, "EMBEDDING_API_KEY 未设置"
    if "api.deepseek.com" in base_url:
        return False, "DeepSeek 不提供 embeddings API，请改为其他 OpenAI 兼容 embedding 服务"

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = _post_json(url, headers, {"model": model, "input": ["ping"]}, 30)
    if resp.status_code >= 300:
        return False, f"{resp.status_code} {resp.text}"
    return True, "ok"


def check_generation():
    api_key = os.getenv("GENERATE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("GENERATE_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("GENERATE_MODEL", "deepseek-chat")
    if not api_key:
        return False, "GENERATE_API_KEY/DEEPSEEK_API_KEY 未设置"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "回复 ok"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    resp = _post_json(url, headers, payload, 30)
    if resp.status_code >= 300:
        return False, f"{resp.status_code} {resp.text}"
    return True, "ok"


def main():
    emb_ok, emb_msg = check_embedding()
    gen_ok, gen_msg = check_generation()

    print(f"embedding: {'PASS' if emb_ok else 'FAIL'} - {emb_msg}")
    print(f"generate : {'PASS' if gen_ok else 'FAIL'} - {gen_msg}")

    if not (emb_ok and gen_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

