# RAG-B

一个可运行的 RAG-B 示例项目：
- 检索阶段支持 embedding API（默认）或 TF-IDF（兜底）
- 生成阶段接入 LLM API（默认 DeepSeek）

## 1. 改造目标

- 从 RAG-A 的本地 TF-IDF + 拼接回答
- 升级为 RAG-B 的 embedding 检索 + LLM 生成

## 2. 环境要求

- Python 3.8+
- Windows PowerShell（其他终端也可）

## 3. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 4. 配置 API（默认低成本 DeepSeek）

先配置统一 key：

```powershell
$env:DEEPSEEK_API_KEY = "你的key"
```

默认配置：
- embedding: `https://api.deepseek.com/v1` + `deepseek-embedding`
- generation: `https://api.deepseek.com/v1` + `deepseek-chat`

如需覆盖，可设置：

```powershell
$env:EMBEDDING_API_KEY = "你的embedding key"
$env:EMBEDDING_BASE_URL = "https://api.deepseek.com/v1"
$env:EMBEDDING_MODEL = "deepseek-embedding"

$env:GENERATE_API_KEY = "你的generation key"
$env:GENERATE_BASE_URL = "https://api.deepseek.com/v1"
$env:GENERATE_MODEL = "deepseek-chat"
```

## 5. 构建索引（Ingest）

默认走 embedding API：

```powershell
python src/ingest.py
```

生成文件：
- `data/index/embed_matrix.npy`
- `data/index/meta.json`
- `data/index/index_params.json`

如需回退本地 TF-IDF：

```powershell
$env:USE_EMBEDDING_API = "0"
python src/ingest.py
```

## 6. 运行问答（Retrieve + Generate）

```powershell
python src/rag.py
```

程序会：
- 先召回 top-k 片段
- 再调用生成模型产出最终答案
- 同时打印召回证据片段

## 7. 低成本建议

- 默认模型已设为 `deepseek-chat`（生成）+ `deepseek-embedding`（召回）
- 如果后续你要切换供应商，只需替换环境变量，不用改代码

## 8. 项目结构

```text
RAG-B/
├─ data/
│  ├─ docs/
│  └─ index/
├─ src/
│  ├─ ingest.py
│  └─ rag.py
└─ requirements.txt
```
