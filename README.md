# RAG-B

RAG-B（API 版）最小实现：
- `Embedding`：向量检索（推荐阿里云 DashScope 兼容接口）
- `Generating`：答案生成（DeepSeek）

当前代码已验证可跑通的组合：
- Embedding: 阿里云 `text-embedding-v4`
- Generation: DeepSeek `deepseek-chat`

## 1. 项目结构

```text
RAG-B/
├─ data/
│  ├─ docs/                  # 你的知识库文档（.txt/.md）
│  └─ index/                 # 索引产物
├─ src/
│  ├─ healthcheck.py         # API 健康检查
│  ├─ ingest.py              # 文档切分 + embedding 建索引
│  └─ rag.py                 # 检索 + 生成问答
└─ requirements.txt
```

## 2. 一次性完整运行指南（可直接复制）

### 2.0 一段式完整 PowerShell（推荐）

最简单方式：直接执行一键脚本。

```powershell
cd D:\0code\RAG-B
.\scripts\run_all.ps1 `
  -DeepseekApiKey "你的_deepseek_key" `
  -EmbeddingApiKey "你的_阿里云_key"
```

脚本会自动执行：
- 创建并激活 `.venv`（可跳过）
- 安装依赖（可跳过）
- `healthcheck -> ingest -> rag`

可选参数示例：

```powershell
.\scripts\run_all.ps1 `
  -DeepseekApiKey "xxx" `
  -EmbeddingApiKey "yyy" `
  -EmbeddingModel "text-embedding-v4" `
  -EmbeddingBatchSize 10 `
  -SkipInstall
```

### 2.1 创建环境并安装依赖

```powershell
cd D:\0code\RAG-B
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2.2 配置 API Key 与模型（当前推荐）

```powershell
# 生成模型（DeepSeek）
$env:DEEPSEEK_API_KEY = "你的_deepseek_key"
$env:GENERATE_BASE_URL = "https://api.deepseek.com/v1"
$env:GENERATE_MODEL = "deepseek-chat"

# 向量模型（阿里云 DashScope 兼容接口）
$env:EMBEDDING_API_KEY = "你的_阿里云_key"
$env:EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:EMBEDDING_MODEL = "text-embedding-v4"

# 阿里云 embedding 批量上限建议 10
$env:EMBEDDING_BATCH_SIZE = "10"
```

### 2.3 （可选）清理错误代理环境变量

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

### 2.4 先做健康检查

```powershell
python src/healthcheck.py
```

预期输出：
- `embedding: PASS - ok`
- `generate : PASS - ok`

### 2.5 准备知识库文档

把你的 `.txt/.md` 文件放到：
- `data/docs/`

项目内已有示例文档，可直接跑。

### 2.6 构建向量索引

```powershell
python src/ingest.py
```

成功后会生成：
- `data/index/embed_matrix.npy`
- `data/index/meta.json`
- `data/index/index_params.json`

### 2.7 启动问答

```powershell
python src/rag.py
```

输入问题示例：

```text
RAG 为什么能减少幻觉？
```

## 3. 关键配置说明

- `EMBEDDING_API_KEY`：必须单独配置，不能用 DeepSeek key 代替
- `DEEPSEEK_API_KEY`：用于生成（也可用 `GENERATE_API_KEY` 显式覆盖）
- `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`：控制向量服务
- `GENERATE_BASE_URL` / `GENERATE_MODEL`：控制生成服务
- `EMBEDDING_BATCH_SIZE`：向量批量请求大小（阿里云建议 `10`）

## 4. 常见问题

### 4.1 `embedding: FAIL - 401 FAILED_TO_AUTH`

说明 embedding 平台鉴权失败，常见原因：
- `EMBEDDING_API_KEY` 错误
- 账户余额不足
- key 没有 embedding 权限
- `EMBEDDING_BASE_URL` 与 key 不属于同一平台

### 4.2 `InvalidParameter ... batch size is invalid`

说明单次 embedding 批量过大。设置：

```powershell
$env:EMBEDDING_BATCH_SIZE = "10"
```

### 4.3 `Embedding index not found. Run: python src/ingest.py`

说明你还没建索引或建索引失败。先执行：

```powershell
python src/ingest.py
```

## 5. 安全建议

你在对话中曾明文暴露过 key，建议立刻在对应平台轮换（重置）key。
