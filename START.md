# CrossLister 启动指南

## 系统要求

- Python 3.11+
- 内存: 最低 8GB（本地嵌入模型需要约 2GB）
- 磁盘: 至少 5GB（模型文件 + 依赖）

## 快速启动

### 1. 创建环境

#### 方式一：使用 Conda（推荐）

```bash
# 创建并激活 conda 环境
conda create -n crosslister python=3.11 -y
conda activate crosslister
cd CrossLister

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用 uv

```bash
# 使用 uv 创建虚拟环境并安装依赖
uv sync
```

#### 方式三：使用 venv

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下关键项：

```ini
# 应用配置
DEBUG=true

# Vision 模型（图片分析）
VISION_MODE=api
VISION_API_BASE=https://your-api-endpoint/v1
VISION_API_KEY=your-api-key
VISION_MODEL=qwen3.6-flash

# LLM 模型（文本生成）
LLM_MODE=api
LLM_API_BASE=https://your-api-endpoint/v1
LLM_API_KEY=your-api-key
LLM_MODEL=qwen3.6-flash

# 嵌入模型（本地）
EMBEDDING_MODE=local
EMBEDDING_LOCAL_MODEL_PATH=/path/to/your/embedding/model
```

### 3. 启动服务

```bash
# 使用 conda 环境启动（推荐）
/home/huajuanx/miniconda3/envs/crosslister/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 或者先激活 conda 环境再启动
conda activate crosslister
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 开发模式（支持热重载）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 4. 验证服务

```bash
curl http://localhost:8080/api/v1/health
```

应返回：
```json
{
  "status": "ok",
  "version": "0.1.0",
  "vision_mode": "api",
  "llm_mode": "api",
  "embedding_mode": "local"
}
```

## 运行测试

```bash
# 单元测试（Mock 模式）
pytest tests/ -v

# 集成测试（真实 API 调用）
python scripts/integration_test.py
```

## Docker 部署（可选）

```bash
docker-compose up -d
```

## 注意事项

1. **代理设置**: 如果使用代理，确保本地请求绕过代理：
   ```bash
   export NO_PROXY="*"
   export no_proxy="*"
   ```

2. **嵌入模型路径**: `EMBEDDING_LOCAL_MODEL_PATH` 应指向模型快照目录，而非 HuggingFace 缓存根目录。

3. **模型名称**: 确保 `.env` 中的模型名称与 API 提供商支持的名称一致。
