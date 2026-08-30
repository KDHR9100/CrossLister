# CrossLister 优化任务清单

> 2026-08-30 制定。按 P1–P4 优先级逐批交付，每阶段完成后全量回归并提交一次 commit。
> 完成一项勾选一项，并在"备注"列记录关键决策或偏差。

## P1 健壮性修复（bug 级）

- [x] **T1 模型客户端模块级单例**（`app/llm/client.py`、`app/vision/client.py`）
  每次节点调用都重建 AsyncOpenAI + httpx.AsyncClient（重复 TLS 握手、旧连接无人 close）。
  改为按 base_url/key/timeout 缓存的共享工厂，提供测试用 reset 钩子。
- [x] **T2 上传大小先校验后读取**（`app/api/routes.py`）
  `_read_and_validate` 与 `parse_import_file` 先用 `UploadFile.size` 判 20MB 上限再读内容，
  超限直接 400，避免超大文件整体进内存。
- [x] **T3 图片压缩移出事件循环**（`app/vision/client.py`）
  `_remote_analyze` 中 `preprocess_image` 批量调用包进线程池，避免大图 resize 阻塞事件循环。
- [x] **T4 合规复核解析失败兜底**（`app/guardrails/llm_checker.py`、`compliance_node.py`）
  解析失败重试一次 LLM 调用；仍失败 fail-open，但把"语义复核已跳过"写入
  `ComplianceResult.warnings`，不再静默。

## P2 合规校验强化（核心价值）

- [x] **T5 结构性合规校验器**（新建 `app/guardrails/structural_validator.py`）
  违规级（触发重写回环）：标题长度（amazon≤200 / shopee≤120 / temu 20–120）、
  amazon bullet ≤5 条且每条 ≤500 字符、amazon 描述 ≤2000 字符、amazon 关键词 ≤249 字节、
  shopee/temu 标题 emoji、全平台 URL/外链、temu 价格字样。
  警告级（不触发重写）：ALL-CAPS 词（含 acronym 白名单）、amazon 标题 `!*$?`、
  amazon bullet >250 字符建议。违规消息带 rule_id 前缀，经现有 feedback 机制回给生成节点。
- [x] **T6 LLM JSON 结构化输出**（`app/llm/client.py`）
  `chat()` 增加 `response_format` 参数，generate/checker/translate 启用 `json_object`，
  保留 `extract_json_object` 容错回退。
- [x] **T7 max_tokens 上限**（`app/config.py`、`app/llm/client.py`）
  新增 `llm_max_output_tokens`（默认 2048），`chat()` 传入。

## P3 体验

- [x] **T8 SSE 真进度**
  后端：`graph.py` 基于 `astream(stream_mode="updates")` 的流式入口；
  `routes.py` 新增 `POST /api/v1/listing/batch_generate_stream`（product_start / node /
  product_done / done 事件，复用并发限流与单品超时），原端点保留兼容。
  前端：`handleGenerate` 改 fetch 流式读取，替换假流水线动画，结果逐个渲染。
- [ ] **T9 大批量任务化**：暂缓。T8 落地后已有真进度与错误隔离，job 队列复杂度暂不值。
- [x] **T10 启动索引检查**（`app/main.py` lifespan）
  逐平台检查 chroma collection 存在且非空，缺失记 warning；
  新增 `rag_autobuild_on_startup`（默认 false），为 true 时后台自动重建。

## P4 RAG / 历史 / 部署

- [ ] **T11 RAG 分数阈值**（`app/rag/retriever.py`）
  新增 `rag_min_score`（默认 0.0），`retrieve()` 过滤低分规则。
- [ ] **T12 历史系统增强**（`app/history/store.py`、`app/api/history.py`、`static/index.html`）
  新增 `DELETE /api/v1/history/{record_id}` + 前端删除按钮；
  index.json 损坏时从 record 目录扫描重建；列表支持按平台/状态过滤。
- [ ] **T13 可选 API Key 鉴权**（`app/main.py`、`app/config.py`）
  新增 `auth_api_key`（默认空 = 不启用）；启用后 `/api/*`（除 health）要求 `X-API-Key`；
  前端 401 提示并支持 localStorage 保存 key。
- [ ] **T14 Dockerfile**：CMD 改 `uv run --no-sync`，避免容器启动重复 sync。
- [ ] **T15 工程清理**
  重试逻辑抽公共 async helper（`utils/retry.py`）；`ListingMetadata.vision_model` 字段；
  vision docstring "1-5"→"1–20"；移除 `requirements.txt`（以 uv.lock 为准）并同步 README。

## 交付记录

| 日期 | 阶段 | commit | 备注 |
| --- | --- | --- | --- |
| 2026-08-30 | P1 健壮性修复 | (见 git log) | 新增 app/utils/openai_client.py 共享客户端工厂；测试数 72→86 |
| 2026-08-30 | P2 合规强化 | (见 git log) | structural_validator 落地；违规/警告分级防误报空转；测试数 86→109 |
| 2026-08-30 | P3 体验 | (见 git log) | SSE 端点 batch_generate_stream；前端真进度+占位渲染；启动索引检查；测试数 109→112 |
