# 并发支持检查报告（100+ 图片场景）

> 检查日期：2026-09-08。范围：`/api/v1/listing/batch_generate`、
> `/api/v1/listing/batch_generate_stream`（SSE）在 **100+ 张图片**（如 100 个单图产品，
> 或 20 个产品 × 5 图）下的并发、内存与稳定性，以及新增视频生成链路的并发设计。

## 一、结论摘要

**原有架构对 100+ 图片可用**：并发有信号量约束、超时与重试完整、事件循环无阻塞点。
本次检查发现并修复了 2 个内存/资源风险，补齐 1 个前端超时问题；新增的视频生成
链路按"独立信号量、不占 Listing 槽位"设计，与批量 Listing 并行。

| # | 检查项 | 结论 | 处理 |
|---|--------|------|------|
| 1 | 批量并发上限 | ✅ `BATCH_MAX_CONCURRENCY=8` 信号量约束并发产品数，保护远端 LLM | 无需改动 |
| 2 | 单产品超时 | ✅ `BATCH_PRODUCT_TIMEOUT_S=300`，超时产品标失败不阻塞批次 | 无需改动 |
| 3 | 瞬态错误重试 | ✅ 指数退避（1s/2s/4s 封顶 8s），覆盖连接中断/限流/5xx | 无需改动 |
| 4 | HTTP 连接复用 | ✅ AsyncOpenAI 进程级缓存（`app/utils/openai_client.py`），复用连接池 | 无需改动 |
| 5 | 事件循环保护 | ✅ 图片压缩 `asyncio.to_thread`，RAG 检索 `run_in_threadpool` | 无需改动 |
| 6 | Token 统计并发安全 | ✅ `ContextVar` 按任务隔离（`app/utils/usage.py`），批内产品互不串扰 | 无需改动 |
| 7 | SSE 断连清理 | ✅ 消费端断连时取消全部 producer 任务 | 无需改动 |
| 8 | **历史图片内存驻留** | ⚠️ 原实现把所有产品的**原图字节**留到整批结束（100 产品×5 图×3MB ≈ 1.5GB+） | ✅ **已加固**：校验后立即换成压缩副本（约缩小 10–15 倍），原图仅在该产品处理期间存活 |
| 9 | **批量规模无上限** | ⚠️ 原实现不限制单请求产品数/图片数，恶意或误操作可在 multipart 解析阶段耗尽内存 | ✅ **已加固**：`BATCH_MAX_PRODUCTS=200`、`BATCH_MAX_IMAGES=1000`（可调），超出返回 400 |
| 10 | 前端固定 15 分钟超时 | ⚠️ 大批量 + 视频生成会超过 15 分钟被前端主动中断 | ✅ **已加固**：开启视频时按 `⌈N/3⌉×4+10` 分钟动态放宽（上限 240 分钟） |
| 11 | 视频生成并发 | 🆕 独立 `VIDEO_MAX_CONCURRENCY=3` 信号量；与 Listing 并行、**不占** Listing 槽位；单个视频 900s 预算；断连时取消 | 本次新增 |

## 二、100+ 图片容量估算（默认配置）

- **Listing 吞吐**：8 并发 × 30–60s/产品 → 100 个产品约 **10–15 分钟**完成
  （SSE 逐个回报，无需等整批）。
- **内存**：加固后历史副本约 100–200MB（100 产品×5 图，压缩 JPEG）；原图仅在
  ≤8 个在处理产品中驻留（≤8×5×20MB 上限，实际通常 ≤200MB）。
- **视频**（如开启）：网关实测 5s/720P 约 **70–120s/条**；3 并发 → 100 条视频约
  **40–70 分钟**，与 Listing 并行推进，`video_done` 事件逐条到达，前端即时渲染。

## 三、运维注意事项

1. **保持单 worker**：`uvicorn app.main:app`（默认 1 进程）。历史存储的
   `index.json` 用**进程内**线程锁串行化（`app/history/store.py`），多 worker
   会产生跨进程竞争；如需横向扩容先改造该锁或改用外部存储。
2. **RAG/嵌入是本地 CPU 计算**：sentence-transformers 推理在默认线程池
   （`min(32, cpu+4)` workers）中排队；100+ 产品时与远端调用相比可忽略，
   但若换更大的嵌入模型需关注 CPU。
3. **网关限流**：8 并发是该网关（token-plan）的稳妥值；若出现 429/连接重置风潮，
   调低 `BATCH_MAX_CONCURRENCY`，重试机制会兜底。
4. **视频产物清理**：`data/videos/` 保留最近 `VIDEO_MAX_OUTPUT_FILES=400` 个
   MP4（约 1–2GB），超出自动删最旧；网关原始链接 24 小时过期，本地副本是唯一
   持久来源。
5. **观测**：`GET /api/v1/diag` 现在额外暴露批量上限与视频配置，便于确认运行中
   服务加载了最新配置。

## 四、本次代码改动落点

- `app/config.py`：`batch_max_products` / `batch_max_images` + 全部 `video_*` 配置。
- `app/api/routes.py`：批量上限校验；`_history_copies()`（内存加固）；视频任务
  并行编排（`_start_video_task` / `_await_video` / SSE `video_start|video_done`
  事件）；`GET /api/v1/video/{filename}` 文件服务（严格文件名白名单）。
- `app/video/client.py`：DashScope 异步视频协议（创建任务 → 轮询 → 下载 MP4 →
  本地落盘 + 旧文件清理），mock 模式供离线测试。
- `static/index.html`：动态前端超时；视频开关/提示词框/播放器/汇总计数。
- `tests/test_video.py`：9 个测试覆盖 mock 客户端、单产品/批量/SSE 视频链路、
  文件路由防穿越、批量上限。
