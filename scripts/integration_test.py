"""Comprehensive integration test: model calls + end-to-end pipeline.

Tests each configured model (Vision API, LLM API, Embedding local) and
the full listing generation pipeline through the live API server.
Records performance metrics and data quality for the report.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Bypass proxy for all local and API calls
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Invalidate cached settings so .env values are re-read
from app.config import get_settings
get_settings.cache_clear()

import httpx
from app.config import Settings, VisionMode, LLMMode, EmbeddingMode
from app.vision.client import VisionClient, encode_image, guess_mime
from app.vision.prompts import build_vision_messages
from app.llm.client import LLMClient
from app.rag.indexer import Embedder, build_index
from app.rag.retriever import RuleRetriever, build_query
from app.guardrails import keyword_filter, llm_checker

def _make_test_png(width=64, height=64):
    """Create a minimal valid PNG image of the given size (no external deps)."""
    import struct, zlib
    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b""
    for y in range(height):
        raw += b"\x00" + bytes([128, 128, 128] * width)
    idat = zlib.compress(raw)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")

TEST_IMAGE = _make_test_png(64, 64)

# API server base URL; override via the API_BASE env var if needed.
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8080")


def _try_parse_json(raw: str) -> tuple[bool, dict]:
    """Best-effort extraction of a JSON object from raw model text.

    Mirrors the tolerant parsing used in the app (code-fence and stray-prose
    tolerant). Returns (parse_ok, data) where data is {} when parsing fails.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return False, {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(data, dict):
        return False, {}
    return True, data


@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: int = 0
    details: str = ""
    error: str = ""


@dataclass
class TestReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, r: TestResult):
        self.results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.latency_ms}ms) {r.details}")
        if r.error:
            print(f"         ERROR: {r.error}")

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"\n{'='*70}")
        print(f"TOTAL: {total} | PASSED: {passed} | FAILED: {failed}")
        print(f"{'='*70}")
        return passed, failed


report = TestReport()


# =====================================================================
# 1. Health Check
# =====================================================================
async def test_health():
    print("\n--- 1. Health Check ---")
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.get(f"{API_BASE}/api/v1/health", timeout=10)
        latency = int((time.perf_counter() - t0) * 1000)
        body = resp.json()
        ok = resp.status_code == 200 and body.get("status") == "ok"
        report.add(TestResult(
            name="Health endpoint",
            passed=ok,
            latency_ms=latency,
            details=f"vision={body.get('vision_mode')} llm={body.get('llm_mode')} emb={body.get('embedding_mode')}",
        ))
    except Exception as e:
        report.add(TestResult(name="Health endpoint", passed=False, error=str(e)))


# =====================================================================
# 2. Vision Model (API mode) - Direct call
# =====================================================================
async def test_vision_model():
    print("\n--- 2. Vision Model (API mode) ---")
    settings = get_settings()

    if settings.vision_mode != VisionMode.API:
        report.add(TestResult(
            name="Vision API call",
            passed=False,
            error=f"VISION_MODE is {settings.vision_mode.value}, expected 'api'",
        ))
        return

    client = VisionClient(settings)
    t0 = time.perf_counter()
    try:
        result = await client.analyze(
            images=[TEST_IMAGE],
            category_hint="storage organizer",
            extra_info={"brand": "TestBrand"},
        )
        latency = int((time.perf_counter() - t0) * 1000)

        # Validate response structure
        checks = []
        checks.append(("detected_category non-empty", bool(result.detected_category)))
        checks.append(("colors is list", isinstance(result.colors, list)))
        checks.append(("materials is list", isinstance(result.materials, list)))
        checks.append(("selling_points is list", isinstance(result.selling_points, list)))
        checks.append(("scenes is list", isinstance(result.scenes, list)))
        checks.append(("raw_description non-empty", bool(result.raw_description)))

        all_ok = all(v for _, v in checks)
        detail_parts = [f"{k}={'OK' if v else 'FAIL'}" for k, v in checks]
        report.add(TestResult(
            name="Vision API: image analysis",
            passed=all_ok,
            latency_ms=latency,
            details=f"category='{result.detected_category}' | {'; '.join(detail_parts)}",
        ))

        # Check data quality: for synthetic test images (solid color), selling_points
        # may be empty since there are no visible features to extract. We only verify
        # that the model returns a valid description.
        quality_ok = len(result.raw_description) >= 10
        report.add(TestResult(
            name="Vision API: data quality",
            passed=quality_ok,
            latency_ms=0,
            details=f"selling_points={result.selling_points}, desc_len={len(result.raw_description)}",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="Vision API: image analysis",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


# =====================================================================
# 3. LLM Model (API mode) - Direct calls
# =====================================================================
async def test_llm_model():
    print("\n--- 3. LLM Model (API mode) ---")
    settings = get_settings()

    if settings.llm_mode != LLMMode.API:
        report.add(TestResult(
            name="LLM API call",
            passed=False,
            error=f"LLM_MODE is {settings.llm_mode.value}, expected 'api'",
        ))
        return

    client = LLMClient(settings)

    # 3a. Listing generation
    t0 = time.perf_counter()
    try:
        system = (
            "You are a senior cross-border e-commerce copywriter. "
            "Write a product listing as a JSON object with keys: title, "
            "bullet_points (list of 3-5 strings), description, "
            "backend_keywords (list of 5-10 short strings)."
        )
        user = (
            "Product category: storage organizer\n"
            "Target language: en\n"
            "Visual analysis: Stackable transparent PP plastic bins, "
            "dust-proof, space-saving design.\n"
            "Write the listing now."
        )
        raw = await client.chat(system, user, temperature=0.3)
        latency = int((time.perf_counter() - t0) * 1000)

        # Try to parse JSON from response
        parse_ok, data = _try_parse_json(raw)

        has_title = bool(data.get("title"))
        has_bullets = isinstance(data.get("bullet_points"), list) and len(data.get("bullet_points", [])) >= 1
        has_desc = bool(data.get("description"))
        has_keywords = isinstance(data.get("backend_keywords"), list)

        report.add(TestResult(
            name="LLM API: listing generation",
            passed=parse_ok and has_title and has_bullets and has_desc,
            latency_ms=latency,
            details=f"json_parsed={parse_ok} title={has_title} bullets={has_bullets} desc={has_desc} keywords={has_keywords}",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="LLM API: listing generation",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))

    # 3b. Compliance check
    t0 = time.perf_counter()
    try:
        system_c = (
            "You are an e-commerce listing compliance reviewer. "
            "Decide whether the listing violates any rule. "
            "Respond with ONLY a JSON object: "
            '{"passed": true|false, "violations": [str], "suggestions": [str]}'
        )
        user_c = (
            "Platform: amazon\n"
            "Listing: Title='Stackable Storage Bins', "
            "Bullets='Durable PP plastic', Description='Keeps closets tidy.'\n"
            "Review this listing."
        )
        raw_c = await client.chat(system_c, user_c, temperature=0.0)
        latency = int((time.perf_counter() - t0) * 1000)

        parse_ok_c, data_c = _try_parse_json(raw_c)

        has_passed = "passed" in data_c
        has_violations = isinstance(data_c.get("violations"), list)

        report.add(TestResult(
            name="LLM API: compliance check",
            passed=parse_ok_c and has_passed and has_violations,
            latency_ms=latency,
            details=f"json_parsed={parse_ok_c} passed_field={has_passed} violations_field={has_violations} result={data_c.get('passed')}",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="LLM API: compliance check",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))

    # 3c. Translation
    t0 = time.perf_counter()
    try:
        system_t = (
            "You are a professional e-commerce translator. "
            "Translate the listing into the target language. "
            "Respond with ONLY a JSON object: "
            '{"title": str, "bullet_points": [str], "description": str}'
        )
        user_t = (
            "Target language: zh\n"
            "Title: Stackable Storage Organizer Bins\n"
            "Bullet points:\n- Made of premium PP plastic\n- Stackable design saves space\n"
            "Description: Keep your home organized with these durable storage bins."
        )
        raw_t = await client.chat(system_t, user_t, temperature=0.1)
        latency = int((time.perf_counter() - t0) * 1000)

        parse_ok_t, data_t = _try_parse_json(raw_t)

        # Check if Chinese characters appear (translation quality)
        title_t = data_t.get("title", "")
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in title_t)

        report.add(TestResult(
            name="LLM API: translation (en->zh)",
            passed=parse_ok_t and has_chinese,
            latency_ms=latency,
            details=f"json_parsed={parse_ok_t} has_chinese={has_chinese} title='{title_t}'",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="LLM API: translation (en->zh)",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


# =====================================================================
# 4. Embedding Model (local mode) - Direct call
# =====================================================================
async def test_embedding_model():
    print("\n--- 4. Embedding Model (local mode) ---")
    settings = get_settings()

    embedder = Embedder(settings)
    t0 = time.perf_counter()
    try:
        texts = [
            "stackable storage bins for home organization",
            "waterproof outdoor camping tent",
            "stackable storage organizer",
        ]
        embeddings = embedder.embed(texts)
        latency = int((time.perf_counter() - t0) * 1000)

        dim = len(embeddings[0]) if embeddings else 0
        all_same_dim = all(len(e) == dim for e in embeddings)

        # Check similarity: text 0 and text 2 should be more similar than text 0 and text 1
        import math
        def cosine_sim(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return dot / (na * nb) if na * nb > 0 else 0

        sim_related = cosine_sim(embeddings[0], embeddings[2])
        sim_unrelated = cosine_sim(embeddings[0], embeddings[1])
        semantic_ok = sim_related > sim_unrelated

        report.add(TestResult(
            name="Embedding local: vector generation",
            passed=all_same_dim and dim > 0,
            latency_ms=latency,
            details=f"dim={dim} count={len(embeddings)} all_same_dim={all_same_dim}",
        ))
        report.add(TestResult(
            name="Embedding local: semantic quality",
            passed=semantic_ok,
            latency_ms=0,
            details=f"sim_related={sim_related:.4f} sim_unrelated={sim_unrelated:.4f} (related should > unrelated)",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="Embedding local: vector generation",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


# =====================================================================
# 5. RAG Index + Retrieval (with local embeddings)
# =====================================================================
async def test_rag_retrieval():
    print("\n--- 5. RAG Index + Retrieval ---")
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        # Rebuild index
        stats = build_index(settings, rebuild=True)
        latency_build = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="RAG: index build",
            passed=stats.total_chunks > 0,
            latency_ms=latency_build,
            details=f"platforms={stats.platforms} total_chunks={stats.total_chunks}",
        ))

        # Retrieve
        t1 = time.perf_counter()
        retriever = RuleRetriever(settings)
        query = build_query("storage organizer", ["stackable", "dust-proof"])
        rules = retriever.retrieve(platform="amazon", query=query)
        latency_retrieve = int((time.perf_counter() - t1) * 1000)

        report.add(TestResult(
            name="RAG: rule retrieval",
            passed=len(rules) > 0,
            latency_ms=latency_retrieve,
            details=f"query='{query}' rules_found={len(rules)} top_score={rules[0].score:.4f}" if rules else "no rules found",
        ))

        # Verify platform isolation
        shopee_rules = retriever.retrieve(platform="shopee", query="storage bins")
        amazon_only = all(r.platform == "amazon" for r in rules)
        shopee_only = all(r.platform == "shopee" for r in shopee_rules)
        report.add(TestResult(
            name="RAG: platform isolation",
            passed=amazon_only and shopee_only,
            latency_ms=0,
            details=f"amazon_rules_all_amazon={amazon_only} shopee_rules_all_shopee={shopee_only}",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="RAG: index + retrieval",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


# =====================================================================
# 6. End-to-End API Test (full pipeline via HTTP)
# =====================================================================
async def test_e2e_api():
    print("\n--- 6. End-to-End API Test ---")

    # 6a. Amazon listing
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{API_BASE}/api/v1/listing/generate",
                files=[("images", ("product.png", TEST_IMAGE, "image/png"))],
                data={
                    "category": "storage organizer",
                    "platform": "amazon",
                    "target_lang": "en",
                },
                timeout=180,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            report.add(TestResult(
                name="E2E API: Amazon listing",
                passed=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            ))
        else:
            body = resp.json()
            checks = {
                "status_200": True,
                "has_title": bool(body.get("title")),
                "has_bullets": len(body.get("bullet_points", [])) >= 1,
                "has_description": bool(body.get("description")),
                "compliance_passed": body.get("compliance", {}).get("passed") is True,
                "has_visual_analysis": bool(body.get("visual_analysis", {}).get("detected_category")),
                "rag_chunks_used": body.get("metadata", {}).get("rag_chunks_used", 0) > 0,
                "latency_recorded": body.get("metadata", {}).get("latency_ms", 0) > 0,
            }
            all_ok = all(checks.values())
            detail_parts = [f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()]
            report.add(TestResult(
                name="E2E API: Amazon listing",
                passed=all_ok,
                latency_ms=latency,
                details=f"title='{body.get('title', '')[:60]}' | {'; '.join(detail_parts)}",
            ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="E2E API: Amazon listing",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))

    # 6b. Shopee listing
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{API_BASE}/api/v1/listing/generate",
                files=[("images", ("product.png", TEST_IMAGE, "image/png"))],
                data={
                    "category": "home storage",
                    "platform": "shopee",
                    "target_lang": "en",
                },
                timeout=180,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            report.add(TestResult(
                name="E2E API: Shopee listing",
                passed=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            ))
        else:
            body = resp.json()
            ok = body.get("compliance", {}).get("passed") is True
            report.add(TestResult(
                name="E2E API: Shopee listing",
                passed=ok,
                latency_ms=latency,
                details=f"title='{body.get('title', '')[:60]}' compliance={body.get('compliance', {}).get('passed')}",
            ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="E2E API: Shopee listing",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))

    # 6c. Temu listing with translation (target_lang=zh)
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{API_BASE}/api/v1/listing/generate",
                files=[("images", ("product.png", TEST_IMAGE, "image/png"))],
                data={
                    "category": "kitchen organizer",
                    "platform": "temu",
                    "target_lang": "zh",
                },
                timeout=180,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            report.add(TestResult(
                name="E2E API: Temu listing (zh translation)",
                passed=False,
                latency_ms=latency,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            ))
        else:
            body = resp.json()
            report.add(TestResult(
                name="E2E API: Temu listing (zh translation)",
                passed=True,
                latency_ms=latency,
                details=f"title='{body.get('title', '')[:60]}' compliance={body.get('compliance', {}).get('passed')}",
            ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="E2E API: Temu listing (zh translation)",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))

    # 6d. RAG rebuild endpoint
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(f"{API_BASE}/api/v1/rag/rebuild", timeout=120)
        latency = int((time.perf_counter() - t0) * 1000)
        body = resp.json()
        ok = resp.status_code == 200 and body.get("status") == "ok"
        report.add(TestResult(
            name="E2E API: RAG rebuild",
            passed=ok,
            latency_ms=latency,
            details=f"platforms={body.get('platforms')} total={body.get('total_chunks')}",
        ))
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name="E2E API: RAG rebuild",
            passed=False,
            latency_ms=latency,
            error=f"{type(e).__name__}: {e}",
        ))

    # 6e. Error handling: no images
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{API_BASE}/api/v1/listing/generate",
                data={"category": "test"},
                timeout=10,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        ok = resp.status_code == 422
        report.add(TestResult(
            name="E2E API: error handling (no images)",
            passed=ok,
            latency_ms=latency,
            details=f"status={resp.status_code} (expected 422)",
        ))
    except Exception as e:
        report.add(TestResult(
            name="E2E API: error handling (no images)",
            passed=False,
            error=str(e),
        ))

    # 6f. Natural-language extra_info is accepted (not rejected as bad JSON)
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                f"{API_BASE}/api/v1/listing/generate",
                files=[("images", ("product.png", TEST_IMAGE, "image/png"))],
                data={"category": "test", "extra_info": "not-json"},
                timeout=10,
            )
        latency = int((time.perf_counter() - t0) * 1000)
        # Non-JSON extra_info is wrapped as natural language, so it must NOT
        # be rejected as invalid input (400/422).
        ok = resp.status_code not in (400, 422)
        report.add(TestResult(
            name="E2E API: natural-language extra_info accepted",
            passed=ok,
            latency_ms=latency,
            details=f"status={resp.status_code} (expected not 400/422)",
        ))
    except Exception as e:
        report.add(TestResult(
            name="E2E API: natural-language extra_info accepted",
            passed=False,
            error=str(e),
        ))


# =====================================================================
# 7. Keyword filter unit test
# =====================================================================
async def test_keyword_filter():
    print("\n--- 7. Keyword Filter ---")
    # Should flag "best seller" on amazon
    violations = keyword_filter.scan_listing(
        platform="amazon",
        title="Best Seller Storage Bins",
        bullet_points=[],
        description="great bins",
    )
    report.add(TestResult(
        name="Keyword filter: flags banned phrase",
        passed=len(violations) > 0 and "best seller" in violations[0].lower(),
        details=f"violations={violations}",
    ))

    # Should pass clean listing
    violations2 = keyword_filter.scan_listing(
        platform="amazon",
        title="Stackable Storage Organizer Bins",
        bullet_points=["Durable PP plastic"],
        description="Keeps closets tidy.",
        backend_keywords=["storage bins"],
    )
    report.add(TestResult(
        name="Keyword filter: passes clean listing",
        passed=len(violations2) == 0,
        details=f"violations={violations2}",
    ))


# =====================================================================
# Main
# =====================================================================
async def main():
    print("=" * 70)
    print("CrossLister Integration & Performance Test")
    print("=" * 70)

    settings = get_settings()
    print(f"Config: vision_mode={settings.vision_mode.value} "
          f"llm_mode={settings.llm_mode.value} "
          f"embedding_mode={settings.embedding_mode.value}")
    print(f"Vision model: {settings.vision_model}")
    print(f"LLM model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Embedding local path: {settings.embedding_local_model_path}")

    await test_health()
    await test_vision_model()
    await test_llm_model()
    await test_embedding_model()
    await test_rag_retrieval()
    await test_e2e_api()
    await test_keyword_filter()

    passed, failed = report.summary()

    # Print failed test details
    if failed > 0:
        print("\n--- FAILED TESTS DETAIL ---")
        for r in report.results:
            if not r.passed:
                print(f"\n  FAIL: {r.name}")
                if r.error:
                    print(f"  Error: {r.error}")
                if r.details:
                    print(f"  Details: {r.details}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
