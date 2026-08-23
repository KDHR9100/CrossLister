"""Batch product import: CSV/Excel parsing and validation."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Template column definitions
TEMPLATE_COLUMNS = [
    ("商品类目", "category", True),
    ("目标平台", "platform", True),
    ("目标语言", "target_lang", True),
    ("补充信息", "extra_info", False),
]

VALID_PLATFORMS = {"amazon", "shopee", "temu"}
VALID_LANGUAGES = {"en", "zh", "ja", "ko", "es", "fr", "de", "pt", "th", "vi", "id", "ms"}


@dataclass
class ParsedProduct:
    """A single parsed product row."""
    row_number: int
    category: str = ""
    platform: str = "amazon"
    target_lang: str = "en"
    extra_info: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ParseResult:
    """Result of parsing a batch import file."""
    products: list[ParsedProduct] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return len(self.products)

    @property
    def valid_count(self) -> int:
        return sum(1 for p in self.products if p.is_valid)

    @property
    def error_count(self) -> int:
        return sum(1 for p in self.products if not p.is_valid)


def generate_csv_template() -> str:
    """Generate a CSV template string with headers and example rows."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    headers = [col[0] for col in TEMPLATE_COLUMNS]
    writer.writerow(headers)

    # Example row 1
    writer.writerow([
        "Home & Kitchen > Storage & Organization",
        "amazon",
        "en",
        '{"brand": "MyBrand", "material": "plastic"}',
    ])

    # Example row 2
    writer.writerow([
        "Electronics > Accessories",
        "shopee",
        "zh",
        "品牌是XX，材质为硅胶，售价约20美元",
    ])

    return output.getvalue()


def parse_csv(content: str) -> ParseResult:
    """Parse CSV content and validate each row."""
    result = ParseResult()

    logger.info("batch_import.parse_csv.start", content_length=len(content))

    try:
        reader = csv.DictReader(io.StringIO(content))
    except Exception as e:
        logger.error("batch_import.parse_csv.dictreader_failed", error=str(e))
        result.errors.append(f"CSV 解析失败: {e}")
        return result

    if not reader.fieldnames:
        logger.warning("batch_import.parse_csv.no_headers")
        result.errors.append("CSV 文件为空或缺少表头")
        return result

    logger.info("batch_import.parse_csv.headers_detected", headers=reader.fieldnames)

    # Check required columns
    required_cols = {col[0] for col in TEMPLATE_COLUMNS if col[2]}
    actual_cols = set(reader.fieldnames)
    missing_cols = required_cols - actual_cols
    if missing_cols:
        logger.error("batch_import.parse_csv.missing_columns", missing=sorted(missing_cols))
        result.errors.append(f"缺少必填列: {', '.join(sorted(missing_cols))}")
        return result

    # Parse each row
    for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
        product = _parse_row(row, row_num)
        result.products.append(product)
        if product.is_valid:
            logger.info("batch_import.parse_csv.row_valid", row=row_num, category=product.category)
        else:
            logger.warning("batch_import.parse_csv.row_invalid", row=row_num, errors=product.errors)

    logger.info(
        "batch_import.parse_csv.done",
        total=result.total_rows,
        valid=result.valid_count,
        errors=result.error_count,
    )
    return result


def parse_excel(content: bytes) -> ParseResult:
    """Parse Excel content and validate each row."""
    logger.info("batch_import.parse_excel.start", content_length=len(content))

    try:
        import openpyxl
    except ImportError:
        logger.error("batch_import.parse_excel.openpyxl_not_installed")
        return ParseResult(errors=["openpyxl 未安装，无法解析 Excel 文件"])

    result = ParseResult()

    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        ws = wb.active
    except Exception as e:
        logger.error("batch_import.parse_excel.load_failed", error=str(e))
        result.errors.append(f"Excel 解析失败: {e}")
        return result

    # Get headers from first row
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        logger.warning("batch_import.parse_excel.empty_file")
        result.errors.append("Excel 文件为空")
        return result

    headers = [str(h).strip() if h else "" for h in rows[0]]
    logger.info("batch_import.parse_excel.headers_detected", headers=headers)

    # Check required columns
    required_cols = {col[0] for col in TEMPLATE_COLUMNS if col[2]}
    actual_cols = set(headers)
    missing_cols = required_cols - actual_cols
    if missing_cols:
        logger.error("batch_import.parse_excel.missing_columns", missing=sorted(missing_cols))
        result.errors.append(f"缺少必填列: {', '.join(sorted(missing_cols))}")
        return result

    # Create column index mapping
    col_map = {h: i for i, h in enumerate(headers)}

    # Parse each data row
    for row_num, row in enumerate(rows[1:], start=2):
        row_dict = {}
        for col_name in headers:
            idx = col_map.get(col_name)
            if idx is not None and idx < len(row):
                row_dict[col_name] = row[idx]
        product = _parse_row(row_dict, row_num)
        result.products.append(product)
        if product.is_valid:
            logger.info("batch_import.parse_excel.row_valid", row=row_num, category=product.category)
        else:
            logger.warning("batch_import.parse_excel.row_invalid", row=row_num, errors=product.errors)

    logger.info(
        "batch_import.parse_excel.done",
        total=result.total_rows,
        valid=result.valid_count,
        errors=result.error_count,
    )
    return result


def _parse_row(row: dict, row_num: int) -> ParsedProduct:
    """Parse and validate a single row."""
    logger.debug("batch_import.parse_row.start", row=row_num, raw_data=dict(row))
    product = ParsedProduct(row_number=row_num)

    # Extract and validate category
    category = str(row.get("商品类目", "") or "").strip()
    if not category:
        logger.debug("batch_import.parse_row.empty_category", row=row_num)
        product.errors.append("商品类目不能为空")
    product.category = category

    # Extract and validate platform
    platform = str(row.get("目标平台", "") or "").strip().lower()
    if not platform:
        logger.debug("batch_import.parse_row.empty_platform", row=row_num)
        product.errors.append("目标平台不能为空")
    elif platform not in VALID_PLATFORMS:
        logger.debug("batch_import.parse_row.invalid_platform", row=row_num, platform=platform)
        product.errors.append(f"无效的目标平台 '{platform}'，可选值: {', '.join(sorted(VALID_PLATFORMS))}")
    product.platform = platform or "amazon"

    # Extract and validate target language
    target_lang = str(row.get("目标语言", "") or "").strip().lower()
    if not target_lang:
        logger.debug("batch_import.parse_row.empty_language", row=row_num)
        product.errors.append("目标语言不能为空")
    elif target_lang not in VALID_LANGUAGES:
        logger.debug("batch_import.parse_row.invalid_language", row=row_num, language=target_lang)
        product.errors.append(f"无效的目标语言 '{target_lang}'，可选值: {', '.join(sorted(VALID_LANGUAGES))}")
    product.target_lang = target_lang or "en"

    # Extract extra_info (optional)
    extra_info = str(row.get("补充信息", "") or "").strip()
    product.extra_info = extra_info

    logger.debug(
        "batch_import.parse_row.done",
        row=row_num,
        valid=product.is_valid,
        errors=product.errors if not product.is_valid else None,
    )
    return product
