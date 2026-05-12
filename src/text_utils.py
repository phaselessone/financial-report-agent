from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
NUMBERING_TITLE_RE = re.compile(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+[、.]|第[一二三四五六七八九十0-9]+[章节部分])")
PAGE_NUMBER_RE = re.compile(r"^(第?\s*\d+\s*页|page\s*\d+|\d+\s*/\s*\d+)$", re.IGNORECASE)

EASTMONEY_NOISE_PATTERNS = (
    "东方财富",
    "eastmoney",
    "choice数据",
    "choice金融终端",
    "股吧",
    "天天基金",
    "扫描下载app",
    "举报邮箱",
    "沪icp",
    "关于我们",
    "友情链接",
    "免责声明",
    "证券开户",
    "基金交易",
    "龙虎机单",
    "全球财经快讯",
)

NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:%|倍|亿元|万元|元|例|家|月|年|季度)?")
YEAR_TOKEN_RE = re.compile(r"(?:19|20)\d{2}年?")
PERCENT_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%")
TOKEN_SPLIT_RE = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")
NEW_FILE_NAME_RE = re.compile(r"^(?P<prefix>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<ref>[A-Za-z0-9]+)_(?P<title>.+)$")
COMMON_STOPWORDS = {
    "根据",
    "当前",
    "证据",
    "无法",
    "确认",
    "报告",
    "行业",
    "公司",
    "问题",
    "主要",
    "相关",
    "以及",
    "其中",
    "进行",
    "分别",
    "哪些",
    "什么",
    "情况",
    "主题",
    "强调",
    "指出",
    "聚焦",
    "近期",
    "这些",
    "显示",
    "认为",
    "提到",
    "围绕",
    "一个",
    "没有",
    "需要",
    "并且",
    "可以",
    "还是",
    "问题",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def normalize_line(text: str) -> str:
    return NON_WORD_RE.sub("", text).lower()


def normalize_for_match(text: str) -> str:
    return normalize_line(normalize_text(text))


def guess_industry(file_name: str) -> str:
    stem = strip_file_extension(Path(file_name).name)
    match = NEW_FILE_NAME_RE.match(stem)
    if match:
        prefix = match.group("prefix")
        prefix_map = {
            "半导体": "semiconductor",
            "新能源": "new_energy",
            "消费": "consumer",
            "白酒": "liquor",
        }
        if prefix in prefix_map:
            return prefix_map[prefix]

    lowered = stem.lower()
    rules = {
        "semiconductor": ("半导体", "电子", "晶圆", "芯片", "存储", "算力", "semicon", "micron", "hynix"),
        "new_energy": ("新能源", "光伏", "风电", "锂电", "储能", "电车", "固态电池", "动力电池"),
        "consumer": ("消费", "商社", "美护", "家电", "零售", "食品饮料", "饮料", "商贸", "内需"),
        "liquor": ("白酒", "酒企", "茅台", "五粮液", "汾酒", "泸州老窖"),
        "agriculture": ("农业", "农", "生猪", "猪价", "种业", "饲料", "usda", "牧渔", "养殖", "牛价"),
        "healthcare": ("医药", "医疗", "健康", "药", "临床", "肿瘤", "肥胖", "biotech", "who", "iqvia"),
    }
    for industry, keywords in rules.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return industry
    return "other"


def guess_source_type(file_name: str, sample_text: str = "") -> str:
    combined = f"{file_name}\n{sample_text}".lower()
    if "东方财富网" in file_name or any(pattern in combined for pattern in ("eastmoney", "东方财富", "choice数据")):
        return "web_export_pdf"
    return "report_pdf"


def make_doc_id(pdf_path: Path) -> str:
    stem = re.sub(r"\s+", "-", pdf_path.stem.strip())
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", stem).strip("-").lower()
    digest = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem[:48]}-{digest}" if stem else f"doc-{digest}"


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    paragraphs = [chunk.strip() for chunk in text.split("\n\n")]
    expanded: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if len(lines) == 1:
            expanded.append(lines[0])
        else:
            expanded.append(" ".join(lines))
    return expanded


def split_with_overlap(text: str, target_size: int, overlap: int) -> list[str]:
    if len(text) <= target_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def is_probable_noise_line(text: str) -> bool:
    stripped = normalize_text(text)
    if not stripped:
        return True
    lowered = stripped.lower()
    if PAGE_NUMBER_RE.fullmatch(stripped):
        return True
    if any(pattern in lowered for pattern in EASTMONEY_NOISE_PATTERNS):
        return True
    if len(stripped) > 120 and sum(char.isdigit() for char in stripped) > 20:
        return True
    if len(stripped.split()) > 20 and len(set(stripped.split())) < len(stripped.split()) * 0.4:
        return True
    return False


def is_title_like(text: str, *, font_size: float = 0.0, baseline_font_size: float = 0.0, is_bold: bool = False) -> bool:
    stripped = normalize_text(text)
    if not stripped or len(stripped) > 60:
        return False
    if is_probable_noise_line(stripped):
        return False
    punctuation_hits = sum(stripped.count(mark) for mark in ("。", "；", "！", "？", ",", "，"))
    if punctuation_hits > 1:
        return False
    if NUMBERING_TITLE_RE.match(stripped):
        return True
    if font_size and baseline_font_size and font_size >= baseline_font_size * 1.15:
        return True
    if is_bold and len(stripped) <= 32:
        return True
    return False


def is_table_like(text: str) -> bool:
    stripped = normalize_text(text)
    if not stripped:
        return False
    lines = [line for line in stripped.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    numeric_hits = sum(len(re.findall(r"\d+(?:\.\d+)?%?", line)) for line in lines)
    delimiter_hits = sum(line.count(" | ") + line.count("\t") for line in lines)
    finance_token_hits = sum(
        sum(token in line for token in ("亿元", "万元", "同比", "环比", "pct", "%", "市盈率", "毛利率"))
        for line in lines
    )
    short_line_ratio = sum(len(line) <= 40 for line in lines) / max(len(lines), 1)
    return (numeric_hits >= 4 and short_line_ratio >= 0.5) or delimiter_hits >= 2 or finance_token_hits >= 5


def tokenize_for_bm25(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jieba is required for BM25 tokenization") from exc
    return [token.strip() for token in jieba.cut_for_search(text) if token.strip()]


def keyword_hits(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def strip_file_extension(file_name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", file_name)


def extract_numeric_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = NUMERIC_TOKEN_RE.findall(normalized)
    tokens.extend(token for token in YEAR_TOKEN_RE.findall(normalized) if token not in tokens)
    tokens.extend(token for token in PERCENT_TOKEN_RE.findall(normalized) if token not in tokens)
    return list(dict.fromkeys(tokens))


def extract_terms(text: str, *, top_k: int = 8, min_length: int = 2) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    try:
        tokens.extend(tokenize_for_bm25(normalized))
    except RuntimeError:
        tokens.extend(TOKEN_SPLIT_RE.findall(normalized))

    counter: Counter[str] = Counter()
    for token in tokens:
        stripped = token.strip().lower()
        if len(stripped) < min_length:
            continue
        if stripped in COMMON_STOPWORDS:
            continue
        if stripped.isdigit():
            continue
        if normalize_for_match(stripped) and len(normalize_for_match(stripped)) < min_length:
            continue
        counter[stripped] += 1
    return [token for token, _ in counter.most_common(top_k)]


def first_sentence(text: str, *, max_chars: int = 80) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    parts = re.split(r"[。！？!?;\n]", normalized)
    sentence = next((part.strip() for part in parts if part.strip()), normalized)
    return sentence[:max_chars].rstrip("，,；; ") + ("..." if len(sentence) > max_chars else "")


def safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", model_name).strip("-")
