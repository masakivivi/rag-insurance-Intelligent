# coding: utf-8
"""输入清洗 + 注入检测（P2）。

设计原则：先保守（只拒明确模式），靠日志观察再调，避免误报拒正常 query。
"""
import re
from dataclasses import dataclass

MAX_QUERY_LEN = 2000

# 控制字符（保留换行/回车/制表）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# 注入模式：role 劫持 / 指令覆盖 / 分隔符伪造。命中即拒。
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|prior|all)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"忽略(以上|之前|前面|所有)的?(指令|提示|说明|规则)", re.IGNORECASE),
    re.compile(r"disregard\s+(previous|above|all)\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\b", re.IGNORECASE),
    re.compile(r"从现在起(你|你是一个)", re.IGNORECASE),
    re.compile(r"^\s*system\s*[:：]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"</?user_input>", re.IGNORECASE),  # 分隔符伪造
    re.compile(r"输出(你|系统)的?(系统提示|prompt|指令)", re.IGNORECASE),
    re.compile(r"(reveal|repeat|print)\s+(your|the)\s+(system|prompt|instructions?)", re.IGNORECASE),
]


@dataclass
class SanitizedQuery:
    raw: str
    sanitized: str
    truncated: bool
    is_injection: bool
    matched_pattern: str | None = None


def sanitize(query: str) -> SanitizedQuery:
    """清洗用户输入并检测注入。

    - 去除控制字符
    - 超长截断并标记
    - 命中注入模式则 is_injection=True（由调用方决定拒绝）
    """
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串")

    truncated = False
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]
        truncated = True

    cleaned = _CONTROL_RE.sub("", query).strip()

    matched = None
    is_injection = False
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            matched = pat.pattern
            is_injection = True
            break

    return SanitizedQuery(
        raw=query,
        sanitized=cleaned,
        truncated=truncated,
        is_injection=is_injection,
        matched_pattern=matched,
    )
