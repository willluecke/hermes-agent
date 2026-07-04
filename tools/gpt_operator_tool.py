#!/usr/bin/env python3
"""GPT operator brief/review tools.

Creates a compact, sanitized decision brief from worker findings, report files,
or raw text. ``gpt_operator_brief`` does not call OpenAI. ``gpt_operator_review``
uses the same sanitizer, then calls OpenAI with only the brief and no tools.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result


DEFAULT_MAX_FINDINGS = 12
DEFAULT_MAX_BRIEF_CHARS = 16_000
DEFAULT_OPENAI_REVIEW_MODEL = "gpt-5.5"
DEFAULT_OPENAI_REVIEW_REASONING = "xhigh"
DEFAULT_OPENAI_REVIEW_MAX_OUTPUT_TOKENS = 1_200
DEFAULT_OPENAI_REVIEW_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_REVIEW_INPUT_CHARS = 18_000
DEFAULT_OPENAI_REVIEW_ENDPOINT = "chat_completions"
DEFAULT_DIRECT_INPUT_CHARS = 4_000
MAX_SOURCE_CHARS_PER_FILE = 60_000
MAX_RAW_TEXT_CHARS = 60_000
MAX_FIELD_CHARS = 900
MAX_RETURN_BRIEF_CHARS = 18_000
MAX_REVIEW_TEXT_CHARS = 12_000

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret)\s*[:=]\s*['\"]?[^'\"\s,;]{6,}"),
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WHITESPACE_RE = re.compile(r"[ \t]+")
BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{160,}$")

TITLE_KEYS = ("title", "summary", "claim", "issue", "bug", "finding", "name")
DETAIL_KEYS = ("description", "details", "rationale", "reason", "impact", "analysis")
EVIDENCE_KEYS = ("evidence", "excerpt", "snippet", "proof", "repro", "test", "observed")
RECOMMEND_KEYS = ("recommendation", "fix", "suggested_fix", "next_step", "action")
PATH_KEYS = ("file", "path", "filepath", "file_path", "location")
DROP_KEYS = {
    "raw",
    "raw_log",
    "log",
    "logs",
    "stdout",
    "stderr",
    "output",
    "transcript",
    "messages",
    "conversation",
    "diff",
    "patch",
    "full_text",
    "content",
}


def _cap(text: Any, limit: int = MAX_FIELD_CHARS) -> str:
    value = "" if text is None else str(text)
    value = _redact(value)
    value = ANSI_RE.sub("", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 24)].rstrip() + " [truncated]"


def _redact(text: str) -> str:
    result = text
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def _first_string(obj: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in obj and obj[key] not in (None, "", [], {}):
            value = obj[key]
            if isinstance(value, (dict, list)):
                return _cap(json.dumps(value, ensure_ascii=False), MAX_FIELD_CHARS)
            return _cap(value)
    return ""


def _line_value(obj: dict[str, Any]) -> str:
    for key in ("line", "line_number", "lineno", "start_line"):
        value = obj.get(key)
        if value not in (None, ""):
            return _cap(value, 40)
    return ""


def _severity_value(obj: dict[str, Any]) -> str:
    return _first_string(obj, ("severity", "priority", "risk"))


def _confidence_value(obj: dict[str, Any]) -> str:
    return _first_string(obj, ("confidence", "certainty"))


def _dict_to_finding(obj: dict[str, Any], source: str) -> dict[str, str] | None:
    lower_keys = {str(k).lower(): k for k in obj.keys()}

    def get_any(keys: Iterable[str]) -> Any:
        for key in keys:
            original = lower_keys.get(key)
            if original is not None:
                return obj.get(original)
        return None

    title = _first_string({k.lower(): v for k, v in obj.items()}, TITLE_KEYS)
    details = _first_string({k.lower(): v for k, v in obj.items()}, DETAIL_KEYS)
    evidence = _first_string({k.lower(): v for k, v in obj.items()}, EVIDENCE_KEYS)
    recommendation = _first_string({k.lower(): v for k, v in obj.items()}, RECOMMEND_KEYS)
    path = _first_string({k.lower(): v for k, v in obj.items()}, PATH_KEYS)
    line = _line_value({k.lower(): v for k, v in obj.items()})
    severity = _severity_value({k.lower(): v for k, v in obj.items()})
    confidence = _confidence_value({k.lower(): v for k, v in obj.items()})

    if not title:
        if details:
            title = details[:160]
        elif evidence:
            title = evidence[:160]

    has_finding_shape = any(get_any(keys) for keys in (TITLE_KEYS, DETAIL_KEYS, EVIDENCE_KEYS))
    if not title or not has_finding_shape:
        return None

    location = path
    if location and line:
        location = f"{location}:{line}"

    return {
        "title": title,
        "severity": severity or "unknown",
        "confidence": confidence or "unknown",
        "location": location or "unspecified",
        "evidence": evidence or details or "No short evidence provided.",
        "recommendation": recommendation or "Needs operator judgment.",
        "source": source,
    }


def _walk_json(value: Any, source: str, findings: list[dict[str, str]], depth: int = 0) -> None:
    if depth > 8 or len(findings) >= 200:
        return
    if isinstance(value, dict):
        maybe = _dict_to_finding(value, source)
        if maybe:
            findings.append(maybe)
        for key, child in value.items():
            if str(key).lower() in DROP_KEYS:
                continue
            _walk_json(child, source, findings, depth + 1)
    elif isinstance(value, list):
        for child in value[:300]:
            _walk_json(child, source, findings, depth + 1)


def _looks_like_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 900:
        return True
    if BASE64ISH_RE.match(stripped):
        return True
    if stripped.startswith(("diff --git", "index ", "+++", "---", "@@", "```")):
        return True
    if stripped.startswith(("{", "}", "[", "]")) and len(stripped) > 180:
        return True
    if stripped.count("/") > 18 and len(stripped) > 240:
        return True
    return False


def _sanitize_free_text(text: str, limit: int = 4_000) -> str:
    text = _redact(ANSI_RE.sub("", text))
    lines: list[str] = []
    blank = False
    total = 0
    for raw_line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if not blank and lines:
                lines.append("")
                total += 1
            blank = True
            continue
        blank = False
        if _looks_like_noise(line):
            continue
        if len(line) > 260:
            line = line[:240].rstrip() + " [truncated line]"
        total += len(line) + 1
        if total > limit:
            lines.append("[additional sanitized text omitted]")
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _read_source(path_text: str) -> tuple[str, str | None]:
    path = Path(path_text).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return "", f"not found: {path_text}"
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > MAX_SOURCE_CHARS_PER_FILE:
            data = data[:MAX_SOURCE_CHARS_PER_FILE] + "\n[rest of source file omitted]"
        return data, None
    except Exception as exc:
        return "", f"could not read {path_text}: {exc}"


def _extract_from_source(text: str, source: str) -> tuple[list[dict[str, str]], str]:
    findings: list[dict[str, str]] = []
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            data = json.loads(stripped)
            _walk_json(data, source, findings)
        except Exception:
            pass
    sanitized = _sanitize_free_text(text, limit=4_000)
    return findings, sanitized


def _normalize_key(finding: dict[str, str]) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", finding.get("title", "").lower()).strip()
    location = re.sub(r"[^a-z0-9/_.:-]+", " ", finding.get("location", "").lower()).strip()
    return f"{location}|{title[:100]}"


def _dedupe_findings(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for finding in findings:
        key = _normalize_key(finding)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
    result.sort(key=lambda f: severity_order.get(f.get("severity", "unknown").lower(), 4))
    return result


def _coerce_manual_findings(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    findings: list[dict[str, str]] = []
    for item in items[:200]:
        if isinstance(item, dict):
            finding = _dict_to_finding(item, "manual_findings")
            if finding:
                findings.append(finding)
        elif isinstance(item, str) and item.strip():
            findings.append(
                {
                    "title": _cap(item, 220),
                    "severity": "unknown",
                    "confidence": "unknown",
                    "location": "unspecified",
                    "evidence": _cap(item),
                    "recommendation": "Needs operator judgment.",
                    "source": "manual_findings",
                }
            )
    return findings


def _render_brief(
    *,
    task: str,
    findings: list[dict[str, str]],
    excerpts: list[tuple[str, str]],
    source_errors: list[str],
    max_findings: int,
    max_brief_chars: int,
) -> str:
    lines: list[str] = [
        "# GPT Operator Brief",
        "",
        "## Decision Needed",
        _cap(task, 1_200) or "Review the sanitized findings and provide high-level direction.",
        "",
        "## Operator Constraints",
        "- This brief is sanitized. Raw logs, full JSON, diffs, secrets, and long transcripts were intentionally excluded.",
        "- Do not ask for raw worker logs unless a specific missing fact is required.",
        "- Give high-level judgment, prioritization, risk assessment, and next actions.",
        "- Hermes (Fable orchestrator + GPT executor subagent) should execute tool work after this review.",
        "",
    ]

    if findings:
        lines.append("## Sanitized Findings")
        for idx, finding in enumerate(findings[:max_findings], 1):
            lines.extend(
                [
                    f"{idx}. {finding['title']}",
                    f"   - Severity: {finding.get('severity', 'unknown')} | Confidence: {finding.get('confidence', 'unknown')}",
                    f"   - Location: {finding.get('location', 'unspecified')}",
                    f"   - Evidence: {_cap(finding.get('evidence', ''), 700)}",
                    f"   - Suggested action: {_cap(finding.get('recommendation', ''), 500)}",
                    f"   - Source: {_cap(finding.get('source', ''), 180)}",
                    "",
                ]
            )
    else:
        lines.extend(["## Sanitized Findings", "No structured findings were extracted.", ""])

    if excerpts:
        lines.append("## Sanitized Source Excerpts")
        for source, excerpt in excerpts[:6]:
            if not excerpt:
                continue
            lines.extend([f"### {source}", excerpt[:2_000].strip(), ""])

    lines.extend(
        [
            "## Exclusions",
            "- Raw logs and command transcripts",
            "- Full JSON result payloads",
            "- Full diffs and patches",
            "- Secrets or token-like strings",
            "- Repeated search output noise",
        ]
    )
    if source_errors:
        lines.extend(["", "## Source Read Issues"])
        lines.extend(f"- {_cap(error, 300)}" for error in source_errors[:12])

    brief = "\n".join(lines).strip() + "\n"
    if len(brief) > max_brief_chars:
        brief = brief[: max(0, max_brief_chars - 80)].rstrip() + "\n\n[brief truncated to configured size]\n"
    return brief


def _int_arg(args: dict[str, Any], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _create_operator_brief(
    args: dict[str, Any],
    *,
    return_brief_default: bool = True,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    task = _cap(args.get("task", ""), 1_200)
    if not task:
        return None, None, "task is required"

    max_findings = _int_arg(args, "max_findings", DEFAULT_MAX_FINDINGS, 1, 40)
    max_brief_chars = _int_arg(args, "max_brief_chars", DEFAULT_MAX_BRIEF_CHARS, 4_000, 50_000)

    source_paths = args.get("source_paths") or []
    if not isinstance(source_paths, list):
        return None, None, "source_paths must be an array of file paths"

    findings = _coerce_manual_findings(args.get("findings"))
    excerpts: list[tuple[str, str]] = []
    source_errors: list[str] = []

    for path in source_paths[:20]:
        source_label = str(path)
        text, error = _read_source(source_label)
        if error:
            source_errors.append(error)
            continue
        extracted, excerpt = _extract_from_source(text, source_label)
        findings.extend(extracted)
        if excerpt:
            excerpts.append((source_label, excerpt))

    raw_text = args.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        text = raw_text[:MAX_RAW_TEXT_CHARS]
        extracted, excerpt = _extract_from_source(text, "raw_text")
        findings.extend(extracted)
        if excerpt:
            excerpts.append(("raw_text", excerpt))

    deduped = _dedupe_findings(findings)
    brief = _render_brief(
        task=task,
        findings=deduped,
        excerpts=excerpts,
        source_errors=source_errors,
        max_findings=max_findings,
        max_brief_chars=max_brief_chars,
    )

    out_dir_arg = args.get("output_dir")
    out_dir = Path(out_dir_arg).expanduser() if out_dir_arg else get_hermes_home() / "operator-briefs"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gpt_operator_brief_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text(brief, encoding="utf-8")
    except Exception as exc:
        return None, None, f"could not write brief: {exc}"

    return_brief = bool(args.get("return_brief", return_brief_default))
    payload = {
        "success": True,
        "brief_path": str(path),
        "brief_chars": len(brief),
        "finding_count": len(deduped),
        "returned_findings": min(len(deduped), max_findings),
        "source_count": len(source_paths),
        "source_errors": source_errors[:12],
        "instructions": (
            "Use this file as the only input to a fresh GPT operator session. "
            "Do not attach raw logs, full worker JSON, diffs, or transcripts."
        ),
    }
    if return_brief:
        payload["brief"] = brief[:MAX_RETURN_BRIEF_CHARS]
        if len(brief) > MAX_RETURN_BRIEF_CHARS:
            payload["brief_truncated_in_tool_result"] = True
    return brief, payload, None


def gpt_operator_brief_tool(args: dict[str, Any], **_: Any) -> str:
    brief, payload, error = _create_operator_brief(args, return_brief_default=True)
    if error:
        return tool_error(error)
    return tool_result(payload)


def _openai_base_url() -> str:
    config_base = ""
    try:
        from hermes_cli.config import get_env_value

        config_base = get_env_value("OPENAI_BASE_URL") or get_env_value("OPENAI_API_BASE") or ""
    except Exception:
        config_base = ""
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or config_base
        or "https://api.openai.com/v1"
    ).rstrip("/")
    if base == "https://api.openai.com":
        base = f"{base}/v1"
    return base


def _extract_openai_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if isinstance(content, dict):
            content = [content]
        for chunk in content or []:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text") or chunk.get("output_text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    if parts:
        return "\n\n".join(parts).strip()

    choices = data.get("choices", [])
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
    return ""


def _openai_usage_summary(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return {}

    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    if not isinstance(output_details, dict):
        output_details = {}

    return {
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "cached_input_tokens": int(details.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _call_openai_operator_review(
    *,
    review_input: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout_seconds: int,
    endpoint: str,
) -> tuple[str, dict[str, int], str | None]:
    config_key = ""
    try:
        from hermes_cli.config import get_env_value

        config_key = get_env_value("OPENAI_API_KEY") or ""
    except Exception:
        config_key = ""
    api_key = (os.environ.get("OPENAI_API_KEY") or config_key).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    instructions = (
        "You are the expensive GPT operator review path for Hermes. "
        "You receive only a sanitized brief, never raw worker logs. "
        "Do high-level judgment: prioritize, identify risks, call out weak evidence, "
        "and give concise next actions for the Hermes worker to execute. "
        "Do not ask for raw logs unless a specific missing fact blocks the decision."
    )
    endpoint = endpoint if endpoint in {"responses", "chat_completions"} else DEFAULT_OPENAI_REVIEW_ENDPOINT
    if endpoint == "chat_completions":
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": review_input},
            ],
            "max_completion_tokens": max_output_tokens,
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        api_path = "chat/completions"
    else:
        body = {
            "model": model,
            "instructions": instructions,
            "input": review_input,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        api_path = "responses"

    request = urllib.request.Request(
        f"{_openai_base_url()}/{api_path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {_redact(_cap(raw_error, 1_200))}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {_redact(_cap(exc, 500))}") from exc

    data = json.loads(raw)
    response_text = _extract_openai_text(data)
    usage = _openai_usage_summary(data)
    response_id = data.get("id") if isinstance(data.get("id"), str) else None
    return response_text, usage, response_id


def gpt_operator_review_tool(args: dict[str, Any], **_: Any) -> str:
    dry_run = bool(args.get("dry_run", False))
    if not dry_run and args.get("confirm_openai_call") is not True:
        return tool_error(
            "confirm_openai_call must be true to call OpenAI. "
            "Use dry_run=true to generate and inspect the sanitized request without spending tokens."
        )

    use_brief = bool(args.get("use_brief", True))
    direct_input = args.get("direct_input")
    if direct_input is None:
        direct_input = args.get("input_text")

    if not use_brief:
        raw_direct = direct_input if isinstance(direct_input, str) and direct_input.strip() else args.get("task", "")
        direct_limit = _int_arg(args, "max_direct_input_chars", DEFAULT_DIRECT_INPUT_CHARS, 1, 8_000)
        review_input = _sanitize_free_text(str(raw_direct), limit=direct_limit)
        if not review_input:
            return tool_error("direct_input/input_text or task is required when use_brief=false")
        out_dir_arg = args.get("output_dir")
        out_dir = Path(out_dir_arg).expanduser() if out_dir_arg else get_hermes_home() / "operator-briefs"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"gpt_operator_direct_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            path.write_text(review_input, encoding="utf-8")
        except Exception as exc:
            return tool_error(f"could not write direct input audit file: {exc}")
        payload: dict[str, Any] = {
            "success": True,
            "brief_path": str(path),
            "brief_chars": len(review_input),
            "input_mode": "direct",
            "finding_count": 0,
            "returned_findings": 0,
            "source_count": 0,
            "source_errors": [],
            "instructions": (
                "Direct GPT operator input was sanitized and saved for audit. "
                "No terminal curl or Hermes provider switch is needed."
            ),
        }
    else:
        brief_args = dict(args)
        brief_args.setdefault("return_brief", False)
        brief, payload, error = _create_operator_brief(brief_args, return_brief_default=False)
        if error:
            return tool_error(error)
        assert brief is not None and payload is not None
        review_input = brief
        payload["input_mode"] = "brief"

    model = _cap(
        args.get("model")
        or os.environ.get("HERMES_GPT_OPERATOR_MODEL")
        or DEFAULT_OPENAI_REVIEW_MODEL,
        80,
    )
    reasoning_effort = _cap(
        args.get("reasoning_effort")
        or os.environ.get("HERMES_GPT_OPERATOR_REASONING")
        or DEFAULT_OPENAI_REVIEW_REASONING,
        40,
    )
    max_output_tokens = _int_arg(
        args,
        "max_output_tokens",
        DEFAULT_OPENAI_REVIEW_MAX_OUTPUT_TOKENS,
        128,
        4_000,
    )
    timeout_seconds = _int_arg(
        args,
        "timeout_seconds",
        DEFAULT_OPENAI_REVIEW_TIMEOUT_SECONDS,
        10,
        300,
    )
    max_openai_input_chars = _int_arg(
        args,
        "max_openai_input_chars",
        DEFAULT_OPENAI_REVIEW_INPUT_CHARS,
        4_000,
        24_000,
    )
    endpoint = _cap(
        args.get("endpoint")
        or args.get("api_endpoint")
        or os.environ.get("HERMES_GPT_OPERATOR_ENDPOINT")
        or DEFAULT_OPENAI_REVIEW_ENDPOINT,
        40,
    )
    if endpoint not in {"responses", "chat_completions"}:
        return tool_error("endpoint must be 'responses' or 'chat_completions'")

    if len(review_input) > max_openai_input_chars:
        return tool_error(
            f"sanitized GPT input is {len(review_input):,} chars, above max_openai_input_chars "
            f"{max_openai_input_chars:,}; reduce max_findings/source text before GPT review"
        )

    review_payload = {
        **payload,
        "success": True,
        "openai_called": False,
        "openai_request": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "brief_chars": len(review_input),
            "tools": "disabled",
            "endpoint": endpoint,
        },
    }

    if dry_run:
        review_payload["dry_run"] = True
        review_payload["instructions"] = (
            "Dry run only. No OpenAI call was made. Set confirm_openai_call=true "
            "and dry_run=false to send only this sanitized brief to OpenAI."
        )
        if bool(args.get("return_brief", False)):
            review_payload["brief"] = review_input[:MAX_RETURN_BRIEF_CHARS]
        return tool_result(review_payload)

    try:
        review_text, usage, response_id = _call_openai_operator_review(
            review_input=review_input,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            endpoint=endpoint,
        )
    except Exception as exc:
        return tool_error(str(exc))

    review_payload.update(
        {
            "openai_called": True,
            "openai_response_id": response_id,
            "openai_usage": usage,
            "review": _redact(review_text)[:MAX_REVIEW_TEXT_CHARS],
            "instructions": (
                "This GPT review saw only the sanitized brief. Execute follow-up work "
                "with the normal Hermes tool path unless another explicit "
                "operator review is requested."
            ),
        }
    )
    if len(review_text) > MAX_REVIEW_TEXT_CHARS:
        review_payload["review_truncated_in_tool_result"] = True
    if bool(args.get("return_brief", False)):
        review_payload["brief"] = review_input[:MAX_RETURN_BRIEF_CHARS]
    return tool_result(review_payload)


GPT_OPERATOR_BRIEF_SCHEMA = {
    "name": "gpt_operator_brief",
    "description": (
        "Create a sanitized GPT operator brief from worker findings, report files, "
        "or raw text. Use after cheap agents/searches finish and before asking GPT "
        "for high-level judgment. This tool strips secrets, avoids raw logs/full "
        "JSON/diffs, dedupes findings, writes a brief file, and returns a compact "
        "GPT-ready brief. It does not call OpenAI."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Decision or high-level question GPT should answer from the sanitized brief.",
            },
            "findings": {
                "type": "array",
                "description": "Optional structured findings. Each item can include title, severity, confidence, path/location, evidence, and recommendation.",
                "items": {"type": "object"},
            },
            "source_paths": {
                "type": "array",
                "description": "Optional local report files to scan and summarize. Use final result files, not raw logs, when possible.",
                "items": {"type": "string"},
            },
            "raw_text": {
                "type": "string",
                "description": "Optional pasted findings text. Secrets and noisy lines are redacted/removed.",
            },
            "max_findings": {
                "type": "integer",
                "description": "Maximum findings to include in the brief. Default 12, max 40.",
                "default": DEFAULT_MAX_FINDINGS,
            },
            "max_brief_chars": {
                "type": "integer",
                "description": "Maximum brief size in characters. Default 16000, max 50000.",
                "default": DEFAULT_MAX_BRIEF_CHARS,
            },
            "output_dir": {
                "type": "string",
                "description": "Optional directory for the generated brief. Defaults to ~/.hermes/operator-briefs.",
            },
            "return_brief": {
                "type": "boolean",
                "description": "Whether to include the generated brief in the tool result. Default true.",
                "default": True,
            },
        },
        "required": ["task"],
    },
}


GPT_OPERATOR_REVIEW_SCHEMA = {
    "name": "gpt_operator_review",
    "description": (
        "Call OpenAI for explicit GPT operator review with tools disabled. Use "
        "this instead of terminal/curl, even for tiny smoke tests. By default it "
        "creates a sanitized brief from findings; for direct prompts like 'say hi "
        "chat', set use_brief=false and direct_input/input_text. Never change "
        "Hermes model.provider/model.default to use this tool. Requires "
        "confirm_openai_call=true unless dry_run=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Decision or high-level question GPT should answer from the sanitized brief.",
            },
            "confirm_openai_call": {
                "type": "boolean",
                "description": "Must be true to spend OpenAI tokens. Use dry_run=true to inspect the request without calling OpenAI.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Generate the sanitized brief and request summary without calling OpenAI. Default false.",
                "default": False,
            },
            "use_brief": {
                "type": "boolean",
                "description": "When true, build a sanitized operator brief from findings/sources. When false, send sanitized direct_input/input_text or task as the OpenAI input. Default true.",
                "default": True,
            },
            "direct_input": {
                "type": "string",
                "description": "Direct prompt to send to GPT when use_brief=false. Use for small smoke tests or exact-answer requests instead of terminal/curl.",
            },
            "input_text": {
                "type": "string",
                "description": "Alias for direct_input when use_brief=false.",
            },
            "max_direct_input_chars": {
                "type": "integer",
                "description": "Hard cap on direct input characters before OpenAI. Default 4000, max 8000.",
                "default": DEFAULT_DIRECT_INPUT_CHARS,
            },
            "findings": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["findings"],
            "source_paths": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["source_paths"],
            "raw_text": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["raw_text"],
            "max_findings": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["max_findings"],
            "max_brief_chars": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["max_brief_chars"],
            "max_openai_input_chars": {
                "type": "integer",
                "description": "Hard cap on sanitized brief characters sent to OpenAI. Default 18000, max 24000.",
                "default": DEFAULT_OPENAI_REVIEW_INPUT_CHARS,
            },
            "endpoint": {
                "type": "string",
                "description": "OpenAI endpoint to use: chat_completions or responses. Default chat_completions.",
                "default": DEFAULT_OPENAI_REVIEW_ENDPOINT,
                "enum": ["chat_completions", "responses"],
            },
            "model": {
                "type": "string",
                "description": "OpenAI model for operator review. Defaults to HERMES_GPT_OPERATOR_MODEL or gpt-5.5.",
                "default": DEFAULT_OPENAI_REVIEW_MODEL,
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort sent to OpenAI Responses API. Defaults to HERMES_GPT_OPERATOR_REASONING or xhigh.",
                "default": DEFAULT_OPENAI_REVIEW_REASONING,
            },
            "max_output_tokens": {
                "type": "integer",
                "description": "Maximum review output tokens. Default 1200, max 4000.",
                "default": DEFAULT_OPENAI_REVIEW_MAX_OUTPUT_TOKENS,
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "HTTP timeout for the OpenAI call. Default 120 seconds.",
                "default": DEFAULT_OPENAI_REVIEW_TIMEOUT_SECONDS,
            },
            "output_dir": GPT_OPERATOR_BRIEF_SCHEMA["parameters"]["properties"]["output_dir"],
            "return_brief": {
                "type": "boolean",
                "description": "Whether to include the sanitized brief in the tool result. Default false for review calls.",
                "default": False,
            },
        },
        "required": ["task", "confirm_openai_call"],
    },
}


def check_gpt_operator_brief_requirements() -> bool:
    return True


registry.register(
    name="gpt_operator_brief",
    toolset="gpt_operator",
    schema=GPT_OPERATOR_BRIEF_SCHEMA,
    handler=gpt_operator_brief_tool,
    check_fn=check_gpt_operator_brief_requirements,
    description=GPT_OPERATOR_BRIEF_SCHEMA["description"],
    emoji="",
    max_result_size_chars=20_000,
)

registry.register(
    name="gpt_operator_review",
    toolset="gpt_operator",
    schema=GPT_OPERATOR_REVIEW_SCHEMA,
    handler=gpt_operator_review_tool,
    check_fn=check_gpt_operator_brief_requirements,
    description=GPT_OPERATOR_REVIEW_SCHEMA["description"],
    emoji="",
    max_result_size_chars=20_000,
)
