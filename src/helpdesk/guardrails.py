"""Stage 6: guardrails. Three layers, cheapest first.

1. INPUT   : mask personal data (PII) and refuse obvious prompt-injection attempts.
2. RETRIEVAL GATE (in pipeline.py): if even the best article is not similar enough, refuse; the LLM is never called.
3. OUTPUT  : the answer must cite articles we actually retrieved; the model may say NOT_IN_KB.

Honest limits: regexes catch the obvious cases only. A determined attacker or an unusual PII format will get through,
so these are a first line of defence, not a complete one.
"""
import re
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:[6-9]\d{9}|\d{3}[ -]\d{3}[ -]\d{4})(?!\d)")

_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (?:all |any |the |your )?(?:previous|prior|above|earlier)[\w\s]{0,20}(?:instructions|prompts?|rules)",
    r"disregard (?:all |any |the |your )?(?:previous|prior|above|earlier)?[\w\s]{0,20}(?:instructions|prompts?|rules)",
    r"(?:reveal|show|print|repeat|display)[\w\s]{0,25}(?:system|hidden|initial) (?:prompt|instructions|message)",
    r"you are now (?:dan|an? unrestricted|free|jailbroken)",
    r"\bjailbreak\b", r"developer mode", r"pretend (?:to be|you are)",
)]

_NORMALISE = {**dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-"),   # non-breaking / fancy hyphens
              ord("\u3010"): "[", ord("\u3011"): "]", ord("\uff3b"): "[", ord("\uff3d"): "]"}    # full-width brackets
_ID = re.compile(r"KB-\d{3}")
_GROUP = re.compile(r"\[([^\[\]]*)\]")
NOT_IN_KB = "NOT_IN_KB"


def luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    return len(nums) >= 13 and total % 10 == 0


def redact_pii(text: str):
    """In: raw text. Out: (text with personal data replaced by [EMAIL], [PHONE]..., list of the types found)."""
    found = []

    def sub(pattern, label, check=None):
        nonlocal text
        def repl(m):
            if check and not check(m.group(0)):
                return m.group(0)
            found.append(label)
            return f"[{label}]"
        text = pattern.sub(repl, text)

    sub(_EMAIL, "EMAIL")
    sub(_PAN, "PAN")
    sub(_CARD, "CARD", luhn_ok)
    sub(_AADHAAR, "ID_NUMBER")
    sub(_PHONE, "PHONE")
    return text, sorted(set(found))


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


@dataclass
class InputCheck:
    clean_text: str
    pii_types: list = field(default_factory=list)
    injection: bool = False


def check_input(text: str, gcfg: dict) -> InputCheck:
    clean, pii = (redact_pii(text) if gcfg.get("pii", "redact") == "redact" else (text, []))
    return InputCheck(clean.strip(), pii, bool(gcfg.get("block_prompt_injection", True)) and looks_like_injection(text))


@dataclass
class OutputCheck:
    text: str
    cited_ids: list
    not_in_kb: bool
    invalid_removed: list


def validate_output(answer: str, valid_ids) -> OutputCheck:
    """Keep only citations that point to articles we really gave the model; detect the NOT_IN_KB signal.
    Normalises Unicode hyphens and full-width brackets first (the model may write 【KB-009】 or KB\u2011009) and accepts
    grouped citations such as [KB-009, KB-010]."""
    answer = answer.translate(_NORMALISE)
    valid = set(valid_ids)
    if answer.strip().upper().startswith(NOT_IN_KB):
        return OutputCheck("", [], True, [])
    found = list(dict.fromkeys(_ID.findall(answer)))
    cited = [c for c in found if c in valid]
    invalid = [c for c in found if c not in valid]

    def fix(m):
        ids = _ID.findall(m.group(1))
        if not ids:
            return m.group(0)
        keep = [i for i in ids if i in valid]
        return f"[{', '.join(keep)}]" if keep else ""

    text = re.sub(r"\s+([.,;])", r"\1", re.sub(r" {2,}", " ", _GROUP.sub(fix, answer))).strip()
    return OutputCheck(text, cited, False, invalid)
