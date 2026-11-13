"""Text normalisation shared by every lexical retriever.

In : raw text (a query or an article)
Out: a list of tokens, e.g. "Passwords expire in 15 minutes" -> ["password", "expire", "15", "minute"]

Why one shared function: if queries and articles are not cleaned the SAME way,
"passwords" in an article would never match "password" in a query.
"""
import re

_TOKEN = re.compile(r"[a-z0-9]+")   # keeps error codes like 0x80070005 as ONE token

STOPWORDS = frozenset("""
a an and are as at be but by can do does for from had has have how i if in into is it its my me
no not of on or our so such that the their then there these they this to was we were what when
where which who why will with would you your am been being than too very
""".split())


def light_stem(tok: str) -> str:
    """Tiny, deterministic stemmer: plural/-ing/-ed endings only. Tokens with digits are untouched
    so error codes and numbers stay exact."""
    if any(c.isdigit() for c in tok) or len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        return tok[:-2]
    if tok.endswith("ing") and len(tok) > 5:
        tok = tok[:-3]
        return tok[:-1] if len(tok) > 2 and tok[-1] == tok[-2] and tok[-1] not in "sl" else tok
    if tok.endswith("ed") and len(tok) > 4:
        tok = tok[:-2]
        return tok[:-1] if len(tok) > 2 and tok[-1] == tok[-2] and tok[-1] not in "sl" else tok
    if tok.endswith(("ches", "shes", "xes", "zes", "sses")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def make_analyzer(lexical_cfg: dict):
    """Build the tokenizer function from config, so settings are not hard-coded."""
    stem = lexical_cfg.get("stemming", "light") == "light"
    drop_stop = lexical_cfg.get("remove_stopwords", True)

    def analyze(text: str) -> list:
        toks = _TOKEN.findall(text.lower())
        if drop_stop:
            toks = [t for t in toks if t not in STOPWORDS]
        return [light_stem(t) for t in toks] if stem else toks

    return analyze
