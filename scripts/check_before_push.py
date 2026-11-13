"""Pre-push safety check. Lists what git WOULD commit (respecting .gitignore) and flags problems.

    python scripts/check_before_push.py

FAIL (exit code 1): a real .env, API-key-looking strings, files over 5 MB, zip/pdf/npy files.
WARN (review by eye): e-mail addresses, personal file paths such as C:\\Users\\<name>.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = [re.compile(p) for p in (r"gsk_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{20,}",
                                   r"ghp_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}",
                                   r"(?i)api[_-]?key\s*[:=]\s*['\"]?(?!your_key_here)[A-Za-z0-9_\-]{24,}")]
PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+|/Users/[^/\s\"']+|/home/[^/\s\"']+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
OK_EMAIL_DOMAINS = ("corp.com", "example.com", "users.noreply.github.com")
BAD_SUFFIX = (".zip", ".pdf", ".npy", ".pem", ".key")


def files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        return [ROOT / p for p in out if (ROOT / p).is_file()]
    except Exception:
        print("Not a git repository yet (or git missing): scanning every file, .gitignore NOT applied.\n")
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts]


fails, warns = [], []
fs = files()
for f in fs:
    rel = f.relative_to(ROOT).as_posix()
    if f.name == ".env" or (f.name.startswith(".env.") and f.name != ".env.example"):
        fails.append(f"{rel}: environment file would be committed")
    if f.suffix.lower() in BAD_SUFFIX:
        fails.append(f"{rel}: {f.suffix} files should not be committed")
    if f.stat().st_size > 5 * 1024 * 1024:
        fails.append(f"{rel}: larger than 5 MB")
    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif") or f.stat().st_size > 2 * 1024 * 1024:
        continue
    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        continue
    if rel == "scripts/check_before_push.py":
        continue
    for pat in SECRET:
        m = pat.search(text)
        if m:
            fails.append(f"{rel}: looks like a secret ({m.group(0)[:8]}...)")
    for m in PATH.finditer(text):
        warns.append(f"{rel}: personal path {m.group(0)}")
        break
    for m in EMAIL.finditer(text):
        if not m.group(0).lower().endswith(OK_EMAIL_DOMAINS):
            warns.append(f"{rel}: e-mail address {m.group(0)}")
            break

print(f"Checked {len(fs)} files that would be committed.")
for w in warns:
    print("WARN ", w)
for x in fails:
    print("FAIL ", x)
if fails:
    print("\nNot safe to push yet.")
    sys.exit(1)
print("\nNo blocking problems found." + (" Review the warnings above by eye." if warns else ""))
