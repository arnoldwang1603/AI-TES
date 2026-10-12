"""Validate frozen inputs across Git's LF/CRLF checkout conversion.

Keep the original registered SHA256 values. Only known UTF-8 text inputs
may match their LF or CRLF representation; do not parse/reformat CSV/JSON.
Runtime journals, report fingerprints and test-source seals still use the
strict byte hash in paper_report.sha256, as do all binary resources.
"""
import hashlib
from pathlib import Path

from paper_report import sha256

TEXT_SUFFIXES = frozenset((".py", ".json", ".csv", ".txt", ".md"))


def matches_frozen_sha256(path, expected):
    """Accept exact bytes, or exclusively LF/CRLF differences in UTF-8 text."""
    path = Path(path)
    if sha256(path) == expected:
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    data = path.read_bytes()
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    lf = data.replace(b"\r\n", b"\n")
    if b"\r" in lf:
        return False
    return any(hashlib.sha256(candidate).hexdigest() == expected
               for candidate in (lf, lf.replace(b"\n", b"\r\n")))
