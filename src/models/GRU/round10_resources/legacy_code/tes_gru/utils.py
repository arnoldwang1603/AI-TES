"""Small serialization / formatting helpers (leaf module)."""
import hashlib

import numpy as np
import torch


def format_duration(seconds):
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"



def sha256_file(path, chunk=1 << 20):
    """Hex sha256 of a file (manual loop: hashlib.file_digest is 3.11+)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_lines(items):
    """Hex sha256 of a list of strings, SORTED and newline-joined (utf-8), so
    the digest names the SET of files, not an ordering."""
    return hashlib.sha256("\n".join(sorted(str(x) for x in items)).encode("utf-8")).hexdigest()


def _to_jsonable(obj):
    """Recursively convert numpy / torch scalars to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


