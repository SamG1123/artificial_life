"""
Compressed Storage — lossless gzip-based compression for all memory
and learning persistence files.

All JSON/JSONL data is stored as gzip-compressed files (.json.gz / .jsonl.gz),
typically achieving 5-10x size reduction with zero data loss.

Provides:
  * save_json / load_json  — compressed JSON read/write (dict or list)
  * open_append            — gzip append handle for JSONL streaming
  * open_read_lines        — iterate compressed JSONL line by line
  * migrate_if_needed      — transparently upgrade uncompressed → compressed

Thread-safety: callers are responsible for their own locking.
"""

import gzip
import json
import os
import shutil
from pathlib import Path


# ── JSON (dict / list) ───────────────────────────────────────────

def save_json(path: str, data, indent: int | None = None) -> None:
    """Write *data* as gzip-compressed JSON.

    The file is written atomically (write to tmp then rename) to
    prevent corruption if the process is killed mid-write.
    """
    gz_path = _ensure_gz_ext(path)
    tmp_path = gz_path + ".tmp"
    raw = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write(raw)
    # Atomic rename (Windows: os.replace handles overwriting)
    os.replace(tmp_path, gz_path)


def load_json(path: str, default=None):
    """Load gzip-compressed JSON. Falls back to plain JSON if .gz missing.

    Returns *default* if neither compressed nor plain file exists.
    """
    gz_path = _ensure_gz_ext(path)

    # Try compressed first
    if os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback: plain JSON (pre-migration)
    plain = _strip_gz_ext(path)
    if os.path.exists(plain):
        try:
            with open(plain, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return default


# ── JSONL (append-only log files) ────────────────────────────────

def open_append(path: str):
    """Open a gzip file for appending JSONL lines.

    Returns a gzip file handle. Caller must close it.
    """
    gz_path = _ensure_gz_ext(path)
    return gzip.open(gz_path, "at", encoding="utf-8", compresslevel=6)


def append_line(path: str, entry: dict) -> None:
    """Append a single JSON line to a gzip JSONL file."""
    gz_path = _ensure_gz_ext(path)
    with gzip.open(gz_path, "at", encoding="utf-8", compresslevel=6) as f:
        f.write(json.dumps(entry, default=str) + "\n")


def iter_lines(path: str):
    """Yield parsed JSON objects from a gzip-compressed JSONL file.

    Falls back to plain text if no .gz file exists.
    """
    gz_path = _ensure_gz_ext(path)

    if os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except Exception:
            pass
        return

    # Fallback: plain JSONL
    plain = _strip_gz_ext(path)
    if os.path.exists(plain):
        try:
            with open(plain, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except Exception:
            pass


# ── Migration helper ─────────────────────────────────────────────

def migrate_if_needed(path: str) -> bool:
    """If a plain-text file exists but no .gz version, compress it in-place.

    Returns True if migration occurred.
    """
    plain = _strip_gz_ext(path)
    gz_path = _ensure_gz_ext(path)

    if os.path.exists(plain) and not os.path.exists(gz_path):
        try:
            with open(plain, "rb") as f_in:
                with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(plain)
            print(f"[Storage] Migrated {plain} → {gz_path}")
            return True
        except Exception as e:
            print(f"[Storage] Migration failed for {plain}: {e}")
    return False


def migrate_directory(directory: str, pattern: str = "*.json") -> int:
    """Migrate all matching plain files in a directory to .gz.

    Returns count of files migrated.
    """
    count = 0
    d = Path(directory)
    if not d.exists():
        return 0
    for f in d.glob(pattern):
        gz = f.with_suffix(f.suffix + ".gz")
        if not gz.exists():
            if migrate_if_needed(str(f)):
                count += 1
    return count


# ── Internal helpers ─────────────────────────────────────────────

def _ensure_gz_ext(path: str) -> str:
    """Add .gz extension if not already present."""
    if not path.endswith(".gz"):
        return path + ".gz"
    return path


def _strip_gz_ext(path: str) -> str:
    """Remove .gz extension if present."""
    if path.endswith(".gz"):
        return path[:-3]
    return path
