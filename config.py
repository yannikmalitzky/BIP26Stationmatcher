"""Reads a local .env file (next to this file) into the environment.

Lines look like  HTW_API_KEY="..."  or  export HTW_API_KEY="...". Variables that are already set in the
environment win. The .env file is listed in .gitignore and must never be committed.
"""
import os
from pathlib import Path


def load_env(path=None):
    path = Path(path or Path(__file__).parent / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
