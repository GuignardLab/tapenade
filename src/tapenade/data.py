from __future__ import annotations

import shutil
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Iterable

_SENTINELS: set[str] = {".keep", ".gitkeep", ".placeholder"}  # files that *don’t* count as data


def _is_effectively_empty(path: Path, sentinels: Iterable[str]) -> bool:
    """True if folder contains nothing except sentinel files."""
    try:
        return all(p.name in sentinels for p in path.iterdir())
    except FileNotFoundError:
        return True


def get_path_to_data() -> Path:
    """
    Ensure `tapenade/notebooks/demo_data` contains data; download if still empty.
    Returns
    -------
    pathlib.Path pointing to the data directory (guaranteed to exist).
    """
    package = "tapenade.notebooks"
    subfolder = "demo_data"
    url = "https://example.com/demo.zip"
    sentinels = _SENTINELS
    # ── locate a *writable* directory ────────────────────────────────
    try:
        base = resources.files(package)         # works for wheels *and* -e installs
        data_dir = base / subfolder
        data_dir.mkdir(parents=True, exist_ok=True)
    except (ModuleNotFoundError, FileNotFoundError):
        tmp_root = Path(tempfile.gettempdir()) / f"{package.replace('.', '_')}_data"
        tmp_root.mkdir(exist_ok=True)
        data_dir = tmp_root

    # ── download if still empty ──────────────────────────────────────
    if _is_effectively_empty(data_dir, sentinels):
        print(f"🔽 First run – downloading data into {data_dir} …")
        tmp = data_dir / "payload"
        urllib.request.urlretrieve(url, tmp)     # <- simple, std-lib only
        try:
            shutil.unpack_archive(tmp, data_dir)
            tmp.unlink()                         # remove archive after unpack
        except shutil.ReadError:
            # Not an archive (e.g. CSV) – just leave it where it is
            pass
        print("✅ Data ready")
    else:
        print(f"✅ Using existing data in {data_dir}")

    return data_dir