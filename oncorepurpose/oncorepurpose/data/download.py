"""Download the PrimeKG knowledge graph from Harvard Dataverse."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from oncorepurpose.config import PRIMEKG_KG_CSV, PRIMEKG_KG_URL


def download_primekg(dest: Path = PRIMEKG_KG_CSV, force: bool = False) -> Path:
    """Download PrimeKG kg.csv if not already present."""
    dest = Path(dest)
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"PrimeKG already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PrimeKG from {PRIMEKG_KG_URL} -> {dest}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(PRIMEKG_KG_URL, dest, _progress)
    sys.stdout.write("\n")
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


if __name__ == "__main__":
    download_primekg()
