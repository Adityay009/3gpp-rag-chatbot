"""
Downloads the official ETSI-hosted PDF of 3GPP TS 23.501 at build/setup time.

We fetch this at build time rather than committing the PDF to the repo, since the
document carries an ETSI copyright notice restricting redistribution. Downloading
directly from ETSI's own public deliverable server for local indexing use is the
appropriate way to source it.
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = "https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/18.05.00_60/ts_123501v180500p.pdf"
DEST = Path(__file__).resolve().parent.parent / "data" / "ts_123501.pdf"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {DEST}")
    urlretrieve(url, DEST)
    print(f"Done. {DEST.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
