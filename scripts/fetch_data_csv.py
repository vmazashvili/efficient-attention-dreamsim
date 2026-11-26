"""Extract NIGHTS data.csv from the remote nights.zip with HTTP range requests (SPEC §6).

The chunked download has no data.csv; this reads ~2 MB instead of the 61.5 GB archive.

    python scripts/fetch_data_csv.py --out data/data.csv
"""

import argparse
import hashlib
import sys
from pathlib import Path

from remotezip import RemoteZip

URL = "https://data.csail.mit.edu/nights/nights.zip"
MEMBER = "nights/data.csv"
SHA256 = "5c8d23041917b34758c4308d7310a20eaaa34ab4ea32e134074950204f840c0a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=Path("data/data.csv"), type=Path)
    args = ap.parse_args()

    with RemoteZip(URL) as z:
        data = z.read(MEMBER)
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        sys.exit(f"sha256 mismatch: got {digest}, expected {SHA256}. Not written.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"{args.out}: {len(data)} bytes, sha256 ok")


if __name__ == "__main__":
    main()
