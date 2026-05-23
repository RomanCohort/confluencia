"""
download_circbase.py — Download circBase data with fallbacks.

circBase official: http://www.circbase.org/
"""

import os
import sys
import gzip
import time
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

# Try requests if available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "circrna"
OUTPUT_FILE = OUTPUT_DIR / "circbase_seqs.fa.gz"

# Download sources (try multiple)
DOWNLOAD_URLS = [
    # circBase official (may be slow/unavailable)
    "http://www.circbase.org/download/human_hg19_circRNA.fa.gz",
    "http://circbase.org/download/human_hg19_circRNA.fa.gz",
    # Alternative mirrors (if available)
    "https://data.cyberpandas.org/circrna/circbase_seqs.fa.gz",
]


def download_with_requests(url: str, output: Path, timeout: int = 60) -> bool:
    """Download using requests library."""
    if not HAS_REQUESTS:
        return False

    try:
        print(f"  Trying: {url}")
        response = requests.get(url, timeout=timeout, stream=True)

        if response.status_code != 200:
            print(f"    Status: {response.status_code}")
            return False

        total_size = int(response.headers.get('content-length', 0))
        print(f"    Size: {total_size / 1024 / 1024:.1f} MB")

        with open(output, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) < 8192:
                        pct = downloaded / total_size * 100
                        print(f"    Progress: {pct:.1f}%")

        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_with_urllib(url: str, output: Path, timeout: int = 60) -> bool:
    """Download using urllib."""
    try:
        print(f"  Trying: {url}")

        def report_hook(count, block_size, total_size):
            if total_size > 0:
                pct = count * block_size / total_size * 100
                print(f"    Progress: {pct:.1f}%")

        urlretrieve(url, output, reporthook=report_hook)
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def verify_file(filepath: Path) -> dict:
    """Verify downloaded file."""
    if not filepath.exists():
        return {"valid": False, "error": "File not found"}

    size = filepath.stat().st_size

    if size < 1000:
        return {"valid": False, "error": "File too small"}

    # Count sequences
    try:
        with gzip.open(filepath, 'rt') as f:
            seq_count = 0
            for line in f:
                if line.startswith('>'):
                    seq_count += 1
    except Exception as e:
        return {"valid": False, "error": f"Cannot read: {e}"}

    return {
        "valid": seq_count > 100000,
        "size_mb": size / 1024 / 1024,
        "seq_count": seq_count,
    }


def main():
    print("=" * 60)
    print("Downloading circBase Data")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try each URL
    for url in DOWNLOAD_URLS:
        print(f"\n[Attempt] {url}")

        # Remove partial download
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

        # Try download
        success = False
        if HAS_REQUESTS:
            success = download_with_requests(url, OUTPUT_FILE)
        if not success:
            success = download_with_urllib(url, OUTPUT_FILE)

        if success:
            # Verify
            print(f"\n[Verify] Checking download...")
            result = verify_file(OUTPUT_FILE)

            if result.get("valid"):
                print(f"  ✓ Size: {result['size_mb']:.1f} MB")
                print(f"  ✓ Sequences: {result['seq_count']}")
                print(f"\n✓ Download successful!")
                print(f"  Output: {OUTPUT_FILE}")
                return
            else:
                print(f"  ✗ Invalid: {result.get('error')}")
                OUTPUT_FILE.unlink()

    # All URLs failed
    print("\n" + "=" * 60)
    print("⚠ Download failed from all sources")
    print("=" * 60)
    print("\nManual download options:")
    print("  1. Visit http://www.circbase.org/")
    print("  2. Download 'human_hg19_circRNA.fa.gz'")
    print("  3. Place in: data/circrna/circbase_seqs.fa.gz")
    print("\nOr use local data if available:")

    # Check if local file exists (from previous session)
    local_path = Path("D:/IGEM集成方案/data/circrna/circbase_seqs.fa.gz")
    if local_path.exists():
        result = verify_file(local_path)
        if result.get("valid"):
            print(f"\n  ✓ Local file found: {local_path}")
            print(f"    Size: {result['size_mb']:.1f} MB")
            print(f"    Sequences: {result['seq_count']}")
            print("  Upload this to AutoDL manually")


if __name__ == "__main__":
    main()