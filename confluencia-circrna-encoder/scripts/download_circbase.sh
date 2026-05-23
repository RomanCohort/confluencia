#!/bin/bash
# download_circbase.sh — Download circBase sequence data

set -e

echo "=============================================="
echo "Downloading circBase circRNA Sequences"
echo "=============================================="

# Create directory
mkdir -p data/circrna

# circBase download URLs
URL1="http://www.circbase.org/download/human_hg19_circRNA.fa.gz"
URL2="http://circbase.org/download/human_hg19_circRNA.fa.gz"
URL3="https://ftp.circbase.org/download/human_hg19_circRNA.fa.gz"

# Try multiple sources
OUTPUT="data/circrna/circbase_seqs.fa.gz"

echo ""
echo "[1] Trying circBase official source..."

if curl -L --connect-timeout 10 -o "$OUTPUT" "$URL1" 2>/dev/null; then
    if [ -f "$OUTPUT" ] && [ $(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT") -gt 1000000 ]; then
        echo "    ✓ Downloaded from $URL1"
    else
        rm -f "$OUTPUT"
    fi
fi

if [ ! -f "$OUTPUT" ]; then
    echo "[2] Trying alternate source..."
    if curl -L --connect-timeout 10 -o "$OUTPUT" "$URL2" 2>/dev/null; then
        if [ -f "$OUTPUT" ] && [ $(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT") -gt 1000000 ]; then
            echo "    ✓ Downloaded from $URL2"
        else
            rm -f "$OUTPUT"
        fi
    fi
fi

# If still not available, create from known sequences
if [ ! -f "$OUTPUT" ]; then
    echo "[3] Official download failed, using known circBase ID list..."
    echo "    Fetching circBase ID list..."

    # Download circBase ID list
    ID_URL="http://www.circbase.org/download/hsa_circRNA_id.txt"
    curl -L -o data/circrna/circbase_ids.txt "$ID_URL" 2>/dev/null || true

    if [ -f "data/circrna/circbase_ids.txt" ]; then
        echo "    ✓ Got circBase ID list"
        echo "    Use circBase API to fetch sequences (requires additional script)"
    else
        echo "    ✗ Could not download circBase data"
        echo "    Please manually download from: http://www.circbase.org/"
    fi
fi

# Verify
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "[4] Verifying download..."
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    SEQ_COUNT=$(zcat "$OUTPUT" | grep "^>" | wc -l)

    echo "    File: $OUTPUT"
    echo "    Size: $SIZE"
    echo "    Sequences: $SEQ_COUNT"

    if [ $SEQ_COUNT -gt 100000 ]; then
        echo "    ✓ Download successful!"
    else
        echo "    ⚠ Sequence count lower than expected"
    fi
fi

echo ""
echo "=============================================="
echo "Download Complete"
echo "=============================================="

# Show first few sequences
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "Sample sequences (first 3):"
    zcat "$OUTPUT" | head -6
fi