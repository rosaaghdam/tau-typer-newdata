#!/bin/bash
# Step 0b – check that Github/pySuStaIn-master is identical to the official pySuStaIn
# at the commit we pinned in environment.yml. Needs internet. Changes nothing locally.
# Run from anywhere:  bash code/00b_compare_pysustain_with_github.sh
# First run: 2026-10-15 -> all files identical.

set -e
cd "$(dirname "$0")/.."

COMMIT=708fa22d89c9692dbeafcc2c31a1c8460ced5640   # same commit as environment.yml
URL=https://raw.githubusercontent.com/ucl-pond/pySuStaIn/$COMMIT/pySuStaIn
TMP=$(mktemp -d)                                  # throw-away folder for downloads

for file in Github/pySuStaIn-master/pySuStaIn/*.py; do
    name=$(basename "$file")
    curl -sfL -o "$TMP/$name" "$URL/$name"        # download the official version
    if cmp -s "$file" "$TMP/$name"; then echo "identical: $name"; else echo "DIFFERENT: $name"; fi
done
