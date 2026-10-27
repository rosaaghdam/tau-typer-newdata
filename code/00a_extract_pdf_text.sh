#!/bin/bash
# Step 0a – turn every PDF in papers/ into a .txt file in papers/text/ for reading.
# Run from anywhere:  bash code/00a_extract_pdf_text.sh   (macOS only)
# First run: 2026-10-15.

set -e                          # stop at the first error
cd "$(dirname "$0")/.."         # go to the project root

mkdir -p papers/text            # output folder (created if missing)

for pdf in papers/*.pdf; do
    name=$(basename "$pdf" .pdf)            # e.g. "tau-subtyping_clean"
    out="papers/text/$name.txt"
    pages=$(osascript -l JavaScript code/helpers/pdf2txt.js "$PWD/$pdf" "$PWD/$out")
    echo "$pdf -> $out ($pages)"
done
