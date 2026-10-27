#!/bin/bash
# Check that the pySuStaIn INSTALLED in ./env (what the app and src/ import) is the unchanged
# original: installed from the official repository at the pinned commit, every file identical.
# Run from anywhere:  bash code/00c_check_installed_pysustain.sh   (needs internet)
# First run: 2026-10-17 -> all 7 files identical, installed from commit 708fa22.

set -e
cd "$(dirname "$0")/.."                               # project root

COMMIT=708fa22d89c9692dbeafcc2c31a1c8460ced5640       # same commit as environment.yml
URL=https://raw.githubusercontent.com/ucl-pond/pySuStaIn/$COMMIT/pySuStaIn
INSTALLED=$(./env/bin/python -c "import pySuStaIn, os; print(os.path.dirname(pySuStaIn.__file__))")
echo "installed package folder: $INSTALLED"
./env/bin/python -c "import json, importlib.metadata as m; info = json.loads(m.distribution('pySuStaIn').read_text('direct_url.json')); print('installed from:', info['url'], 'commit', info['vcs_info']['commit_id'])"

TMP=$(mktemp -d)                                      # throw-away folder for the official files
for file in "$INSTALLED"/*.py; do
    name=$(basename "$file")
    curl -sfL -o "$TMP/$name" "$URL/$name"
    if cmp -s "$file" "$TMP/$name"; then echo "identical: $name"; else echo "DIFFERENT: $name"; fi
done
