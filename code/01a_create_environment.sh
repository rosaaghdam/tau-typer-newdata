#!/bin/bash
# Step 1a – create the project environment in ./env from environment.yml.
# Run from anywhere:  bash code/01a_create_environment.sh   (takes ~10 minutes)
# First run: 2026-10-15 (conda 24.9.2, miniforge, macOS arm64).

set -e                          # stop at the first error
cd "$(dirname "$0")/.."         # go to the project root (one level above code/)

# Never overwrite an existing environment by accident.
if [ -d env ]; then
    echo "./env already exists. To rebuild it, delete the env/ folder first."
    exit 1
fi

# Build the environment inside the project folder (-p = path, instead of a name).
conda env create -f environment.yml -p ./env

# Check: pySuStaIn imports, and it is the pinned commit.
./env/bin/python -c "from pySuStaIn import ZscoreSustain; print('pySuStaIn OK')"
./env/bin/pip freeze | grep -i sustain
