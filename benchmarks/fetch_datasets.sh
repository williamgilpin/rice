#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="$(pwd)/benchmark_datasets"
REPO="williamgilpin/grn"
REPO_URL="https://huggingface.co/datasets/${REPO}"

# Only proceed if the target folder doesn't already exist
if [ -d "${DEST_DIR}" ]; then
    echo "Directory ${DEST_DIR} already exists; skipping download."
    exit 0
fi

# Create and enter the destination directory
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

## Download the dataset via Hugging Face CLI (requires huggingface-cli to be installed)
huggingface-cli download "${REPO}" \
  --repo-type dataset \
  --local-dir "${DEST_DIR}" \
  --local-dir-use-symlinks False \
  --resume-download