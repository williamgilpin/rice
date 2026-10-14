#!/usr/bin/env bash
set -euo pipefail

## Download the datasets created for this study (TWIST, Smoketest)
## Requires the Hugging Face CLI: pip install -U "huggingface_hub[cli]"

DEST_DIR="$(pwd)/benchmark_datasets"
REPO="williamgilpin/grn"

mkdir -p "${DEST_DIR}"

hf download "${REPO}" \
  --repo-type dataset \
  --local-dir "${DEST_DIR}" \
  --include "gene_expression/*" \
  --include "smoketest/*" 
