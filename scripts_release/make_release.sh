#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/dist"
DATE="$(date +%Y%m%d)"
FULL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts_release/make_release.sh [--full] [--out <path.tar.gz>]

By default, creates a lightweight code release tarball and excludes large
datasets/results artifacts. Use --full to include everything.
EOF
}

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FULL=1
      shift
      ;;
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUT_DIR}"
if [[ -z "${OUT}" ]]; then
  OUT="${OUT_DIR}/pharmaprompt-code-${DATE}.tar.gz"
fi

EXCLUDES=(
  ".vscode"
  "__pycache__"
  "**/__pycache__"
  "wandb"
  "tmp"
  "dist"
)

if [[ "${FULL}" -eq 0 ]]; then
  EXCLUDES+=(
    "checkpoints"
    "datasets"
    "pretrained"
    "results"
    "results_*"
    "results_docking"
    "results_screening"
    "results_significance"
    "save"
    "EGFR_ligand_folders"
    "JAK1_ligand_folders"
    "*.zip"
  )
fi

TAR_ARGS=()
for ex in "${EXCLUDES[@]}"; do
  TAR_ARGS+=( "--exclude=${ex}" )
done

TMP_OUT="${OUT}.tmp"
rm -f "${TMP_OUT}"
tar -czf "${TMP_OUT}" "${TAR_ARGS[@]}" -C "${ROOT}" .
mv -f "${TMP_OUT}" "${OUT}"
echo "Wrote release tarball: ${OUT}"

