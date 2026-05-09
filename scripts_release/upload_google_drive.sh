#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date +%Y%m%d)"
REMOTE="gdrive"
DRIVE_DIR="PharmAgent-public-release/${DATE}"
MODE="archives"
DIST_DIR="${ROOT}/dist"
SHARED_ARCHIVE="${DIST_DIR}/pharmagent_shared_models.zip"
DEN1_ARCHIVE="${DIST_DIR}/pharmagent_chembert_den1.zip"
TASK_ARCHIVE="${DIST_DIR}/pharmagent_example_task_checkpoints.zip"
PRINT_ONLY=0
INCLUDE_DEN1=0
SKIP_SHARED=0
SKIP_DEFAULT_TASKS=0

DEFAULT_TASKS=(
  EGFR
  HPK1_IC50
  FGFR1_IC50
  JAK1
)

CURATED_PUBLIC_TASKS=(
  EGFR
  FGFR1_IC50
  HPK1_IC50
  JAK1
  bbbp
  clintox
  sider
  tox21
  toxcast
)

TASKS=("${DEFAULT_TASKS[@]}")
EXTRA_TASKS=()
TASK_PATHS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts_release/upload_google_drive.sh [options]

Options:
  --remote <name>       rclone remote name. Default: gdrive
  --drive-dir <path>    Google Drive target folder. Default: PharmAgent-public-release/<date>
  --mode <value>        archives | files. Default: archives
  --task <name>         Add a public task checkpoint. Repeatable. Use all-public for the curated release set.
  --include-den1        Upload checkpoints/DEN1/pytorch_model.bin for chembert-based paths
  --skip-shared         Skip pretrained/base and BiomedBERT uploads
  --skip-default-tasks  Skip the default EGFR, HPK1_IC50, and bace task checkpoints
  --print-only          Print commands without executing upload
  -h, --help            Show this help

Examples:
  bash scripts_release/upload_google_drive.sh --print-only
  bash scripts_release/upload_google_drive.sh --remote mydrive --drive-dir PharmAgent-public-release/v1
  bash scripts_release/upload_google_drive.sh --mode files --remote mydrive
  bash scripts_release/upload_google_drive.sh --include-den1 --task all-public
  bash scripts_release/upload_google_drive.sh --include-den1 --skip-shared --skip-default-tasks --task FGFR1_IC50 --task JAK1

Notes:
  - This script uses rclone.
  - Run 'rclone config' first to authorize Google Drive.
  - In 'archives' mode, two zip files are created under dist/ and uploaded.
  - In 'files' mode, the minimal release files are uploaded with original relative paths preserved.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --drive-dir)
      DRIVE_DIR="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --task)
      EXTRA_TASKS+=("${2:-}")
      shift 2
      ;;
    --include-den1)
      INCLUDE_DEN1=1
      shift
      ;;
    --skip-shared)
      SKIP_SHARED=1
      shift
      ;;
    --skip-default-tasks)
      SKIP_DEFAULT_TASKS=1
      shift
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
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

if [[ "${MODE}" != "archives" && "${MODE}" != "files" ]]; then
  echo "Unsupported mode: ${MODE}" >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

add_unique_task() {
  local candidate="$1"
  local existing
  for existing in "${TASKS[@]}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  TASKS+=("${candidate}")
}

normalize_tasks() {
  local requested
  local curated

  for requested in "${EXTRA_TASKS[@]}"; do
    if [[ -z "${requested}" ]]; then
      echo "Empty --task value is not allowed" >&2
      exit 1
    fi

    if [[ "${requested}" == "all-public" ]]; then
      for curated in "${CURATED_PUBLIC_TASKS[@]}"; do
        add_unique_task "${curated}"
      done
      continue
    fi

    add_unique_task "${requested}"
  done
}

resolve_task_checkpoint() {
  local task="$1"
  local task_dir="${ROOT}/save/${task}"
  local matches=()

  if [[ ! -d "${task_dir}" ]]; then
    echo "Task directory not found: save/${task}" >&2
    exit 1
  fi

  mapfile -t matches < <(find "${task_dir}" -type f -name best_model.pth | sort)

  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No checkpoint found under save/${task}" >&2
    exit 1
  fi

  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "Multiple checkpoints found under save/${task}; using ${matches[0]#${ROOT}/}" >&2
  fi

  printf '%s\n' "${matches[0]#${ROOT}/}"
}

build_task_paths() {
  local task
  TASK_PATHS=()
  for task in "${TASKS[@]}"; do
    TASK_PATHS+=("$(resolve_task_checkpoint "${task}")")
  done
}

print_zip_command() {
  local archive="$1"
  shift
  local files=("$@")
  local idx

  printf 'zip -r "%s" \\\n' "${archive}"
  for idx in "${!files[@]}"; do
    if [[ "${idx}" -lt $((${#files[@]} - 1)) ]]; then
      printf '  %s \\\n' "${files[idx]}"
    else
      printf '  %s\n' "${files[idx]}"
    fi
  done
}

print_raw_copy_commands() {
  local path
  local target_base="${REMOTE}:${DRIVE_DIR}"

  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    printf 'rclone copyto pretrained/base/base.pth "%s/pretrained/base/base.pth" --progress\n' "${target_base}"
    printf 'rclone copyto pretrained/BiomedBERT/config.json "%s/pretrained/BiomedBERT/config.json" --progress\n' "${target_base}"
    printf 'rclone copyto pretrained/BiomedBERT/vocab.txt "%s/pretrained/BiomedBERT/vocab.txt" --progress\n' "${target_base}"
    printf 'rclone copyto pretrained/BiomedBERT/pytorch_model.bin "%s/pretrained/BiomedBERT/pytorch_model.bin" --progress\n' "${target_base}"
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    printf 'rclone copyto checkpoints/DEN1/pytorch_model.bin "%s/checkpoints/DEN1/pytorch_model.bin" --progress\n' "${target_base}"
  fi

  for path in "${TASK_PATHS[@]}"; do
    printf 'rclone copyto %s "%s/%s" --progress\n' "${path}" "${target_base}" "${path}"
  done
}

if [[ "${SKIP_DEFAULT_TASKS}" -eq 1 ]]; then
  TASKS=()
fi

normalize_tasks
build_task_paths

print_prereqs() {
  cat <<EOF
# 1) Install and configure rclone
sudo apt-get update
sudo apt-get install -y rclone
rclone config

# 2) Check remote access
rclone listremotes
rclone lsd ${REMOTE}:
EOF
}

print_archive_commands() {
  cat <<EOF
mkdir -p "${DIST_DIR}"

cd "${ROOT}"
EOF

  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    print_zip_command \
      "${SHARED_ARCHIVE}" \
      pretrained/base/base.pth \
      pretrained/BiomedBERT/config.json \
      pretrained/BiomedBERT/vocab.txt \
      pretrained/BiomedBERT/pytorch_model.bin
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    cat <<EOF

cd "${ROOT}"
EOF
    print_zip_command \
      "${DEN1_ARCHIVE}" \
      checkpoints/DEN1/pytorch_model.bin
  fi

  cat <<EOF

cd "${ROOT}"
EOF

  print_zip_command "${TASK_ARCHIVE}" "${TASK_PATHS[@]}"

  cat <<EOF

rclone mkdir "${REMOTE}:${DRIVE_DIR}"
EOF

  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    cat <<EOF
rclone copyto "${SHARED_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${SHARED_ARCHIVE}")" --progress
EOF
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    cat <<EOF
rclone copyto "${DEN1_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${DEN1_ARCHIVE}")" --progress
EOF
  fi

  cat <<EOF
rclone copyto "${TASK_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${TASK_ARCHIVE}")" --progress
rclone ls "${REMOTE}:${DRIVE_DIR}"
EOF
}

print_file_commands() {
  cat <<EOF
cd "${ROOT}"
rclone mkdir "${REMOTE}:${DRIVE_DIR}"

EOF

  print_raw_copy_commands

  cat <<EOF

rclone lsf -R "${REMOTE}:${DRIVE_DIR}"
EOF
}

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  print_prereqs
  echo
  if [[ "${MODE}" == "archives" ]]; then
    print_archive_commands
  else
    print_file_commands
  fi
  exit 0
fi

require_cmd rclone

if [[ "${MODE}" == "archives" ]]; then
  require_cmd zip
  mkdir -p "${DIST_DIR}"

  cd "${ROOT}"
  rm -f "${SHARED_ARCHIVE}" "${DEN1_ARCHIVE}" "${TASK_ARCHIVE}"
  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    zip -r "${SHARED_ARCHIVE}" \
      pretrained/base/base.pth \
      pretrained/BiomedBERT/config.json \
      pretrained/BiomedBERT/vocab.txt \
      pretrained/BiomedBERT/pytorch_model.bin
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    cd "${ROOT}"
    zip -r "${DEN1_ARCHIVE}" \
      checkpoints/DEN1/pytorch_model.bin
  fi

  cd "${ROOT}"
  zip -r "${TASK_ARCHIVE}" "${TASK_PATHS[@]}"

  rclone mkdir "${REMOTE}:${DRIVE_DIR}"
  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    rclone copyto "${SHARED_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${SHARED_ARCHIVE}")" --progress
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    rclone copyto "${DEN1_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${DEN1_ARCHIVE}")" --progress
  fi

  rclone copyto "${TASK_ARCHIVE}" "${REMOTE}:${DRIVE_DIR}/$(basename "${TASK_ARCHIVE}")" --progress
  rclone ls "${REMOTE}:${DRIVE_DIR}"
else
  cd "${ROOT}"
  rclone mkdir "${REMOTE}:${DRIVE_DIR}"

  if [[ "${SKIP_SHARED}" -eq 0 ]]; then
    rclone copyto pretrained/base/base.pth "${REMOTE}:${DRIVE_DIR}/pretrained/base/base.pth" --progress
    rclone copyto pretrained/BiomedBERT/config.json "${REMOTE}:${DRIVE_DIR}/pretrained/BiomedBERT/config.json" --progress
    rclone copyto pretrained/BiomedBERT/vocab.txt "${REMOTE}:${DRIVE_DIR}/pretrained/BiomedBERT/vocab.txt" --progress
    rclone copyto pretrained/BiomedBERT/pytorch_model.bin "${REMOTE}:${DRIVE_DIR}/pretrained/BiomedBERT/pytorch_model.bin" --progress
  fi

  if [[ "${INCLUDE_DEN1}" -eq 1 ]]; then
    rclone copyto checkpoints/DEN1/pytorch_model.bin "${REMOTE}:${DRIVE_DIR}/checkpoints/DEN1/pytorch_model.bin" --progress
  fi

  for path in "${TASK_PATHS[@]}"; do
    rclone copyto "${path}" "${REMOTE}:${DRIVE_DIR}/${path}" --progress
  done

  rclone lsf -R "${REMOTE}:${DRIVE_DIR}"
fi