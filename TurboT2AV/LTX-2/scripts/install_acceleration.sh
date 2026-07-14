#!/usr/bin/env bash

set -euo pipefail

SAGEATTENTION_COMMIT="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SPARGEATTN_COMMIT="ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/turbot2av-acceleration.XXXXXX")"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

export MAX_JOBS="${MAX_JOBS:-4}"
export EXT_PARALLEL="${EXT_PARALLEL:-1}"

clone_commit() {
    local repository="$1"
    local commit="$2"
    local destination="$3"

    git init --quiet "${destination}"
    git -C "${destination}" remote add origin "${repository}"
    git -C "${destination}" fetch --quiet --depth 1 origin "${commit}"
    git -C "${destination}" checkout --quiet --detach FETCH_HEAD
}

nvidia_include_flags() {
    python - <<'PY'
import site
from pathlib import Path

include_dirs = set()
for site_package in site.getsitepackages():
    nvidia_dir = Path(site_package) / "nvidia"
    if not nvidia_dir.is_dir():
        continue
    include_dirs.update(path for path in nvidia_dir.glob("*/include") if path.is_dir())
print(" ".join(f"-I{path}" for path in sorted(include_dirs)))
PY
}

install_sageattention() {
    local package="${SAGEATTENTION_PACKAGE:-}"
    if [[ -z "${package}" ]]; then
        package="${BUILD_ROOT}/SageAttention"
        clone_commit "https://github.com/thu-ml/SageAttention.git" "${SAGEATTENTION_COMMIT}" "${package}"
        git -C "${package}" apply --unidiff-zero "${PROJECT_DIR}/patches/sageattention-sm90-build.patch"
    fi

    local include_flags
    include_flags="$(nvidia_include_flags)"
    CXX_APPEND_FLAGS="${CXX_APPEND_FLAGS:-} ${include_flags}" \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} ${include_flags}" \
        python -m pip install --no-build-isolation "${package}"
}

install_spargeattn() {
    local package="${SPARGEATTN_PACKAGE:-}"
    # SpargeAttn builds hundreds of independent CUDA instantiations, so give
    # this extension a scoped job count without changing the other builders.
    local spargeattn_max_jobs="${SPARGEATTN_MAX_JOBS:-8}"
    if [[ -z "${package}" ]]; then
        package="${BUILD_ROOT}/SpargeAttn"
        clone_commit "https://github.com/thu-ml/SpargeAttn.git" "${SPARGEATTN_COMMIT}" "${package}"
        git -C "${package}" apply --unidiff-zero "${PROJECT_DIR}/patches/spargeattn-h20-build.patch"
    fi

    MAX_JOBS="${spargeattn_max_jobs}" \
        python -m pip install --no-build-isolation "${package}"
}

install_tilelang() {
    python -m pip install apache-tvm-ffi==0.1.10 tilelang==0.1.11
    python -c "import tilelang"
}

case "${1:-all}" in
    sageattention)
        install_sageattention
        ;;
    spargeattn)
        install_spargeattn
        ;;
    tilelang)
        install_tilelang
        ;;
    all)
        install_sageattention
        install_spargeattn
        install_tilelang
        ;;
    *)
        echo "usage: $0 [all|sageattention|spargeattn|tilelang]" >&2
        exit 2
        ;;
esac
