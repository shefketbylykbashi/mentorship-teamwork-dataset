#!/usr/bin/env bash
# Builds the 10 campaign solvers into campaign/bin with the agreed flags:
#   g++ -O2 -std=c++20   (single thread: no -fopenmp, no -pthread)
# Writes campaign/bin/build_info.json (compiler, flags, source/binary hashes).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/campaign/bin"
CXX="${CXX:-g++}"
FLAGS=(-O2 -std=c++20)
mkdir -p "$BIN"

declare -A SRC=(
  [refactored]="solvers/refactored_Nanored"
  [evolved]="solvers/refactored_Nanored_evolved"
)

json_rows=()
for set in refactored evolved; do
  for s in b c d e f; do
    src="${SRC[$set]}/$s.cpp"
    out="$BIN/${set}_$s"
    echo "building ${set}_$s  <- $src"
    "$CXX" "${FLAGS[@]}" -o "$out" "$REPO/$src"
    [[ -f "$out.exe" ]] && out="$out.exe"  # MinGW (local testing)
    json_rows+=("\"${set}_$s\": {\"source\": \"$src\", \"source_sha256\": \"$(sha256sum "$REPO/$src" | cut -d' ' -f1)\", \"binary_sha256\": \"$(sha256sum "$out" | cut -d' ' -f1)\"}")
  done
done

# Sanity: the solvers must not be threaded.
if grep -nE 'std::thread|std::async|#pragma omp|<thread>|<future>|<omp.h>' "$REPO"/solvers/refactored_Nanored*/*.cpp "$REPO"/solvers/mtp_runtime.hpp; then
  echo "ERROR: threading construct found in solver sources" >&2
  exit 1
fi

hdr_sha=$(sha256sum "$REPO/solvers/mtp_runtime.hpp" | cut -d' ' -f1)
{
  echo "{"
  echo "  \"built_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"host\": \"$(hostname)\","
  echo "  \"compiler\": \"$("$CXX" --version | head -1 | sed 's/"/\\"/g')\","
  echo "  \"flags\": \"${FLAGS[*]}\","
  echo "  \"mtp_runtime_hpp_sha256\": \"$hdr_sha\","
  echo "  \"solvers\": {"
  for i in "${!json_rows[@]}"; do
    sep=","; [[ $i -eq $((${#json_rows[@]} - 1)) ]] && sep=""
    echo "    ${json_rows[$i]}$sep"
  done
  echo "  }"
  echo "}"
} > "$BIN/build_info.json"
echo "OK: 10 solvers in $BIN"
