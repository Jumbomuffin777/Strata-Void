#!/usr/bin/env bash
# build_hotcold.sh - build the llama.cpp that serves hot/cold GGUFs (tools/moe_placement/gguf_hotcold.py):
# clone llama.cpp at the pinned commit, apply patches/llama.cpp-moe-hot-cold-experts.patch and
# patches/llama.cpp-moe-dynamic-residency.patch, build llama-server (+ ggml-rpc-server with RPC) and the tools
# routing-dump and op-profile against it. See docs/HOTCOLD_LLAMACPP.md.
#
#   tools/llamacpp/build_hotcold.sh                      SYCL + RPC (Intel GPUs; needs oneAPI)
#   tools/llamacpp/build_hotcold.sh --cuda               SYCL + CUDA + RPC
#   tools/llamacpp/build_hotcold.sh --cpu                CPU + RPC only (no GPU toolkit; for tests and RPC hosts)
#
# Options:
#   --dir DIR          where the llama.cpp checkout goes (default: third_party/llama.cpp-hotcold)
#   --source URL|DIR   clone from here instead of the pinned repository (an existing clone works offline)
#   --cuda             add the CUDA backend;  --cuda-arch LIST  CMAKE_CUDA_ARCHITECTURES (default: native)
#   --cpu              no SYCL (CPU + RPC);   --no-rpc  leave the RPC backend out
#   --jobs N           parallel build jobs (default: all cores)
#   --check            clone and check that the patch applies, build nothing
#   --no-routing-dump  do not build tools/llamacpp/routing-dump and op-profile
# Extra CMake options: CMAKE_EXTRA="-DGGML_SYCL_DNN=OFF" tools/llamacpp/build_hotcold.sh
set -euo pipefail

REPO_URL="https://github.com/unslothai/llama.cpp.git"   # branch glm5next/upstream: GLM-5.3-Flash + DeepSeek V4 Flash
COMMIT="86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9"        # the commit the patch is made for

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# applied in this order: the hot/cold expert split, then load-time splitting + adaptive residency + glm5-next names
PATCHES=("$ROOT/patches/llama.cpp-moe-hot-cold-experts.patch" "$ROOT/patches/llama.cpp-moe-dynamic-residency.patch")
HEADER="$ROOT/patches/llama-moe-residency.h"          # the residency policy, used by the second patch
DIR="$ROOT/third_party/llama.cpp-hotcold"
SOURCE="$REPO_URL"
SYCL=1 CUDA=0 RPC=1 CHECK=0 DUMP=1 CUDA_ARCH="native"
JOBS="$(nproc 2>/dev/null || echo 4)"

die() { echo "build_hotcold: $*" >&2; exit 1; }
say() { echo "== $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --cuda) CUDA=1; shift ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --cpu) SYCL=0; shift ;;
    --no-rpc) RPC=0; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --no-routing-dump) DUMP=0; shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

command -v git >/dev/null || die "git is not installed"
for P in "${PATCHES[@]}" "$HEADER"; do [ -f "$P" ] || die "missing: $P"; done

# --- 1. the pinned source --------------------------------------------------------------------------------------------
if [ ! -d "$DIR/.git" ]; then
  say "cloning llama.cpp $COMMIT from $SOURCE into $DIR"
  mkdir -p "$DIR"
  git -C "$DIR" init -q
  git -C "$DIR" remote add origin "$SOURCE"
  # a full SHA can be fetched directly from GitHub (and from a local clone that has it)
  git -C "$DIR" fetch -q --depth 1 origin "$COMMIT" || die "could not fetch $COMMIT from $SOURCE"
  git -C "$DIR" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
HEAD="$(git -C "$DIR" rev-parse HEAD)"
[ "$HEAD" = "$COMMIT" ] || die "$DIR is at $HEAD, not the pinned $COMMIT: use another --dir, or remove it"

# --- 2. the patches -------------------------------------------------------------------------------------------------
# apply_series DIR: applies every patch not yet applied, in order; stops with the reason if one does not apply
apply_series() {
  local d="$1" p err
  for p in "${PATCHES[@]}"; do
    if git -C "$d" apply --whitespace=nowarn --reverse --check "$p" 2>/dev/null; then
      say "already applied: $(basename "$p")"
    elif err="$(git -C "$d" apply --whitespace=nowarn --check "$p" 2>&1)"; then
      git -C "$d" apply --whitespace=nowarn "$p"
      say "applied: $(basename "$p")"
    else
      echo "$err" >&2
      die "$(basename "$p") does not apply to $d. The patches are made for $COMMIT exactly; if the checkout has local
  edits, start from a clean one (git -C '$d' checkout -- . ), or rebase the patches (docs/MOE_PLACEMENT.md)"
    fi
  done
  cp "$HEADER" "$d/src/"
}
if [ "$CHECK" = 1 ]; then                       # on a throwaway copy: the checkout itself is left as it is
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone -q --shared "$DIR" "$TMP/l" && git -C "$TMP/l" -c advice.detachedHead=false checkout -q "$COMMIT"
  apply_series "$TMP/l"
  say "--check: the patches apply cleanly to $COMMIT; nothing changed, nothing built"
  exit 0
fi
apply_series "$DIR"

# --- 3. build --------------------------------------------------------------------------------------------------------
command -v cmake >/dev/null || die "cmake is not installed"
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)
OPTS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF)
BUILD="$DIR/build"
if [ "$SYCL" = 1 ]; then
  if ! command -v icpx >/dev/null; then
    # shellcheck disable=SC1091
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
      set +u                                   # setvars.sh reads unset variables: with -u the script would end
      source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
      set -u
    fi
  fi
  command -v icpx >/dev/null || die "SYCL needs the oneAPI compilers (icx/icpx): install oneAPI and run
  'source /opt/intel/oneapi/setvars.sh', or build with --cpu"
  OPTS+=(-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL)
fi
if [ "$CUDA" = 1 ]; then
  command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || die "--cuda needs the CUDA toolkit (nvcc)"
  OPTS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
  command -v nvcc >/dev/null || OPTS+=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
fi
[ "$RPC" = 1 ] && OPTS+=(-DGGML_RPC=ON)
# shellcheck disable=SC2206
[ -n "${CMAKE_EXTRA:-}" ] && OPTS+=($CMAKE_EXTRA)
TARGETS=(llama-server)
[ "$RPC" = 1 ] && TARGETS+=(ggml-rpc-server)   # the RPC worker binary (tools/rpc), for remote hosts

say "configuring: ${OPTS[*]}"
cmake -S "$DIR" -B "$BUILD" "${GEN[@]}" "${OPTS[@]}"
say "building ${TARGETS[*]} with $JOBS jobs"
cmake --build "$BUILD" --target "${TARGETS[@]}" -j "$JOBS"

# --- 4. routing-dump (the trace the planner reads) -------------------------------------------------------------------
if [ "$DUMP" = 1 ]; then
  CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$BUILD/CMakeCache.txt")"
  CXX="${CXX:-c++}"
  say "building routing-dump and op-profile with $CXX"
  cmake --build "$BUILD" --target llama-common -j "$JOBS" >/dev/null 2>&1 || cmake --build "$BUILD" --target common -j "$JOBS"
  COMMON_LIB=llama-common
  ls "$BUILD"/bin/libllama-common.* >/dev/null 2>&1 || COMMON_LIB=common
  for tool in routing-dump op-profile; do
    "$CXX" -O2 -std=c++17 -I"$DIR/include" -I"$DIR/common" -I"$DIR/ggml/include" -I"$DIR/vendor" \
      "$ROOT/tools/llamacpp/$tool.cpp" -o "$BUILD/bin/$tool" \
      -L"$BUILD/bin" -l"$COMMON_LIB" -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin" \
      || echo "build_hotcold: $tool did not build (llama-server did); see tools/llamacpp/$tool.cpp" >&2
  done
fi

say "done: $BUILD/bin/llama-server"
"$BUILD/bin/llama-server" --version 2>&1 | tail -n 2 || true
cat <<EOF
Run hot/cold GGUFs with --no-op-offload (op offload reads expert ids before the hot/cold split computes them).
Serve one through Strata Void:  python -m serve.server --engine llamacpp --config examples/llamacpp-hotcold/server.json
EOF
