#!/usr/bin/env bash
# setup_lean.sh -- pin the Lean+Mathlib pair the JevRSI proofs will be checked against.
#
# WHY A PINNED PAIR, AND WHY THIS PAIR.
#   * Lean 4 is the current major line (Lean 3 is legacy). Latest stable at the time of writing is
#     v4.34.1 (2026-09-24); the locally installed toolchain was v4.29.0, five minor versions behind.
#   * Mathlib publishes a matching release tag for each Lean stable. `v4.34.1` was verified to exist
#     on the remote (commit d13f23b7); Mathlib's master branch instead tracks v4.35.0-rc3, which is
#     why a release tag is pinned rather than a branch.
#   * A claim that "the framework passes the kernel" must be reproducible by a third party. An rc
#     toolchain is a moving target and cannot be the basis of that claim when a stable pair exists.
#     So: Lean v4.34.1 + Mathlib v4.34.1, both exact.
#
# WHY THE NETWORK WORKAROUNDS ARE HERE, because they are measured facts about this host, not
# preferences. A first run failed with `curl 56 Recv failure: Connection was reset` while cloning
# Mathlib, and a second symptom was subtler: a 29-byte fetch from githubusercontent returned HTTP 200
# with ZERO bytes, while the same URL through a mirror returned those same 29 bytes correctly. So the
# link to GitHub completes the handshake, the request is accepted, and the body is then dropped --
# exactly the pattern that kills a multi-gigabyte git transfer while leaving small requests apparently
# fine. Three consequences are encoded below:
#   1. the mirror is injected through GIT_CONFIG_* environment variables, so NO git config file on
#      this machine is read or written (a pre-existing global `insteadOf` rule must not be disturbed);
#   2. the clone is shallow and single-branch, minimising both bytes and ref negotiation -- a full
#      `ls-remote` over the mirror failed with `expected flush after ref listing` even though
#      `--tags` succeeded, so the full history is not merely slower here, it is unreliable;
#   3. an existing checkout is reused when it already points at the pinned commit, so a re-run after
#      an interrupted transfer resumes instead of restarting.
#
# Run once. Idempotent: re-running repairs only what is missing.
set -euo pipefail
export PATH="$HOME/.elan/bin:$PATH"

LEAN_VER="v4.34.1"
MATHLIB_TAG="v4.34.1"
MATHLIB_COMMIT="d13f23b723b8a846827a245b89c10fc7d3f11612"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's own directory

# --- mirror injection, scoped to this process tree only -----------------------------------------
MIRROR="${JEVRSI_GIT_MIRROR:-https://ghproxy.net/}"
if [ -n "$MIRROR" ]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${MIRROR}https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
  echo "  mirror in use: ${MIRROR} (via env only; no git config file touched)"
fi

echo "=== [1/5] install Lean $LEAN_VER ==="
elan toolchain install "leanprover/lean4:${LEAN_VER}" || true
elan toolchain list | sed 's/^/  /'

echo
echo "=== [2/5] project skeleton at $PROJ ==="
mkdir -p "$PROJ/JevRSI"
cd "$PROJ"
echo "leanprover/lean4:${LEAN_VER}" > lean-toolchain
cat > lakefile.toml <<'EOF'
name = "jevRSI"
defaultTargets = ["JevRSI"]

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4"
rev = "v4.34.1"

[[lean_lib]]
name = "JevRSI"
EOF

cat > JevRSI.lean <<'EOF'
import JevRSI.AcceptRule
EOF

echo
echo "=== [3/5] Mathlib source at $MATHLIB_TAG (shallow, resumable) ==="
MATHLIB_DIR="$PROJ/.lake/packages/mathlib"
NEED_CLONE=1
if [ -d "$MATHLIB_DIR/.git" ]; then
  have="$(git -C "$MATHLIB_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$have" = "$MATHLIB_COMMIT" ]; then
    echo "  already at pinned commit ${MATHLIB_COMMIT:0:8} -- reusing"
    NEED_CLONE=0
  else
    echo "  present but at ${have:0:8}, expected ${MATHLIB_COMMIT:0:8} -- refetching"
    rm -rf "$MATHLIB_DIR"
  fi
fi
if [ "$NEED_CLONE" = "1" ]; then
  echo "  cloning (depth 1, single branch) -- slow step, expect tens of minutes"
  rm -rf "$MATHLIB_DIR"
  for attempt in 1 2 3; do
    echo "  attempt $attempt/3 at $(date '+%H:%M:%S')"
    if timeout 5400 git clone --depth 1 --branch "$MATHLIB_TAG" --single-branch \
        https://github.com/leanprover-community/mathlib4 "$MATHLIB_DIR"; then
      break
    fi
    echo "  [WARN] attempt $attempt failed; cleaning partial checkout and retrying"
    rm -rf "$MATHLIB_DIR"
    if [ "$attempt" = "3" ]; then
      echo "  [FAIL] clone failed 3x -- rerun to resume"
      exit 1
    fi
  done
  got="$(git -C "$MATHLIB_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$got" != "$MATHLIB_COMMIT" ]; then
    echo "  [FAIL] commit mismatch: got ${got:0:8}, pinned ${MATHLIB_COMMIT:0:8}"
    exit 1
  fi
  echo "  [OK] $(du -sh "$MATHLIB_DIR" | cut -f1) at ${got:0:8}"
fi

echo
echo "=== [4/5] prebuilt oleans (skips a multi-hour from-source build) ==="
lake exe cache get 2>&1 | tail -8 || echo "  [WARN] cache get failed -- a from-source build is slow but still valid"

echo
echo "=== [5/5] build and kernel-check ==="
set +e
lake build 2>&1 | tail -40
BUILD_EXIT=${PIPESTATUS[0]}
set -e
echo "BUILD_EXIT=$BUILD_EXIT"

if [ "$BUILD_EXIT" = "0" ]; then
  echo
  echo "=== [OK] KERNEL CHECK PASSED -- the theorems in JevRSI/AcceptRule.lean are machine-checked ==="
  echo "  theorems in file: $(grep -c '^theorem' JevRSI/AcceptRule.lean)"
  echo "  sorry occurrences: $(grep -c 'sorry' JevRSI/AcceptRule.lean || true)"
else
  echo
  echo "=== [FAIL] build did not pass -- do NOT record the proofs as verified ==="
fi
echo "=== done: $(date '+%H:%M:%S') ==="
