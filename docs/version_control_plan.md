# Version-control safety plan — OASP (`autoresearch`) and `nanochat-autoresearch`

**Status:** plan only. Nothing has been committed, renamed, or moved. Every fact below was
established by read-only inspection on **2026-09-26** (git 2.53.0.windows.3, git-bash, native
forward-slash paths like `D:/…` and `E:/…` are used throughout because MSYS path translation is
disabled on this host).

**Why this exists:** `docs/incident_2026-09-25_tree_loss.md` records the loss and Recycle-Bin
recovery of this entire working tree. Its Lesson 1 is "commit the tree". Doing that naively —
in the repository that is already present at `D:/2026-AI4S/autoresearch/.git` — would commit the
OASP harness into a **third party's DCIC-2026/deer-flow history**. This plan prevents that.

**Snapshot caveat:** the worktree was being written to *during* this audit (docs/item_expansion_plan.md,
docs/variance_design.md and README.md all changed at 12:22–12:24 while the inspection ran), so every
count below is a timestamped observation rather than a constant. Re-run the Phase 5 verification
immediately before committing, and commit soon — the tree is live and unprotected.

---

## 0. Verdict (one paragraph)

`D:/2026-AI4S/autoresearch/.git` is **not** an OASP repository. It is a locally-initialised
DCIC-2026 competition / deer-flow working copy (4 commits, 2026-04-02 → 2026-04-15, author
`AI Research Assistant <ai-research@example.com>`), and **not one byte of OASP content was ever
committed to it**. It has **no remote configured** and no remote-tracking ref was ever recorded,
so there is no push hazard today — but its history is a foreign project's, and every one of its
908 tracked files is currently missing from the worktree. The safe move is therefore:
**rename that `.git` aside (never delete it), `git init` a brand-new repository in place, commit
the OASP payload under a strict `.gitignore`, and give it a *local bare* backup remote — never a
GitHub remote, and never the deer-flow remote.** `nanochat-autoresearch` needs the opposite kind
of care: it is a legitimate clone whose remote is a third-party fork (`jsegov/autoresearch-win-rtx`),
where only `results.tsv` + an extended `.gitignore` (and optionally `uv.lock`) should be committed
locally, with pushes to `origin` deliberately disabled.

---

## 1. Target 1 — `D:/2026-AI4S/autoresearch`: who owns the existing `.git`?

### 1.1 Evidence

| question | finding | how established |
|---|---|---|
| Remote | **none** — `git remote -v` prints nothing; `git config --local -l` has no `remote.*`; `.git/refs/remotes` does not exist; `.git/logs/refs/remotes/` was never created | `git remote -v`, `git config --local -l`, `ls .git/refs`, `ls -laR .git/logs/refs/` |
| Origin of repo | **locally created with `git init`**, not cloned — the reflog's first entry is `commit (initial): feat: initialize project structure for DCIC 2026 competition`; `grep -c 'clone\|fetch\|push\|origin' .git/logs/HEAD` = **0** | `.git/logs/HEAD` |
| History | **4 commits**, all DCIC-2026 / deer-flow: `10117bd` initialize project structure for DCIC 2026 · `b63486a` DCIC 2026 competition setup docs · `266a7a2` basic orchestrator class · `53b1dbf` data processing agent | `git log --format='%h \| %an <%ae> \| %ad \| %s'` |
| Author identity | `AI Research Assistant <ai-research@example.com>` on all 4 commits | `git log` |
| Head / branches | HEAD = `53b1dbf` on `master`; **10 branches**, all 10 pointing at the *same* commit — tooling-generated worktree branches (`cliff-pie`, `elderly-kilometer`, `enchanted-teacher`, `full-sole`, `smart-middle`, `tender-rowboat`, `worried-tilapia`, `young-snowflake`, `opencode/lucky-harbor`) | `git show-ref`, `git branch -a` |
| Ownership conclusion | A **DCIC-2026 competition repo built on ByteDance's deer-flow** (`.claude/agents/deerflow-professional.json`, `.claude-code-router/`, `deer-flow-main/`, `20260402/`, `2026Timeline/`, `deerflow_2_0_pipeline.py`). Nothing to do with OASP. | `git ls-tree`, `git ls-files` |
| Worktree state | `git status --porcelain` = **931 entries**: 906 ` D` (tracked, deleted from disk), 2 ` M` (`.gitignore`, `README.md`), 23 `??` | `git status --porcelain` |
| Index | **0** entries matching `oasp` — the OASP tree is not even staged | `git ls-files --stage \| grep -c oasp` |
| Dangling objects | 6 218 dangling blobs + 21 dangling commits (`WIP on master: 53b1dbf`, stash-style, 2026-05-30/31 — from the opencode/kilo tooling, not from OASP) | `git fsck --no-progress` |
| Disk | 472 MB total: `.git` 105 MB, `archive/` 334 MB, `benchmarks/` 30 MB, `runs/` 3.8 MB, `analysis/` 0.5 MB, `oasp/` 362 KB | `du -sh` |

### 1.2 Has any OASP file EVER been committed? — **No.**

Three independent checks, all negative:

1. `git log --all --pretty=format: --name-only | sort -u | grep -c '^oasp/'` → **0**.
   The union of paths over *all* commits and *all* branches is 909 paths, every one of them
   deer-flow/DCIC material.
2. No OASP-proprietary filename (`oasp/*`, `tests/test_evaluator.py`, `tests/test_stats.py`,
   `docs/theory.md`, `docs/frontier_positioning.md`, `docs/critical_review_2026-09-26.md`,
   `scripts/run_landscape.py`, `scripts/screen_models.py`, `scripts/power_probe.py`,
   `benchmarks/*/provenance.json`) appears in any commit.
3. Content-level sweep of **every** object in the repository (reachable *and* the 6 218 dangling
   blobs) for the markers `OASP`, `Optimization-as-Search`, `prompt_space`, `git_memory`,
   `test_evaluator` produced exactly one hit — `jTfKk/55v8A98mtN/G3iBXYC/**OASPuL**/hSf8Jv4h`
   — inside the base64 payload of an embedded JPEG in a 700 KB image blob. It is base64 noise,
   not code. **Conclusion: zero OASP content exists anywhere in that `.git`.**

### 1.3 Nested `.git` directories (a real hazard for any naive `git add`)

Exactly three, all inside the legacy archive:

```
archive/vendored/ecc/.git      (full upstream repo, ~2 packs >1 MB each, plus node_modules/)
archive/vendored/MSA/.git
archive/vendored/polar/.git    (plus a .venv/ with cp314 binaries)
```

`archive/` totals **334 MB** (node_modules, `.venv`, images, `examples/*.jsonl`). If a careless
`git add -A` ever ran over this tree, git would turn those three into **gitlinks (mode 160000)**
or warn loudly about embedded repositories, and would push 334 MB of third-party material into
the new history. `archive/` must be ignored wholesale.

Also note the harness itself: `oasp/git_memory.py` creates a **throwaway repository per run** at
`runs/<run_id>/expgit` and, on *reject*, executes `git reset --hard` + `git clean -fd` inside it
(incident doc, "one latent hazard"). No `expgit` directory exists today, but the *first* real
search run will create one. Any `.gitignore` must therefore keep all of `runs/` out of the index,
or that disposable repo's rewritten history becomes entangled with the project repo. (Its
trajectory output lands at `runs/<run_id>/trajectory/`, same directory.)

### 1.4 Linked worktrees registered in that `.git` (must be handled before a rename)

`git worktree list` shows 10 linked worktrees. They live *outside* `.git` but their `.git` files
point *into* it, so renaming `.git` orphans them:

| worktree path | state |
|---|---|
| `D:/2026-AI4S/autoresearch` | main worktree (this one) |
| `C:/Users/Administrator/.local/share/opencode/worktree/10117bd72…/lucky-harbor` | **exists on disk**, holds a full copy of the legacy deer-flow tree (`.claude-code-router/`, `20260402/`, `BACKUP/`, `deer-flow-main/`, `src/`, `experiments/`…), May 31 |
| `D:/2026-AI4S/autoresearch/.kilo/worktrees/{cliff-pie, elderly-kilometer, enchanted-teacher, full-sole, smart-middle, tender-rowboat, young-something…}` | **prunable** — "gitdir file points to non-existent location"; the `.kilo/` directory is not on disk |

Consequence: after the rename, that C: worktree is orphaned. It is a *second, independent copy of
the deer-flow material*, so nothing is uniquely lost — but if any uncommitted deer-flow work in it
matters, deal with it before Phase 2 (step 0d below), or simply accept that restoring the legacy
repo is one `mv` away (Phase 7).

### 1.5 A provenance defect worth fixing while we are here

Every OASP run manifest hard-codes the legacy repo's commit as its provenance:

```json
"environment": { "python": "3.14.7", "platform": "Windows-11-10.0.27891-SP0",
                 "git_commit": "53b1dbf69056a9a1fdaf62904d567abc8f3e077d" }
```

(`runs/pilot_v1_dev/manifest.json`, and likewise `pilot_v2_challenge`, `noise_v1`, `smoke1`.)
That SHA is a **DCIC-2026 commit that does not contain any of the code that produced the data**.
Until the new repository exists and manifests are written against it, every run manifest is
provenance-invalid. Fixing the repository is therefore not housekeeping — it repairs the audit
trail. Expect `runs/pilot_v3_*`-onward manifests to carry the new repo's commit hash; the four
existing manifests stay as historical artefacts (record the mapping in `docs/`).

### 1.6 Recommendation: **Option A — new repository initialised in place; legacy `.git` renamed aside**

| | Option A: `git init` in place, old `.git` renamed aside | Option B: fresh clone elsewhere |
|---|---|---|
| Working tree | stays at `D:/2026-AI4S/autoresearch` — the path every script, manifest and absolute reference already uses | the OASP work would have to be copied or re-run at a second path, creating two divergent copies |
| What gets committed | exactly the same trees either way — a *clone* starts empty, so there is nothing to "clone from" | no advantage |
| Legacy history | preserved, 105 MB, one `mv` away, and it never enters the new repo | same |
| Risk of committing into deer-flow history | eliminated by the rename (the new repo has its own `.git`) | eliminated |
| Disk | 0 extra copies on a D: volume that is at **98 % full (180 GB free)** | +472 MB |

**Option A is recommended**; it is the only option that keeps the *one* live working tree while
guaranteeing the foreign history is out of the index. Use Option B only if you specifically want
the OASP work at a second location — in which case `git init` at the new location and copy the
payload there; never `git clone` the legacy repo.

**Hard prohibitions (any of these silently pollutes a third party):**

* Never run `git add -A`, `git add .`, `git commit -a` before the legacy `.git` has been renamed
  aside. It would stage 906 deletions of the deer-flow tree *plus* the entire OASP harness *plus*
  334 MB of `archive/` and three embedded repos.
* Never `git remote add origin <deer-flow GitHub URL>` in either repo, and never `git push` the
  OASP tree anywhere but the local bare backup from Phase 6.
* Never delete `.git-deerflow-legacy-20260415/` — it is the only copy of those 4 commits and 10
  branches outside the C: worktree (Phase 0 takes a second copy anyway).
* `mkdir` before `mv`; `mv` (rename) not `cp -r` then `rm -rf` — a rename on the same volume is
  atomic and cannot half-copy 105 MB.

---

## 2. Target 1 — exact commands (copy-paste, git-bash)

Work through the phases in order. Phase 0 is a *cold* backup and must be done first; Phase 7 is
the rollback if anything looks wrong.

### Phase 0 — cold backup, before touching git (≈1 minute)

```bash
# 0a. Native / cross-drive backup of the payload (D: is at 98% full; E: has 3.0 TB free)
BK="E:/backup/2026-AI4S/autoresearch_2026-09-26"
mkdir -p "$BK"

cd D:/2026-AI4S/autoresearch
tar -czf "$BK/payload_before_init.tar.gz" \
    --exclude='archive' --exclude='.git' \
    -C D:/2026-AI4S autoresearch
# Verified 2026-09-26: 9.1 MB compressed, 131 members, 0 archive/ or .git members.
# --exclude takes basename patterns (no leading path), so 'archive' and '.git' match at any depth.
# Note: the tarball does contain .env.local (a VERCEL_OIDC_TOKEN) — keep E: as trusted local storage.

# 0b. Preserve the legacy repo's ENTIRE history in one file (all 10 branches)
git -C D:/2026-AI4S/autoresearch bundle create "$BK/legacy_deerflow_53b1dbf.bundle" --all
git -C D:/2026-AI4S/autoresearch bundle verify "$BK/legacy_deerflow_53b1dbf.bundle"

# 0c. Freeze the evidence (this is what makes the plan auditable afterwards)
mkdir -p "$BK/evidence"
cd D:/2026-AI4S/autoresearch
{
  echo "=== git remote -v ==="          ; git remote -v
  echo "=== git log --all ==="          ; git log --all --format='%H %an <%ae> %ad %s'
  echo "=== git status --porcelain ===" ; git status --porcelain
  echo "=== git worktree list ==="      ; git worktree list
  echo "=== HEAD ==="                   ; git rev-parse HEAD
} > "$BK/evidence/target1_git_state.txt" 2>&1
cp .gitignore "$BK/evidence/gitignore_before.txt"
du -sh . .git archive runs analysis benchmarks oasp > "$BK/evidence/sizes_before.txt"

# 0d. No half-finished git operation is in flight, and the C: worktree is accounted for
ls -la .git/index.lock 2>/dev/null || echo "no index.lock (good)"
git -C "C:/Users/Administrator/.local/share/opencode/worktree/10117bd72eed5b68b846697391ba26ee7b19544f/lucky-harbor" status --short | head
#    ^ review, then decide; this worktree becomes unreachable after Phase 2
```

### Phase 1 — rename the foreign `.git` aside (do NOT delete)

```bash
cd D:/2026-AI4S/autoresearch
mv .git .git-deerflow-legacy-20260415          # atomic rename, same volume
ls -d .git-deerflow-legacy-20260415            # must exist
ls -d .git 2>/dev/null || echo "old .git is out of the way (expected)"
git status 2>&1 | head -3                      # expect: fatal: not a git repository
```

### Phase 2 — initialise the OASP repository

```bash
cd D:/2026-AI4S/autoresearch
git init -b main

# Identity for a research repo: set it explicitly, locally (never rely on the inherited
# 'AI Research Assistant' identity, and never let a global default author your commits).
git config --local user.name  "YOUR NAME"
git config --local user.email "you@example.com"

# This host has core.autocrlf=true system-wide. Turn it off *locally*: several artefacts are
# SHA-256'd (benchmark provenance, manifests), so line-ending rewriting must never happen.
git config --local core.autocrlf false
git config --local core.safecrlf false

git config --local -l | head            # verify: no remote.* entries, correct identity
```

### Phase 3 — write `.gitignore` **before** the first `git add`

The `.gitignore` currently on disk is the legacy deer-flow one (Chinese section headings, already
saved to `$BK/evidence/gitignore_before.txt`). Replace it with the content in §3 below:

```bash
cd D:/2026-AI4S/autoresearch
# paste the §3 content here (e.g. `cat > .gitignore <<'EOF' … EOF`)
```

Verify the rules actually bite — **all six lines must print a rule, and the last must fail**:

```bash
cd D:/2026-AI4S/autoresearch
git check-ignore -v runs archive analysis .cache benchmarks/supergpqa/items.jsonl .env.local
git check-ignore -v README.md ; echo "exit=$? (non-zero = README.md is trackable: correct)"
git status --short --ignored | grep '^!!' | head     # what is being held back
```

### Phase 4 — add an explicit allow-list (never `git add -A`)

```bash
cd D:/2026-AI4S/autoresearch
git add .gitignore README.md RESEARCH_PROTOCOL.md Problem.docx \
        oasp tests scripts docs configs benchmarks/*/provenance.json
#   oasp/__pycache__ (209 KB) is skipped by the ignore rules; benchmarks/*/items.jsonl too.
#   `prompts/` is empty — git cannot track empty directories; add prompts/.gitkeep if the
#   directory must exist in a fresh clone.
#   Optional: if you want the written analysis reports versioned, force them in:
#     git add -f analysis/*/report.md analysis/*/report.json
```

### Phase 5 — verify the staged set before committing

```bash
cd D:/2026-AI4S/autoresearch
git status --short | head -40
echo "staged files: $(git ls-files | wc -l)"                 # expect 38 at the time of writing
echo "staged bytes: $(git ls-files -z | xargs -0 du -b 2>/dev/null | awk '{s+=$1} END {print s}')"

# Size guard: nothing larger than 5 MB may be staged (expected max ≈ 0.5 MB)
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -rn | head -5

# Nothing from the danger list may appear in the output of the next command:
git ls-files | grep -E '^(runs/|archive/|analysis/|\.cache/)|items\.jsonl|\.git-deerflow|\.pt$|\.env' \
  && echo "STOP: forbidden paths staged" || echo "clean: no forbidden paths staged"
```

### Phase 6 — commit, then give it a *local* backup remote

```bash
cd D:/2026-AI4S/autoresearch
git commit -m "OASP: initial commit — harness, tests, scripts, docs, benchmark provenance

Version-controlled after the 2026-09-25 tree-loss incident (see
docs/incident_2026-09-25_tree_loss.md). Legacy deer-flow .git preserved
as .git-deerflow-legacy-20260415; no remote is configured, so this commit
can never reach a third-party repository."
git log --stat -1 | tail -5

# Offline durability: a bare repo on E:. LOCAL PATH ONLY — no URL, ever.
git init --bare E:/backup/2026-AI4S/autoresearch.git
git remote add backup E:/backup/2026-AI4S/autoresearch.git
git remote -v                       # exactly one remote, path-based, no github.com
git push -u backup main
git --git-dir=E:/backup/2026-AI4S/autoresearch.git log --oneline   # verify what landed
```

### Phase 7 — rollback (if Phase 5 or 6 looks wrong)

```bash
cd D:/2026-AI4S/autoresearch
mv .git .git-oasp-abandoned-<timestamp>          # move the NEW repo aside (do not delete yet)
mv .git-deerflow-legacy-20260415 .git            # the legacy repo is back, exactly as found
git status --porcelain | wc -l                   # expect 931 again
```

---

## 3. Target 1 — the `.gitignore` (exact content)

Comments must sit on their own lines: `.gitignore` has **no** trailing-comment syntax.

```gitignore
# ============================================================================
# OASP — D:/2026-AI4S/autoresearch
# Rule: version code, configs and small provenance files.
#       Never version bulk data, run outputs, regenerable figures, legacy material.
# ============================================================================

# --- 1. Run outputs & bulk data (append-only JSONL, regenerable plots) -------
runs/
analysis/
.cache/
benchmarks/*/items.jsonl
*.parquet
*.arrow
*.feather

# --- 2. Legacy / third-party material that must never enter this repo -------
archive/
deer-flow-main/
.git-deerflow-legacy-*/

# --- 3. Secrets -------------------------------------------------------------
.env
.env.*
*.env.local
!.env.example

# --- 4. Model weights, checkpoints, binary artefacts ------------------------
*.pt
*.bin
*.ckpt
*.pth
*.safetensors
*.h5
*.hdf5
*.npy
*.npz
*.pkl
*.pickle
*.db
*.sqlite

# --- 5. Python / tooling ----------------------------------------------------
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/

# --- 6. Logs, temp and OS/editor clutter ------------------------------------
*.log
*.tmp
*.bak
~$*
.DS_Store
Thumbs.db
.idea/
.vscode/
```

**Why each of the contentious lines is there**

| line | size / reason |
|---|---|
| `runs/` | 3.8 MB today (`pilot_v1_dev/results.jsonl` 1.4 MB, `pilot_v2_challenge` 2.2 MB) and **append-only by design** — every rerun rewrites history and would bloat the pack forever. It will also contain `runs/<id>/expgit`, a throwaway repo that `oasp/git_memory.py` rewrites with `reset --hard`/`clean -fd`; that must never be entangled with the project repo. |
| `benchmarks/*/items.jsonl` | 30 MB of frozen stimulus (`supergpqa` 20 MB, `mmlu_pro` 9 MB). Fully rebuildable — `oasp/benchmark.py::freeze_benchmark(key)` downloads from the pinned HF mirror URL and re-derives `source_sha256`; that hash, the `hf_repo`/`hf_path`, `n_items` and `frozen: true` live in the 2.5 KB `benchmarks/*/provenance.json`, which **is** committed. |
| `analysis/` | 0.5 MB of figures + `report.md`/`report.json` regenerated from `runs/` by `oasp/figures.py`/`oasp/stats.py`; already carries v1…v5 churn (`analysis/pilot_v2_challenge/fig1_…v5.pdf`), i.e. exactly the binary-churn pattern that bloats a repo. Opt a specific file back in with `git add -f`. |
| `archive/` | 334 MB and contains three nested `.git` repos (`ecc`, `MSA`, `polar`) plus `node_modules/`, `.venv/`, PNGs and test JSONL — the single largest source of accidental-pollution and repo-bloat risk. |
| `.git-deerflow-legacy-*/` | without this line, a later `git add -A` would swallow the renamed 105 MB legacy `.git` while looking like an ordinary directory name. |
| `.env`, `.env.*`, `*.env.local` | `.env.local` (1.4 KB, holds `VERCEL_OIDC_TOKEN`) sits in the tree; a token must never be committed. It is a deer-flow leftover with no research value — deletion is the user's call. |
| `~$*` | the 2026-09-25 incident found a `~$roblem.docx` Word lock artefact in the tree; lock files are noise. |
| `prompts/` | deliberately **not** ignored — it is currently empty, so an empty directory simply cannot be tracked; add `.gitkeep` if needed. |

**Payload to be committed:** ≈ 440 KB — `oasp/` (362 KB), `tests/` (12 KB), `docs/` (60 KB),
`scripts/` (36 KB), `configs/oasp.yaml` (2.6 KB), `README.md`, `RESEARCH_PROTOCOL.md`,
`Problem.docx` (19 KB), plus `benchmarks/*/provenance.json`.

**Not committed, and why that is acceptable** — the raw run evidence (3.8 MB of `results.jsonl`)
and the derived figures stay out of git; git is the wrong tool for append-only data anyway, and
the incident showed `runs/` survives kills unharmed. Their durability comes from the cold copy in
Phase 0 plus a periodic `robocopy D:\2026-AI4S\autoresearch\runs E:\backup\…\runs /MIR`. If you
would rather have the evidence *in* git, the honest alternative is to ignore only the JSONL
(`runs/**/results.jsonl`, `runs/**/expgit/`) and track `runs/*/manifest.json` +
`runs/*/variants.json` (31 KB) — do not un-ignore the whole directory by accident.

---

## 4. Target 2 — `D:/2026-AI4S/nanochat-autoresearch`

### 4.1 Evidence

| question | finding |
|---|---|
| Remote | `origin  https://github.com/jsegov/autoresearch-win-rtx.git` (fetch **and** push) — a **third-party fork** of karpathy's `autoresearch`/`nanochat` |
| HEAD | `a4123c6` = `origin/master` = `master` — nothing has been committed since the clone |
| `train.py` vs `a4123c6` | **unchanged** — `git diff a4123c6 -- train.py` is empty; `git diff --name-only HEAD` returns only `uv.lock` |
| `program.md` | **tracked upstream and unmodified** (`git ls-files` lists it; it is not in `git status`). It is the file the *human* iterates on — commit it only after an edit |
| `prepare.py` | tracked, unmodified (and read-only by the project's own rules) |
| Modified | `uv.lock` only — 42 insertions / 401 deletions, pure resolution-marker reordering + trimming from `uv sync` on Windows |
| Untracked | `.cache/` **642 MB** (`datasets/tinystories/data/tinystories_gpt4_clean.parquet` + `tokenizer/{dataset.txt,token_bytes.pt,tokenizer.pkl}`, `active_dataset.txt` = `tinystories`) · `.logs/` 29 KB (`baseline.log`, `prepare.log`, `repl_1.log`, `replicates.log`, `smoke.log`, `uv_sync.log`) · `checkpoint_pre_eval.pt` **152 MB** · `results.tsv` 135 B |
| Live job is writing right now | `checkpoint_pre_eval.pt` and `.logs/repl_1.log` both have mtime **12:21 today**; `results.tsv` 01:04. `train.py:1204` does `torch.save(state_dict, "checkpoint_pre_eval.pt")`. **Do not touch these paths** — no `git clean`, no `git checkout`, no deletion |
| `results.tsv` | 2 lines: header `commit val_bpb memory_gb status description` + `a4123c6 0.863803 1.7 keep baseline (stock train.py, TIME_BUDGET=300, batch=4, act-ckpt on)` — this is the deliverable of the whole experiment loop and **must be versioned** |
| Existing `.gitignore` | upstream 20 lines: `__pycache__/`, `*.py[oc]`, `build/`, `dist/`, `wheels/`, `*.egg-info`, `.venv`, `worktrees/`, `results/`, `queue/`, `CLAUDE.md`, `AGENTS.md`, `dev/`. It does **not** cover `.cache/`, `.logs/`, `*.pt` or `checkpoint_pre_eval.pt`. Note `results/` (a directory) does **not** match `results.tsv` — correctly still untracked, and intended to be committed |
| LFS | no `.gitattributes` in the repo, but the **global and system** git config install `filter.lfs.*` (`git-lfs clean/smudge/process`, `required=true`). Nothing is LFS-tracked today; do not add an LFS rule for the 152 MB checkpoint — ignore it instead |
| Other global settings that matter | `core.autocrlf=true` (system), `safe.directory=*` (any accidental `git` command runs in *any* directory without complaint), `url.https://github.com/.insteadof=git@github.com:`, `http.sslverify=false` |

### 4.2 What to commit vs ignore

**Commit (local only):** `results.tsv` (the experiment log), the extended `.gitignore`, and — as a
deliberate choice — `uv.lock` if you want the exact Windows-resolved environment recorded.
**Ignore:** `.cache/`, `.logs/`, `checkpoint_pre_eval.pt`, and any future `*.pt`.
**Nothing else changes:** `train.py`, `program.md`, `prepare.py` are byte-identical to `a4123c6`,
so there is no code change to commit — and this also means the baseline row in `results.tsv`
honestly describes stock code.

### 4.3 Exact commands

```bash
# 0. Freeze the pre-state, outside the repo
BK="E:/backup/2026-AI4S/nanochat-autoresearch_2026-09-26"
mkdir -p "$BK"
cd D:/2026-AI4S/nanochat-autoresearch
git status --porcelain --untracked-files=all > "$BK/pre_state.txt"
git diff HEAD > "$BK/uv_lock_before_decision.diff"
cp results.tsv "$BK/results_tsv_snapshot_2026-09-26.tsv"

# 1. Make an accidental push to the third-party fork impossible (local config; reversible)
git config remote.origin.pushurl no_push
git remote -v
#   origin  https://github.com/jsegov/autoresearch-win-rtx.git (fetch)
#   origin  no_push (push)                          <- any `git push` now fails harmlessly

# 1b. Belt and braces: refuse pushes from a hook too (.git/hooks is not versioned)
printf '#!/bin/sh\necho "PUSH BLOCKED: local working copy of jsegov/autoresearch-win-rtx" >&2\nexit 1\n' \
  > .git/hooks/pre-push
chmod +x .git/hooks/pre-push

# 2. Extend .gitignore — APPEND ONLY, never rewrite the upstream rules
cat >> .gitignore <<'EOF'

# --- autoresearch-win-rtx local additions: runtime artefacts ----------------
.cache/
.logs/
checkpoint_pre_eval.pt
*.pt
EOF

# 3. Verify: the three big artefacts are ignored, results.tsv is NOT
git check-ignore -v .cache/datasets/tinystories/tokenizer/tokenizer.pkl \
                    .logs/baseline.log checkpoint_pre_eval.pt
git status --short          # expect exactly:  M .gitignore   ?? results.tsv

# 4. Commit the experiment log (the job is appending: confirm the last line is complete first)
tail -2 results.tsv
git add .gitignore results.tsv
git commit -m "local: record baseline run + ignore runtime artefacts (.cache, .logs, checkpoints)"

# 5. uv.lock — either record the Windows resolution, or discard the churn. Pick one.
git add uv.lock && git commit -m "local: uv.lock re-resolved by uv sync on Windows"
#   ...or:  git restore uv.lock

# 6. Local durability, same pattern as Target 1 (path, never a URL)
git init --bare E:/backup/2026-AI4S/nanochat-autoresearch.git
git remote add backup E:/backup/2026-AI4S/nanochat-autoresearch.git
git push -u backup master
git log --oneline -3        # local commits are ahead of origin/master by design
```

**Resulting posture:** `master` diverges from `origin/master` by one or two *local* commits; the
push URL is disabled; the 642 MB cache, 152 MB checkpoint and 29 KB of logs stay out of the index.
Never run `git push origin master` — if the change is ever to be shared, first fork to the user's
own GitHub account and add *that* as a second remote.

**Ongoing loop rule:** after each experiment, `tail -2 results.tsv` (complete line?) →
`git add results.tsv` (+ the edited `train.py`, if that run changed it) → commit with the row's
description as the message. `results.tsv` column 1 already carries the code commit, so the log
and the code stay cross-referenced.

---

## 5. Standing rules (both repos)

1. **Never `git add -A` / `git add .` in either tree.** Use explicit paths; both trees contain
   large, actively-written, or third-party material that a blanket add would sweep in.
2. **No remote in the OASP repo except the local bare `backup`.** No GitHub remote, no deer-flow
   remote, ever. Verify with `git remote -v` before every push.
3. **`.git-deerflow-legacy-20260415/` is read-only history**: it must never be deleted, moved into
   another repo, or `git gc`'d; the bundle in `E:/backup/…` is its second copy.
4. **`runs/<id>/expgit` stays disposable.** `oasp/git_memory.py` may `reset --hard` + `clean -fd`
   inside it; that is only safe while `runs/` is entirely outside the project index. If a future
   change ever tracks anything under `runs/`, re-verify that rule first.
5. **Manifest provenance depends on this plan being executed**: until the new repo exists,
   every `runs/*/manifest.json` cites DCIC commit `53b1dbf`. Fix, then re-verify with
   `git log -1 --format=%H` matched against a new manifest.
6. **Checkpoint after every session**: `git status --short`, commit, `git push backup main` (or
   `master`). The acknowledged top risk in this project is version-control neglect; a 30-second
   commit is the mitigation for a 310 MB Recycle-Bin recovery.
7. **`safe.directory=*` is set globally** — git will happily operate in any directory it is
   pointed at, so "which repo am I in?" must be checked deliberately (`git rev-parse --show-toplevel`)
   before any write command.
8. **A GPU job is running** in `nanochat-autoresearch` while these steps are carried out: none of
   them touch `.cache/`, `.logs/`, `checkpoint_pre_eval.pt` or the GPU, and Step 4 of §4.3 only
   *reads* `results.tsv`. Avoid `git gc`/`git repack` under `D:/` during the run in any case.

---

## 6. Verification checklist (run after executing)

```bash
# Target 1
cd D:/2026-AI4S/autoresearch
git rev-parse --show-toplevel                       # D:/2026-AI4S/autoresearch
git remote -v                                       # only: backup  E:/backup/.../autoresearch.git
git log --oneline -1                                # the OASP initial commit, author = you
git ls-files | wc -l                                # ≈ 38 at the time of writing
git ls-files | grep -cE '^(runs/|archive/|analysis/)'   # must be 0
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -rn | head -3   # max ≈ 0.45 MB
git ls-files | grep -c '\.git-deerflow-legacy'       # must be 0
git status --short | grep -c '^??'                   # remaining untracked = ignored artefacts only

# Target 2
cd D:/2026-AI4S/nanochat-autoresearch
git remote -v                                       # push URL must read no_push
git log --oneline -2 ; git status --short           # clean except live-run artefacts
git check-ignore .cache .logs checkpoint_pre_eval.pt # all three must be listed
git ls-files | grep -c '^results.tsv$'               # must be 1
```

If any check fails, use the Phase 7 rollback (§2) before doing anything else.
