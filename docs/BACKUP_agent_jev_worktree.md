# agent-jev working-tree backup

Taken before the first `autoresearch_agent.py` run, which calls `git reset --hard` on both the
crash and the discard path.

```json
{
  "_what_this_is": "Verified backup of the agent-jev working tree, taken BEFORE any run of scripts/autoresearch_agent.py. That script calls `git reset --hard` twice (L380 on crash, L472 on discard) and the working tree held uncommitted work, so the first cycle would have destroyed it irrecoverably.",
  "_why_a_tar_and_not_a_commit": "agent-jev is the SUBJECT repository, an external clone. Committing its working tree would be an unrequested modification of a repository this project does not own. A tar outside the repo is non-destructive and reversible.",
  "backup_file": "agent-jev_worktree_20260930_153347.tar",
  "sha256": "66f23d1c7a992b41ad44bb9c4947cfd295d1a245493879164e8fd89240f71d22",
  "bytes": 35573760,
  "verification": {
    "method": "extracted to a temp dir and compared sha256 per file against the live working tree",
    "files_matched": 85,
    "files_differing": 0,
    "files_missing": 0,
    "note": "a backup that has not been read back is not a backup; this one was"
  },
  "covered_tops": [
    "agentjev",
    "scripts",
    "typed_decisions",
    ".github",
    "program.md",
    "results.tsv",
    "README.md",
    "requirements.txt"
  ],
  "excluded": [
    ".git",
    "__pycache__",
    "*.pyc"
  ],
  "how_to_restore": "tar -xf agent-jev_worktree_20260930_153347.tar -C E:/2026-AI4S/agent-jev   # run from E:/2026-AI4S/_backups",
  "_state_at_backup": {
    "uncommitted_entries": 18,
    "modified": 5,
    "untracked_dirs": 13,
    "head": "a965ca8"
  }
}
```
