# Reversible Deletion

Status: per-target Trash and workspace snapshots deployed and verified on
command-center, 2026-08-23

Hermes command-center treats ordinary file deletion as a reversible operation.
The selected project remains the write boundary, but a statically inspectable
delete inside that boundary may proceed without interrupting the run after
Hermes has captured a protected copy. Permanent deletion remains a human-owned
action.

## Invariants

1. A source is never deleted automatically unless its protected copy completed
   successfully and the ledger transaction committed.
2. The trash store is outside every configured project workspace and is not
   writable through the normal Codex workspace profile.
3. The browser sends project keys and trash item IDs, never filesystem paths.
4. Restore uses the original server-recorded path and refuses to overwrite an
   occupied destination.
5. Purge requires an authenticated, explicit browser action. Agent commands
   cannot purge or mutate the trash store.
6. Root, home, system directories, project roots, recovery runtime/storage,
   user deny rules, sudo password injection, and permission escalation remain
   non-overridable policy blocks.
7. Direct `rm`, `unlink`, and `rmdir` commands use an owner-only `PATH` shim.
   The shim validates shell-expanded argv, captures each target, and invokes
   the real system binary only after every capture commits. A capture or
   boundary failure returns exit 125 without deleting the source.
8. Non-file destruction such as force push, Git history replacement, database
   drops, service disruption, and remote deletion continues to require approval.
9. When workspace snapshots are enabled, a native Codex turn never starts until
   its pre-turn recovery point has completed. A recognized destructive command
   that emits a Codex exec-approval request gets another checkpoint immediately
   before browser review when it could not be captured as an exact Trash item.

## Storage

The default store is:

```text
$HERMES_HOME/trash/
  trash.db
  payloads/<item-id>/payload

$HERMES_HOME/reversible-delete-runtime/<run-key>/
  bin/{rm,unlink,rmdir}
  runner.py
  lib/
```

The SQLite ledger records item ID, project key, run/session identity, original
path, payload path, item type, byte count, capture time, status, and restoration
or purge time. Store and payload directories are owner-only. Payload capture is
written to a temporary item directory and atomically published only after the
copy succeeds.

Default limits are 5 GiB per item, 100 GiB total active payloads, and 100,000
filesystem entries per item. Exceeding a limit fails closed into interactive
approval rather than deleting without protection. There is no automatic purge.

Enable the command-center policy in `config.yaml`:

```yaml
codex_runtime:
  reversible_deletion:
    enabled: true
    max_item_bytes: 5368709120
    max_total_bytes: 107374182400
    max_entries: 100000
    temp_roots:
      - /tmp
      - /var/tmp
  workspace_snapshots:
    enabled: true
    keep_snapshots: 32
    timeout_seconds: 600
```

Defaults ship disabled so an upstream install does not acquire a new retention
policy without an operator decision. Retained restored copies count against the
store quota until they are explicitly purged.

## Automatic Coverage

The deletion boundary covers:

- direct `rm`, `unlink`, and `rmdir` commands, including shell variables and
  globs, because capture happens after expansion at the argv boundary;
- exact static removal commands at the Codex approval boundary as a fallback
  when a protected shim could not be installed;
- paths whose canonical parent is inside the selected workspace;
- exact descendants of `/tmp` or `/var/tmp`, excluding the temp root itself;
- inspectable Codex `fileChange` delete and rename items.

Plain non-recursive `rm` and recursive `rm -rf` use the same protected path.
The shim is copied into an owner-only, run-scoped directory before Codex starts;
its interpreter, policy, workspace, project, real binaries, and source hashes
are server-bound. The selected runtime directory is prepended to the Codex
subprocess `PATH`. A model cannot redirect it by changing `HERMES_HOME`, and a
policy-source change during the run fails closed.

Compound shell programs, redirects, command substitutions, and explicit binary
paths retain approval-time capture/review rather than being classified as one
shim-covered command. The shim still protects a normal `rm` invocation inside
an allowed compound command, but Hermes does not auto-approve the entire shell
program merely because one segment is reversible.

Common opaque delete APIs such as Python
`shutil.rmtree`/`os.unlink` and Node `fs.rm` are recognized as deletion but are
not rewritten into synthetic `rm` commands. They use browser approval when
Codex emits an exec-approval request; regardless of that protocol detail, the
pre-turn workspace snapshot protects every file that existed when the turn
began.

The review classifier also recognizes `find -delete`/`-exec`, `xargs rm`,
`git clean`/`reset`/`restore`/`checkout`, `rsync --delete*`, `shred -u`,
`truncate`, `dd of=`, non-append `tee`, clean targets in common build/package
commands, BusyBox removal, and common Python, Node, Deno, Ruby, Perl, Java, Go,
Rust, and PowerShell deletion APIs. These dynamic forms rely on the workspace
snapshot rather than pretending their targets can be recovered from command
text safely.

Missing targets under force-style cleanup are safe no-ops. Symlinks are captured
as symlinks and never followed. Paths crossing a filesystem boundary are copied,
not moved, so the original command remains the only process that removes the
source after capture.

## Browser Contract

Hermes Chat exposes a Trash surface that lists active items for the selected
project and shows original path, capture time, type, size, source-presence state,
and originating run. Each item supports:

- **Restore**, which fails if the original destination is occupied;
- **Permanently delete**, which opens a confirmation step and purges only that
  item after explicit confirmation.

The gateway API is authenticated with the existing API-server bearer token:

```text
GET  /v1/trash?project=<project-key>
POST /v1/trash/<item-id>/restore
POST /v1/trash/<item-id>/purge
```

Every endpoint resolves project keys and item IDs server-side. Purged ledger
rows remain as tombstones so permanent deletion remains auditable.

The implementation is server enforcement, not an agent prompt or shell alias.
Hermes installs the protected command runtime before spawning Codex and keeps
the approval-boundary parser as a fallback. For a shim-covered delete, Hermes
auto-accepts the Codex request because the real binary remains unreachable
through that command until the expanded targets are archived. Restore and purge
are not model tools.

## Workspace Snapshots

Workspace snapshots live outside project workspaces under:

```text
$HERMES_HOME/workspace-snapshots/<project-and-hash>/<snapshot-id>/
  manifest.json
  payload/
```

Hermes uses `rsync --link-dest` so unchanged files share read-only recovery
inodes between snapshots while changed files consume new space. Publication is
atomic, snapshots are serialized per project, and the oldest complete recovery
points are pruned after the configured count. The snapshot store and Trash
store are both non-overridable protected namespaces.

List snapshots and materialize one into a new, unoccupied directory:

```bash
python -m tools.workspace_snapshots list --project reg-watch
python -m tools.workspace_snapshots materialize \
  --project reg-watch \
  --snapshot <snapshot-id> \
  --destination /home/will/recovery/reg-watch-<snapshot-id>
```

Materialization never writes over the live workspace. Compare the recovery copy
and promote only the files actually needed.

## Known Boundary

Every file present at turn start is recoverable regardless of which executable
later deletes or overwrites it. Expanded `rm`-family Trash capture and the
just-in-time checkpoint also protect files created earlier in the same turn.
An explicit system-binary path or a completely unknown executable can still
create and delete a brand-new file inside one turn without leaving a version if
Codex emits no approval event. Closing that final syscall-level gap requires
per-command execution in an OverlayFS workspace or a filesystem-native snapshot
boundary, not more command regexes.

## Verification

Automated evidence on 2026-08-23:

- 97 focused Python tests cover parsing, boundary checks, symlinks, quotas,
  idempotency, command and file-change routing, authentication, restore
  conflicts, project isolation, and purge tombstones.
- Hermes Chat's full 82-test application suite is green: 8 chat, 21 management,
  3 transcript, and 50 closure tests.
- The optimized Next.js production build succeeds with all three Trash proxy
  routes present.
- 10 Playwright browser tests pass, including the Trash lifecycle at 1280x800
  and 390x844 with long-path overflow checks and two-step permanent deletion.

Live command-center evidence on 2026-08-23:

- Gateway commit `d60159f5d` was deployed, the policy was enabled through the
  Hermes configuration interface, and `hermes-gateway.service` was restarted
  only after confirming no run was active.
- Run `run_1a03bf3504c243cb9fc6793057a090c7` executed the exact command
  `rm -rf /tmp/hermes-trash-e2e-20260823` and completed normally. The source was
  absent only after the protected capture and ledger commit succeeded.
- Ledger item `b7c145c4804741dc8a875bf946cb3544` retained the directory under
  the `reg-watch` project with its original path, run ID, type, byte count, and
  entry count.
- Restore recreated the file with byte-identical content. Purge then removed
  only the protected payload, left the restored source intact, and retained the
  ledger tombstone with status `purged`.
- Hermes Chat commit `0c8808a` was pushed to `main`. The production Vercel
  Trash route responds with HTTP 401 without credentials, proving both that the
  new route is deployed and that its authentication gate is active.
- Canonical Hermes commit `5bdda3467` is deployed from
  `/home/will/src/hermes-agent` through the external
  `~/.hermes/venvs/hermes-command-center` environment. Gateway and sync remained
  active after one idle-checked restart.
- Run `run_3791fc9a3864431c86ae4c16dc1f6473` executed Python
  `os.remove(...)` against an untracked marker in `hermes-chat`. Codex emitted no
  approval request, which exercised the broad recovery boundary rather than an
  exact command parser.
- Snapshot `20260824T032940.241537Z-badbfb906a` contained the marker before the
  turn. Materializing it into a new `/tmp` destination restored byte-identical
  content after the live source had been deleted.
- The Linux deployment passed 151 focused snapshot, deletion, session, and API
  runtime assertions. One unrelated background-review test retains its existing
  asynchronous failure on both Mac and Linux baselines.
