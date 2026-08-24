# Reversible Deletion

Status: implemented and locally validated, awaiting command-center deployment,
2026-08-23

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
6. Root, home, system directories, project roots, user deny rules, sudo password
   injection, permission escalation, and uninspectable operations remain
   non-overridable policy blocks.
7. A command that cannot be parsed into exact static delete targets uses the
   normal **Allow once** / **Deny** approval path. It is never guessed safe.
8. Non-file destruction such as force push, Git history replacement, database
   drops, service disruption, and remote deletion continues to require approval.

## Storage

The default store is:

```text
$HERMES_HOME/trash/
  trash.db
  payloads/<item-id>/payload
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
```

Defaults ship disabled so an upstream install does not acquire a new retention
policy without an operator decision. Retained restored copies count against the
store quota until they are explicitly purged.

## Automatic Coverage

The first implementation covers:

- exact `rm`, `unlink`, and `rmdir` operands that contain no variables, globs,
  substitutions, redirects, or shell pipelines;
- paths whose canonical parent is inside the selected workspace;
- exact descendants of `/tmp` or `/var/tmp`, excluding the temp root itself;
- inspectable Codex `fileChange` delete and rename items.

Plain non-recursive `rm` is covered independently of the legacy dangerous
command regex catalog. Common opaque delete APIs such as Python
`shutil.rmtree`/`os.unlink` and Node `fs.rm` are recognized as deletion but are
not guessed into static targets; they use browser approval.

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

The implementation lives at the Codex app-server approval boundary, not in an
agent prompt or shell alias. For an exact delete, Hermes validates every target,
publishes every payload, commits the ledger, and only then returns `accept` to
Codex. Capture failure leaves the original operation waiting for browser review.
Restore and purge are not model tools.

## Known Boundary

No command-string policy can intercept arbitrary deletion hidden inside an
opaque program. Hermes therefore detects common opaque deletion forms and sends
them to interactive approval, but this subsystem is not a filesystem snapshot
or kernel-level write monitor. Stronger guarantees require per-run overlay
filesystems or filesystem-native snapshots under a dedicated worker account.

## Verification

Pre-deployment evidence on 2026-08-23:

- 97 focused Python tests cover parsing, boundary checks, symlinks, quotas,
  idempotency, command and file-change routing, authentication, restore
  conflicts, project isolation, and purge tombstones.
- Hermes Chat's full 82-test application suite is green: 8 chat, 21 management,
  3 transcript, and 50 closure tests.
- The optimized Next.js production build succeeds with all three Trash proxy
  routes present.
- 10 Playwright browser tests pass, including the Trash lifecycle at 1280x800
  and 390x844 with long-path overflow checks and two-step permanent deletion.

Live command-center delete/restore/purge evidence is recorded here after
deployment.
