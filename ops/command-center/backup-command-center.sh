#!/usr/bin/env bash
set -euo pipefail

mode=${1:-quick}
case "$mode" in
  quick)
    keep=${COMMAND_CENTER_BACKUP_KEEP_QUICK:-14}
    ;;
  full)
    keep=${COMMAND_CENTER_BACKUP_KEEP_FULL:-4}
    ;;
  *)
    printf 'usage: %s [quick|full]\n' "$0" >&2
    exit 64
    ;;
esac

home_dir=${COMMAND_CENTER_HOME:-$HOME}
hermes_home=${HERMES_HOME:-$home_dir/.hermes}
sync_db=${HERMES_SYNC_DB:-$home_dir/.hermes-chat-sync/sync.db}
backup_root=${COMMAND_CENTER_BACKUP_ROOT:-$home_dir/.local/state/command-center-backups}
hermes_bin=${HERMES_BIN:-$home_dir/.local/bin/hermes}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
final_dir=$backup_root/$mode-$timestamp
staging_dir=$backup_root/.$mode-$timestamp-$$.partial

umask 077
install -d -m 0700 "$backup_root"
exec 9>"$backup_root/.backup.lock"
if ! flock -n 9; then
  printf 'another command-center backup is already running\n'
  exit 0
fi

cleanup() {
  rm -rf -- "$staging_dir"
}
trap cleanup EXIT
install -d -m 0700 "$staging_dir"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'required command not found: %s\n' "$1" >&2
    exit 1
  }
}

require_command flock
require_command git
require_command python3
require_command sha256sum
require_command sqlite3
if [[ ! -x $hermes_bin ]]; then
  printf 'Hermes CLI is not executable: %s\n' "$hermes_bin" >&2
  exit 1
fi
if [[ ! -f $sync_db ]]; then
  printf 'Hermes sync database is missing: %s\n' "$sync_db" >&2
  exit 1
fi

copy_optional() {
  local source=$1
  local relative=$2
  if [[ -f $source ]]; then
    install -D -m 0600 "$source" "$staging_dir/$relative"
  fi
}

sqlite_backup() {
  local source=$1
  local destination=$2
  install -d -m 0700 "$(dirname "$destination")"
  sqlite3 "$source" <<SQL
.timeout 30000
.backup '$destination'
SQL
  local integrity
  integrity=$(sqlite3 "$destination" 'PRAGMA integrity_check;')
  if [[ $integrity != ok ]]; then
    printf 'SQLite integrity check failed for %s: %s\n' "$destination" "$integrity" >&2
    exit 1
  fi
}

if [[ $mode == quick ]]; then
  "$hermes_bin" backup --quick --label command-center-daily
  snapshot_root=$hermes_home/state-snapshots
  mapfile -t snapshots < <(
    find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' \
      -printf '%T@ %p\n' | sort -nr
  )
  if [[ ${#snapshots[@]} -eq 0 ]]; then
    printf 'Hermes quick backup did not create a snapshot\n' >&2
    exit 1
  fi
  latest_snapshot=${snapshots[0]#* }
  if [[ ! -f $latest_snapshot/manifest.json || ! -f $latest_snapshot/state.db ]]; then
    printf 'Hermes quick snapshot is incomplete: %s\n' "$latest_snapshot" >&2
    exit 1
  fi
  if [[ $(sqlite3 "$latest_snapshot/state.db" 'PRAGMA integrity_check;') != ok ]]; then
    printf 'Hermes quick state.db failed integrity verification\n' >&2
    exit 1
  fi
  cp -a "$latest_snapshot" "$staging_dir/hermes-quick"
else
  "$hermes_bin" backup --output "$staging_dir/hermes-full.zip"
  if [[ ! -s $staging_dir/hermes-full.zip ]]; then
    printf 'Hermes full backup was not created\n' >&2
    exit 1
  fi
  python3 - "$staging_dir/hermes-full.zip" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    bad_member = archive.testzip()
if bad_member is not None:
    raise SystemExit(f"zip integrity check failed at {bad_member}")
PY
fi

sqlite_backup "$sync_db" "$staging_dir/hermes-chat-sync/sync.db"

copy_optional "$home_dir/.hermes-api-key" recovery/.hermes-api-key
copy_optional "$hermes_home/subscription-projects.json" recovery/.hermes/subscription-projects.json
copy_optional "$home_dir/hermes-sync/hermes-sync.mjs" recovery/hermes-sync/hermes-sync.mjs
copy_optional \
  "$home_dir/hermes-subscription-worker/subscription-worker.mjs" \
  recovery/hermes-subscription-worker/subscription-worker.mjs

if [[ -d $home_dir/.config/systemd/user ]]; then
  while IFS= read -r -d '' unit; do
    copy_optional \
      "$unit" \
      "recovery/.config/systemd/user/$(basename "$unit")"
  done < <(
    find "$home_dir/.config/systemd/user" -maxdepth 1 -type f \
      \( -name 'hermes-*.service' -o -name 'hermes-*.timer' -o -name 'cloudflared-hermes.service' \) \
      -print0
  )
fi

if [[ -d $home_dir/.cloudflared ]]; then
  while IFS= read -r -d '' cloudflare_file; do
    copy_optional \
      "$cloudflare_file" \
      "recovery/.cloudflared/$(basename "$cloudflare_file")"
  done < <(find "$home_dir/.cloudflared" -maxdepth 1 -type f -print0)
fi

agent_repo=$home_dir/src/hermes-agent-migration
chat_repo=$home_dir/coding-projects/hermes-chat
{
  printf 'schema=command-center-backup-v1\n'
  printf 'mode=%s\n' "$mode"
  printf 'created_at=%s\n' "$timestamp"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'hermes_agent_commit=%s\n' "$(git -C "$agent_repo" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'hermes_chat_commit=%s\n' "$(git -C "$chat_repo" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'sync_integrity=ok\n'
} >"$staging_dir/metadata.txt"

(
  cd "$staging_dir"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum --check --status SHA256SUMS
)
chmod -R go-rwx "$staging_dir"
mv "$staging_dir" "$final_dir"
trap - EXIT

mapfile -t backups < <(
  find "$backup_root" -mindepth 1 -maxdepth 1 -type d -name "$mode-*" \
    -printf '%T@ %p\n' | sort -nr
)
for ((index = keep; index < ${#backups[@]}; index += 1)); do
  rm -rf -- "${backups[$index]#* }"
done

printf 'command-center %s backup complete: %s\n' "$mode" "$final_dir"
