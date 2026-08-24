#!/usr/bin/env bash
set -euo pipefail

source_dir=${SOURCE_DIR:-$HOME/src/hermes-agent/ops/command-center}
unit_dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

install -D -m 0700 \
  "$source_dir/backup-command-center.sh" \
  "$HOME/.local/bin/backup-command-center"
install -d -m 0755 "$unit_dir"
for unit in \
  hermes-command-center-backup.service \
  hermes-command-center-backup.timer \
  hermes-command-center-full-backup.service \
  hermes-command-center-full-backup.timer
do
  install -m 0644 "$source_dir/$unit" "$unit_dir/$unit"
done

systemctl --user daemon-reload
systemctl --user enable --now \
  hermes-command-center-backup.timer \
  hermes-command-center-full-backup.timer

systemctl --user list-timers \
  hermes-command-center-backup.timer \
  hermes-command-center-full-backup.timer \
  --no-pager
