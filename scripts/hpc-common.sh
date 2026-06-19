#!/usr/bin/env bash
# Shared SSH/ControlMaster helper for the UCL HPC (SGE) automation.
#
# Sourced by submit_run.sh and watch_runs.sh. Reads connection + SMTP config
# from a gitignored `.env` at the repo root (copy `.env.example` -> `.env`).
#
# It builds ONE multiplexed SSH connection (ProxyJump + ControlMaster) and
# exposes `remote_ssh`. Open the master once with `bash scripts/hpc-common.sh open`
# (single password prompt); every later cluster touch reuses that one socket
# -- this is the anti-SSH-saturation guarantee. NEVER fan out parallel ssh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — run: cp $ROOT_DIR/.env.example $ROOT_DIR/.env  (then edit it)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${HPC_USER:?Set HPC_USER in .env}"
: "${HPC_GATEWAY:?Set HPC_GATEWAY in .env}"
: "${HPC_LOGIN:?Set HPC_LOGIN in .env}"
: "${HPC_REMOTE_PROJECT:?Set HPC_REMOTE_PROJECT in .env}"
: "${HPC_RUN_HOME:?Set HPC_RUN_HOME in .env (absolute cluster path where run scripts write experiments/ + .last_<tag>_base)}"
: "${HPC_CONTROL_PERSIST:=yes}"

SSH_TARGET="$HPC_USER@$HPC_LOGIN"
SSH_PROXY="$HPC_USER@$HPC_GATEWAY"
CONTROL_DIR="$ROOT_DIR/.ssh-control"
CONTROL_PATH="$CONTROL_DIR/%r@%h:%p"
mkdir -p "$CONTROL_DIR"; chmod 700 "$CONTROL_DIR"

COMMON_SSH_OPTS=(
  -o "ProxyJump=$SSH_PROXY"
  -o "ControlMaster=auto"
  -o "ControlPath=$CONTROL_PATH"
  -o "ControlPersist=$HPC_CONTROL_PERSIST"
  -o "ServerAliveInterval=60"
  -o "ServerAliveCountMax=10"
)
if [[ "${HPC_AUTH_MODE:-password}" == "key" ]]; then
  : "${HPC_KEY:?Set HPC_KEY in .env when HPC_AUTH_MODE=key}"
  SSH_OPTS=("${COMMON_SSH_OPTS[@]}" -o "IdentitiesOnly=yes" -i "$HPC_KEY")
else
  SSH_OPTS=("${COMMON_SSH_OPTS[@]}" -o "PreferredAuthentications=password,keyboard-interactive" -o "PubkeyAuthentication=no")
fi
# A single SSH command string for rsync's -e (same multiplexed socket).
SSH_CMD="ssh ${SSH_OPTS[*]}"

remote_ssh() { ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"; }

# `open`  : seed the ControlMaster (one password prompt; creates no remote files)
# `check` : is the master socket alive?
# `close` : tear down the master socket
case "${1:-}" in
  open)   ssh "${SSH_OPTS[@]}" -Nf "$SSH_TARGET" && echo "ControlMaster open to $SSH_TARGET" ;;
  check)  ssh -O check "${SSH_OPTS[@]}" "$SSH_TARGET" ;;
  close)  ssh -O exit  "${SSH_OPTS[@]}" "$SSH_TARGET" 2>/dev/null || true; echo "closed" ;;
esac
