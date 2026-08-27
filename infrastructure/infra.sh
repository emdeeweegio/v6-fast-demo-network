#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Node profiles are just nodes.<profile>.env files sitting next to this
# script (see nodes.beach.env, nodes.argos.env) — discovered by glob so
# adding a new one never requires touching infra.sh.
list_node_profiles() {
  local f name
  for f in nodes.*.env; do
    [ -e "$f" ] || continue
    name="${f#nodes.}"
    name="${name%.env}"
    printf '%s\n' "$name"
  done
}

# Pull --nodes <profile> / --nodes=<profile> out of the args before
# dispatching, so it works uniformly across every command (up, down,
# preflight, test) instead of each one having to know about profiles.
args=()
nodes_profile=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nodes=*)
      nodes_profile="${1#--nodes=}"
      shift
      ;;
    --nodes)
      shift
      nodes_profile="${1:-}"
      if [ -z "$nodes_profile" ]; then
        echo "--nodes requires a profile name (e.g. --nodes beach)" >&2
        exit 1
      fi
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
set -- "${args[@]}"

if [ -n "$nodes_profile" ]; then
  nodes_profile_file="./nodes.${nodes_profile}.env"
  if [ ! -f "$nodes_profile_file" ]; then
    echo "Unknown node profile '$nodes_profile' (no $nodes_profile_file)" >&2
    echo "Available profiles: $(list_node_profiles | paste -sd' ' -)" >&2
    exit 1
  fi
  export NODES_CONFIG="$nodes_profile_file"
fi

command_name="${1:-help}"
shift || true

case "$command_name" in
  up)
    ./setup.sh "$@"
    ;;
  down)
    ./shutdown.sh "$@"
    ;;
  preflight)
    source ./config.env
    source ./functions.sh
    init_config_defaults
    preflight_checks
    load_node_specs
    validate_node_specs
    print_node_specs
    log "Preflight checks passed"
    ;;
  test)
    "$SCRIPT_DIR/../infrastructure_tests/container_count.sh"
    "$SCRIPT_DIR/../infrastructure_tests/container_naming.sh"
    ;;
  help|--help|-h)
    cat <<USAGE
Usage: ./infra.sh [--nodes <profile>] <command> [args]

Commands:
  preflight           Validate local prerequisites and node specs
  up [--recreate-env] Start server, nodes and UI from config
  down                Stop and clean up server/nodes/UI
  test                Run infrastructure smoke tests

Options:
  --nodes <profile>   Use nodes.<profile>.env (default: lung1)
                       Available profiles: $(list_node_profiles | paste -sd' ' -)
USAGE
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    ./infra.sh help
    exit 1
    ;;
esac
