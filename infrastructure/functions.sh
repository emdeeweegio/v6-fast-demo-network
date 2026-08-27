#!/bin/bash

log() {
  printf '[infra] %s\n' "$*"
}

warn() {
  printf '[infra][warn] %s\n' "$*" >&2
}

fail() {
  printf '[infra][error] %s\n' "$*" >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

expand_tilde() {
  local path="$1"
  printf '%s' "${path/#\~/$HOME}"
}

looks_like_uri() {
  local value="$1"
  [[ "$value" == *"://"* ]] || [[ "$value" == sqlite:* ]]
}

abspath_if_local_path() {
  local value="$1"

  if looks_like_uri "$value"; then
    printf '%s' "$value"
    return
  fi

  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$(pwd)" "${value#./}"
  fi
}

ensure_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || fail "Required command '$command_name' not found in PATH"
}

init_config_defaults() {
  PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}"
  VERSION_VANTAGE6="${VERSION_VANTAGE6:-4.14.0}"
  ENVIRONMENT="${ENVIRONMENT:-DEV}"

  VENV_PATH="$(expand_tilde "${VENV_PATH:-./venv}")"
  REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

  DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/mdw-nl/vantage6/infrastructure}"
  V6_SERVER_IMAGE_NAME="${V6_SERVER_IMAGE_NAME:-server-lite}"
  V6_SERVER_IMAGE_TAG="${V6_SERVER_IMAGE_TAG:-4.14.0-rc8}"
  V6_NODE_IMAGE_NAME="${V6_NODE_IMAGE_NAME:-node-lite}"
  V6_NODE_IMAGE_TAG="${V6_NODE_IMAGE_TAG:-4.14.0-rc8}"
  V6_UI_IMAGE_NAME="${V6_UI_IMAGE_NAME:-ui}"
  V6_UI_IMAGE_TAG="${V6_UI_IMAGE_TAG:-4.14.0-rc8}"
  SERVER_URL="${SERVER_URL:-http://localhost:5070}"
  API_PATH="${API_PATH:-/api}"

  SERVER_CONFIG="${SERVER_CONFIG:-./demoserver.yaml}"
  SERVER_CONFIG="$(abspath_if_local_path "$SERVER_CONFIG")"

  local server_config_basename
  server_config_basename="$(basename "$SERVER_CONFIG")"
  SERVER_NAME="${SERVER_NAME:-${server_config_basename%.*}}"

  NODES_CONFIG="${NODES_CONFIG:-./nodes.lung1.env}"
  NODES_CONFIG="$(abspath_if_local_path "$NODES_CONFIG")"

  GENERATED_DIR="${GENERATED_DIR:-./generated}"
  GENERATED_DIR="$(abspath_if_local_path "$GENERATED_DIR")"

  ENTITIES_FILE="${ENTITIES_FILE:-$GENERATED_DIR/entities.generated.yaml}"
  ENTITIES_FILE="$(abspath_if_local_path "$ENTITIES_FILE")"

  DATA_DIR_DEFAULT="${DATA_DIR_DEFAULT:-../data/lung1}"
  DATA_DIR_DEFAULT="$(abspath_if_local_path "$DATA_DIR_DEFAULT")"

  STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"
  CLEAN_LOCAL_STATE="${CLEAN_LOCAL_STATE:-true}"
  KEEP_CONTAINERS="${KEEP_CONTAINERS:-false}"
  UI_ENABLED="${UI_ENABLED:-true}"
  UI_PORT="${UI_PORT:-80}"
  UI_URL="${UI_URL:-http://localhost}"
  COLLABORATION_NAME="${COLLABORATION_NAME:-v6-demo}"
}

server_image_ref() {
  printf '%s/%s:%s' "$DOCKER_REGISTRY" "$V6_SERVER_IMAGE_NAME" "$V6_SERVER_IMAGE_TAG"
}

node_image_ref() {
  printf '%s/%s:%s' "$DOCKER_REGISTRY" "$V6_NODE_IMAGE_NAME" "$V6_NODE_IMAGE_TAG"
}

ui_image_ref() {
  printf '%s/%s:%s' "$DOCKER_REGISTRY" "$V6_UI_IMAGE_NAME" "$V6_UI_IMAGE_TAG"
}

find_running_server_container_id() {
  docker ps -qf "label=vantage6-type=server" -f "label=name=$SERVER_NAME" | head -n 1
}

find_all_server_container_ids() {
  docker ps -aqf "label=vantage6-type=server" -f "label=name=$SERVER_NAME" | awk 'NF && !seen[$0]++'
}

report_server_diagnostics() {
  local container_ids
  local container_id

  warn "Server '$SERVER_NAME' did not appear as a running container. Collecting diagnostics."
  docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}' | sed -n '1,120p' >&2 || true

  container_ids="$(find_all_server_container_ids)"
  if [ -z "$container_ids" ]; then
    warn "No server container found for server label filters vantage6-type=server and name=$SERVER_NAME"
    return
  fi

  for container_id in $container_ids; do
    warn "Inspecting server container '$container_id'"
    docker inspect --format 'name={{.Name}} status={{.State.Status}} exit_code={{.State.ExitCode}} error={{.State.Error}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$container_id" >&2 || true
    docker logs "$container_id" >&2 || true
  done
}

setup_venv() {
  ensure_command "$PYTHON_INTERPRETER"

  if [ "${RECREATE_ENV:-false}" = true ] && [ -d "$VENV_PATH" ]; then
    log "Removing existing virtual environment at '$VENV_PATH'"
    rm -rf "$VENV_PATH"
  fi

  if [ ! -d "$VENV_PATH" ]; then
    log "Creating virtual environment at '$VENV_PATH' using '$PYTHON_INTERPRETER'"
    "$PYTHON_INTERPRETER" -m venv "$VENV_PATH"
  fi

  # shellcheck source=/dev/null
  . "$VENV_PATH/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
}

install_dependencies() {
  if [ -f "$REQUIREMENTS_FILE" ] && [ -s "$REQUIREMENTS_FILE" ]; then
    log "Installing Python dependencies from '$REQUIREMENTS_FILE'"
    python -m pip install -r "$REQUIREMENTS_FILE"
  else
    log "No non-empty requirements file found at '$REQUIREMENTS_FILE'; skipping"
  fi

  if [ "$VERSION_VANTAGE6" = "latest" ]; then
    python -m pip install vantage6
    VERSION_VANTAGE6="$(python -m pip show vantage6 | awk '/^Version:/ {print $2}')"
  else
    python -m pip install "vantage6==$VERSION_VANTAGE6"
  fi

  log "Using vantage6 version $VERSION_VANTAGE6"
}

ensure_amd64_emulation() {
  local host_arch="${1:-$(uname -m)}"

  case "$host_arch" in
    x86_64|amd64)
      return
      ;;
  esac

  if [ "${V6_AUTO_INSTALL_BINFMT:-true}" != "true" ]; then
    return
  fi

  if [ -r /proc/sys/fs/binfmt_misc/qemu-x86_64 ]; then
    return
  fi

  warn "Installing qemu-x86_64 binfmt so amd64 Vantage6 images can run on host architecture '$host_arch'"
  docker run --privileged --rm tonistiigi/binfmt --install amd64 >/dev/null
}

probe_image_runnable_on_host() {
  local image_ref="$1"
  local image_label="$2"
  local host_arch
  local probe_output
  local override_probe_output

  host_arch="$(uname -m)"

  case "$host_arch" in
    x86_64|amd64)
      return
      ;;
  esac

  probe_output="$(docker run --rm --entrypoint /bin/sh "$image_ref" -c 'true' 2>&1)" && return

  if [ -n "${DOCKER_DEFAULT_PLATFORM:-}" ]; then
    fail "Local ${image_label} image '$image_ref' is not runnable on host architecture '$host_arch' with DOCKER_DEFAULT_PLATFORM=$DOCKER_DEFAULT_PLATFORM. Docker probe failed with: $probe_output"
  fi

  ensure_amd64_emulation "$host_arch"

  override_probe_output="$(DOCKER_DEFAULT_PLATFORM=linux/amd64 docker run --rm --entrypoint /bin/sh "$image_ref" -c 'true' 2>&1)" && {
    DOCKER_DEFAULT_PLATFORM="linux/amd64"
    export DOCKER_DEFAULT_PLATFORM
    warn "Local ${image_label} image '$image_ref' is amd64-only on host architecture '$host_arch'; enabling DOCKER_DEFAULT_PLATFORM=$DOCKER_DEFAULT_PLATFORM for this run"
    return
  }

  fail "Local ${image_label} image '$image_ref' is not runnable on host architecture '$host_arch'. Docker probe failed with: $probe_output. Retrying with DOCKER_DEFAULT_PLATFORM=linux/amd64 also failed with: $override_probe_output"
}

pull_docker_images() {
  ensure_command docker

  if docker image inspect "$(server_image_ref)" >/dev/null 2>&1; then
    log "Using locally available server image '$(server_image_ref)'"
  else
    log "Pulling server image '$(server_image_ref)'"
    docker pull "$(server_image_ref)"
  fi
  probe_image_runnable_on_host "$(server_image_ref)" "server"

  if docker image inspect "$(node_image_ref)" >/dev/null 2>&1; then
    log "Using locally available node image '$(node_image_ref)'"
  else
    log "Pulling node image '$(node_image_ref)'"
    docker pull "$(node_image_ref)"
  fi
  probe_image_runnable_on_host "$(node_image_ref)" "node"

  if parse_bool "$UI_ENABLED"; then
    if docker image inspect "$(ui_image_ref)" >/dev/null 2>&1; then
      log "Using locally available UI image '$(ui_image_ref)'"
    else
      log "Pulling UI image '$(ui_image_ref)'"
      docker pull "$(ui_image_ref)"
    fi
    probe_image_runnable_on_host "$(ui_image_ref)" "ui"
  fi
}

NODE_NAMES=()
NODE_API_KEYS=()
NODE_DB_URIS=()
NODE_DB_TYPES=()
NODE_DB_LABELS=()
# Optional second (folder) database per node — e.g. a NIfTI slice folder
# mounted read-only alongside the primary CSV manifest. Empty string means
# "no second database for this node".
NODE_EXTRA_DB_URIS=()
NODE_EXTRA_DB_TYPES=()
NODE_EXTRA_DB_LABELS=()

load_node_specs() {
  local specs_file="$NODES_CONFIG"

  [ -f "$specs_file" ] || fail "Node spec file not found: $specs_file"

  NODE_NAMES=()
  NODE_API_KEYS=()
  NODE_DB_URIS=()
  NODE_DB_TYPES=()
  NODE_DB_LABELS=()
  NODE_EXTRA_DB_URIS=()
  NODE_EXTRA_DB_TYPES=()
  NODE_EXTRA_DB_LABELS=()

  local line_no=0
  while IFS='|' read -r raw_name raw_api_key raw_db_uri raw_db_type raw_db_label \
                        raw_extra_db_uri raw_extra_db_type raw_extra_db_label raw_extra; do
    line_no=$((line_no + 1))

    local name api_key db_uri db_type db_label extra_db_uri extra_db_type extra_db_label
    name="$(trim "${raw_name:-}")"

    if [ -z "$name" ]; then
      continue
    fi

    if [[ "$name" == \#* ]]; then
      continue
    fi

    api_key="$(trim "${raw_api_key:-}")"
    db_uri="$(trim "${raw_db_uri:-}")"
    db_type="$(trim "${raw_db_type:-}")"
    db_label="$(trim "${raw_db_label:-}")"
    extra_db_uri="$(trim "${raw_extra_db_uri:-}")"
    extra_db_type="$(trim "${raw_extra_db_type:-}")"
    extra_db_label="$(trim "${raw_extra_db_label:-}")"

    if [ -n "$(trim "${raw_extra:-}")" ]; then
      fail "Invalid node spec format at line $line_no in '$specs_file' (too many columns)"
    fi

    [ -n "$api_key" ] || fail "Missing api_key for node '$name' in '$specs_file' (line $line_no)"

    if [ -z "$db_uri" ]; then
      db_uri="$DATA_DIR_DEFAULT/$name.csv"
    fi

    db_type="${db_type:-csv}"
    db_label="${db_label:-default}"

    if ! looks_like_uri "$db_uri"; then
      db_uri="$(abspath_if_local_path "$db_uri")"
    fi

    if [ -n "$extra_db_uri" ]; then
      extra_db_type="${extra_db_type:-folder}"
      extra_db_label="${extra_db_label:-extra}"
      if ! looks_like_uri "$extra_db_uri"; then
        extra_db_uri="$(abspath_if_local_path "$extra_db_uri")"
      fi
    fi

    NODE_NAMES+=("$name")
    NODE_API_KEYS+=("$api_key")
    NODE_DB_URIS+=("$db_uri")
    NODE_DB_TYPES+=("$db_type")
    NODE_DB_LABELS+=("$db_label")
    NODE_EXTRA_DB_URIS+=("$extra_db_uri")
    NODE_EXTRA_DB_TYPES+=("$extra_db_type")
    NODE_EXTRA_DB_LABELS+=("$extra_db_label")
  done < "$specs_file"

  if [ "${#NODE_NAMES[@]}" -eq 0 ]; then
    fail "No node specs found in '$specs_file'"
  fi
}

validate_node_specs() {
  local i
  local strict_data_checks_enabled=false

  if parse_bool "$STRICT_DATA_CHECKS"; then
    strict_data_checks_enabled=true
  fi

  for i in "${!NODE_NAMES[@]}"; do
    local name db_uri db_type extra_db_uri extra_db_type
    name="${NODE_NAMES[$i]}"
    db_uri="${NODE_DB_URIS[$i]}"
    db_type="${NODE_DB_TYPES[$i]}"
    extra_db_uri="${NODE_EXTRA_DB_URIS[$i]:-}"
    extra_db_type="${NODE_EXTRA_DB_TYPES[$i]:-}"

    if $strict_data_checks_enabled && [ "$db_type" = "csv" ] && ! looks_like_uri "$db_uri"; then
      [ -f "$db_uri" ] || fail "CSV data for node '$name' not found: $db_uri"
    fi

    if [ -n "$extra_db_uri" ] && $strict_data_checks_enabled && [ "$extra_db_type" = "folder" ] \
       && ! looks_like_uri "$extra_db_uri"; then
      [ -d "$extra_db_uri" ] || fail "Extra folder database for node '$name' not found: $extra_db_uri"
    fi
  done
}

print_node_specs() {
  local i
  log "Loaded ${#NODE_NAMES[@]} node specs"
  for i in "${!NODE_NAMES[@]}"; do
    log "- ${NODE_NAMES[$i]} (${NODE_DB_TYPES[$i]}:${NODE_DB_LABELS[$i]}) -> ${NODE_DB_URIS[$i]}"
    if [ -n "${NODE_EXTRA_DB_URIS[$i]:-}" ]; then
      log "  + (${NODE_EXTRA_DB_TYPES[$i]}:${NODE_EXTRA_DB_LABELS[$i]}) -> ${NODE_EXTRA_DB_URIS[$i]}"
    fi
  done
}

prepare_runtime_dirs() {
  mkdir -p "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR/nodes"
}

generate_entities_file() {
  local output_file="$ENTITIES_FILE"
  local output_dir
  output_dir="$(dirname "$output_file")"
  mkdir -p "$output_dir"

  {
    echo "collaborations:"
    echo "- encrypted: false"
    echo "  name: $COLLABORATION_NAME"
    echo "  participants:"

    local i
    for i in "${!NODE_NAMES[@]}"; do
      echo "  - api_key: ${NODE_API_KEYS[$i]}"
      echo "    name: ${NODE_NAMES[$i]}"
    done
    echo "  tasks: ['hello-world']"

    echo "nodes: []"
    echo "organizations:"

    for i in "${!NODE_NAMES[@]}"; do
      local name username
      name="${NODE_NAMES[$i]}"
      username="${name}-user"

      echo "- name: $name"
      if [ "$i" = "0" ]; then
        echo "  make_admin: true"
      fi
      echo "  address1: ${name} street 1"
      echo "  address2: ''"
      echo "  country: Unknown"
      echo "  domain: ${name}.local"
      echo "  public_key: ''"
      echo "  users:"
      echo "  - email: ${username}@example.org"
      echo "    username: $username"
      echo "    firstname: ${name}"
      echo "    lastname: User"
      echo "    password: ${name}-password"
      if [ "$i" = "0" ]; then
        echo "  - email: dev-admin@example.org"
        echo "    username: dev_admin"
        echo "    firstname: admin"
        echo "    lastname: robot"
        echo "    password: password"
      fi
      echo "  zipcode: '0000AA'"
    done
  } > "$output_file"

  log "Generated entities file at '$output_file'"
}

build_node_config() {
  local node_name="$1"
  local api_key="$2"
  local db_uri="$3"
  local db_type="$4"
  local db_label="$5"
  local output_file="$6"
  # Optional second (folder) database — e.g. a NIfTI slice folder mounted
  # read-only alongside the primary CSV manifest. Pass empty strings to omit.
  local extra_db_uri="${7:-}"
  local extra_db_type="${8:-folder}"
  local extra_db_label="${9:-extra}"
  local node_image
  node_image="$(node_image_ref)"

  # Extract port from SERVER_URL and build a node-facing URL
  local server_port node_server_url
  server_port="$(printf '%s' "$SERVER_URL" | grep -oE '[0-9]+$')"
  server_port="${server_port:-5070}"
  node_server_url="${SERVER_URL%:[0-9]*}"
  node_server_url="${node_server_url/localhost/host.docker.internal}"
  node_server_url="${node_server_url/127.0.0.1/host.docker.internal}"

  cat > "$output_file" <<EOL
api_key: $api_key
api_path: $API_PATH
databases:
  - label: $db_label
    type: $db_type
    uri: $db_uri
    mount_mode: ro
EOL

  if [ -n "$extra_db_uri" ]; then
    cat >> "$output_file" <<EOL
  - label: $extra_db_label
    type: $extra_db_type
    uri: $extra_db_uri
    mount_mode: ro
EOL
  fi

  cat >> "$output_file" <<EOL
encryption:
  enabled: false
  private_key: ''
policies:
  allowed_algorithms: []
  require_algorithm_pull: false
images:
  node: $node_image
logging:
  backup_count: 5
  datefmt: '%Y-%m-%d %H:%M:%S'
  format: '%(asctime)s - %(name)-14s - %(levelname)-8s - %(message)s'
  level: DEBUG
  loggers:
    - level: warning
      name: urllib3
    - level: warning
      name: requests
    - level: warning
      name: engineio.client
    - level: warning
      name: docker.utils.config
    - level: warning
      name: docker.auth
  max_size: 1024
  use_console: true
port: '$server_port'
server_url: $node_server_url
task_dir: ./$node_name/tasks
share_config: false
share_algorithm_logs: false
run_context_file: true
prometheus:
  enabled: false
node_extra_hosts:
  host.docker.internal: host-gateway
EOL
}

start_server() {
  local keep_flag=()
  if parse_bool "$KEEP_CONTAINERS"; then
    keep_flag+=(--keep)
    log "Container auto-removal disabled (KEEP_CONTAINERS=$KEEP_CONTAINERS)"
  fi

  log "Starting server '$SERVER_NAME' using config '$SERVER_CONFIG'"
  v6 server start --user "${keep_flag[@]}" -c "$SERVER_CONFIG" --image "$(server_image_ref)"
}

import_entities() {
  local server_container_id
  local attempts=10
  local delay_seconds=1

  while [ "$attempts" -gt 0 ]; do
    server_container_id="$(find_running_server_container_id)"
    if [ -n "$server_container_id" ]; then
      break
    fi
    attempts=$((attempts - 1))
    sleep "$delay_seconds"
  done

  if [ -z "$server_container_id" ]; then
    report_server_diagnostics
    fail "Could not find running server container for '$SERVER_NAME'"
  fi

  # Wait for the server API to finish initializing
  local health_url="${SERVER_URL}${API_PATH}/health"
  log "Waiting for server API to be ready at '$health_url'"
  if curl -sf --retry 30 --retry-delay 2 --retry-connrefused "$health_url" >/dev/null 2>&1; then
    log "Server API is ready"
  else
    warn "Server API did not become ready after 60s; proceeding with import anyway"
  fi

  log "Importing entities from '$ENTITIES_FILE'"
  docker cp "$ENTITIES_FILE" "$server_container_id":/entities.yaml

  attempts=5
  while [ "$attempts" -gt 0 ]; do
    if docker exec "$server_container_id" /usr/local/bin/vserver-local import --config /mnt/config.yaml /entities.yaml; then
      return
    fi
    attempts=$((attempts - 1))
    sleep 2
  done

  fail "Could not import entities into server '$SERVER_NAME' after multiple attempts"
}

start_nodes() {
  local i
  local keep_flag=()
  if parse_bool "$KEEP_CONTAINERS"; then
    keep_flag+=(--keep)
  fi

  for i in "${!NODE_NAMES[@]}"; do
    local node_name api_key db_uri db_type db_label node_config_file
    local extra_db_uri extra_db_type extra_db_label
    node_name="${NODE_NAMES[$i]}"
    api_key="${NODE_API_KEYS[$i]}"
    db_uri="${NODE_DB_URIS[$i]}"
    db_type="${NODE_DB_TYPES[$i]}"
    db_label="${NODE_DB_LABELS[$i]}"
    extra_db_uri="${NODE_EXTRA_DB_URIS[$i]:-}"
    extra_db_type="${NODE_EXTRA_DB_TYPES[$i]:-}"
    extra_db_label="${NODE_EXTRA_DB_LABELS[$i]:-}"
    node_config_file="$GENERATED_DIR/nodes/${node_name}.yaml"

    build_node_config "$node_name" "$api_key" "$db_uri" "$db_type" "$db_label" "$node_config_file" \
      "$extra_db_uri" "$extra_db_type" "$extra_db_label"

    log "Starting node '$node_name'"
    v6 node start --user "${keep_flag[@]}" -c "$node_config_file" --image "$(node_image_ref)"
  done
}

start_ui() {
  if ! parse_bool "$UI_ENABLED"; then
    log "UI is disabled (UI_ENABLED=$UI_ENABLED)"
    return
  fi

  log "Starting UI container on port $UI_PORT"
  docker rm -f vantage6-ui >/dev/null 2>&1 || true

  docker run --rm -d \
    --name vantage6-ui \
    -p "$UI_PORT:$UI_PORT" \
    -e "SERVER_URL=$SERVER_URL" \
    -e "API_PATH=$API_PATH" \
    "$(ui_image_ref)"
}

open_browser() {
  local url="$1"
  case "$(uname -s)" in
    Darwin) open "$url" ;;
    Linux)
      if grep -i microsoft /proc/sys/kernel/osrelease >/dev/null 2>&1 && command -v wslview >/dev/null 2>&1; then
        wslview "$url"
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url"
      else
        warn "Automatic browser launch not supported here. Open '$url' manually."
      fi
      ;;
    *) warn "Automatic browser launch not supported on this OS. Open '$url' manually." ;;
  esac
}

preflight_checks() {
  ensure_command docker
  ensure_command "$PYTHON_INTERPRETER"

  [ -f "$SERVER_CONFIG" ] || fail "Server config not found: $SERVER_CONFIG"
  [ -f "$NODES_CONFIG" ] || fail "Node spec file not found: $NODES_CONFIG"

  docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"

  case "$(uname -m)" in
    x86_64|amd64)
      ;;
    *)
      warn "This host is not amd64. Published GHCR Vantage6 infra images are only signoff-tested on amd64; use amd64 CI for authoritative validation."
      ;;
  esac
}

stop_nodes() {
  local i

  for ((i=${#NODE_NAMES[@]}-1; i>=0; i--)); do
    local node_name
    node_name="${NODE_NAMES[$i]}"
    log "Stopping node '$node_name'"
    v6 node stop --user -n "$node_name" >/dev/null 2>&1 || true
  done
}

stop_server() {
  log "Stopping server '$SERVER_NAME'"
  v6 server stop --user -n "$SERVER_NAME" >/dev/null 2>&1 || true
}

remove_containers() {
  log "Removing UI container (if any)"
  docker rm -f vantage6-ui >/dev/null 2>&1 || true

  local server_containers
  server_containers="$(find_all_server_container_ids)"
  if [ -n "$server_containers" ]; then
    docker rm -f $server_containers >/dev/null 2>&1 || true
  fi

  local i
  for i in "${!NODE_NAMES[@]}"; do
    local node_container
    node_container="vantage6-${NODE_NAMES[$i]}-user"
    docker rm -f "$node_container" >/dev/null 2>&1 || true
  done
}

cleanup_local_state() {
  if ! parse_bool "$CLEAN_LOCAL_STATE"; then
    log "Skipping local state cleanup (CLEAN_LOCAL_STATE=$CLEAN_LOCAL_STATE)"
    return
  fi

  case "$(uname -s)" in
    Darwin)
      rm -rf "$HOME/Library/Application Support/vantage6/node" \
             "$HOME/Library/Application Support/vantage6/server"
      ;;
    Linux)
      rm -rf "$HOME/.local/share/vantage6/node" \
             "$HOME/.local/share/vantage6/server" \
             "$HOME/.cache/vantage6"
      ;;
    *)
      warn "Skipping OS-level cleanup for unsupported platform"
      ;;
  esac
}

expected_container_count() {
  local count=1
  count=$((count + ${#NODE_NAMES[@]}))
  if parse_bool "$UI_ENABLED"; then
    count=$((count + 1))
  fi

  printf '%s' "$count"
}

check_container_presence() {
  ensure_command docker
  docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"

  local missing=0
  local running
  running="$(docker ps --format '{{.Names}}')"

  if [ -z "$(find_running_server_container_id)" ]; then
    warn "Missing running server container for label filters vantage6-type=server and name=$SERVER_NAME"
    missing=1
  fi

  local i
  for i in "${!NODE_NAMES[@]}"; do
    local expected_node
    expected_node="vantage6-${NODE_NAMES[$i]}-user"
    if ! echo "$running" | grep -Fxq "$expected_node"; then
      warn "Missing node container '$expected_node'"
      missing=1
    fi
  done

  if parse_bool "$UI_ENABLED"; then
    if ! echo "$running" | grep -Fxq "vantage6-ui"; then
      warn "Missing UI container 'vantage6-ui'"
      missing=1
    fi
  fi

  if [ "$missing" -ne 0 ]; then
    return 1
  fi

  return 0
}
