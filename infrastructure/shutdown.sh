#!/bin/bash

# Default values for environment variables (can be overridden in CI or locally)
VENV_PATH=${VENV_PATH:-"./venv"}
PYTHON_INTERPRETER=${PYTHON_INTERPRETER:-"python3.11"}

# Expand the tilde (~) in VENV_PATH
VENV_PATH="${VENV_PATH/#\~/$HOME}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment at '$VENV_PATH' does not exist."
    exit 1
fi

. "$VENV_PATH/bin/activate"

stop_node() {
    local node_name="$1"
    echo "Stopping node '$node_name'..."
    v6 node stop --user -n "$node_name"
}

stop_server() {
    local server_name="$1"
    echo "Stopping server '$server_name'..."
    v6 server stop --user -n "$server_name"
}

cleanup_docker() {
    echo "Stopping and removing the UI Docker container..."
    docker stop vantage6-ui
    docker rm -f vantage6-demoserver-user-ServerType.V6SERVER vantage6-alpha-user vantage6-beta-user vantage6-gamma-user
}

cleanup_files() {
    local base_dir="$1"
    echo "Removing directory '$base_dir'..."
    rm -Rf "$base_dir"
}

# Stop services
stop_node "gamma"
stop_node "beta"
stop_node "alpha"
stop_server "demoserver"

# Cleanup Docker and files
cleanup_docker

# Depending on the OS, adjust the cleanup path
case "$(uname -s)" in
    Darwin)  # macOS paths
        cleanup_files "$HOME/Library/Application Support/vantage6/node"
        cleanup_files "$HOME/Library/Application Support/vantage6/server"
        ;;
    Linux)  # Linux paths
        cleanup_files "$HOME/.local/share/vantage6/node"
        cleanup_files "$HOME/.local/share/vantage6/server"
        cleanup_files "$HOME/.cache/vantage6"
        ;;
    *) echo "Unsupported OS for cleanup operations." ;;
esac

deactivate

echo "Shutdown complete."
