#!/bin/bash

# Activate the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "Activated venv: $(which python)"
else
    echo "WARNING: No .venv found at $SCRIPT_DIR/.venv — using system Python"
fi

# CHARM 3.0: make sure every principal has its signing key (does nothing if they exist).
python "$SCRIPT_DIR/charm_guard.py" init-keys > /dev/null || { echo "ERROR: could not create agent keys"; exit 1; }

# Dashboard port comes from .env (WEB_PORT) so there is one source of truth.
WEB_PORT=$(grep -E "^WEB_PORT=" "$SCRIPT_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' \r')
WEB_PORT=${WEB_PORT:-9091}

# Kill any processes on our ports before starting.
# NOTE: this only ever targets THIS app's ports. If you change WEB_PORT, make sure
# the new port isn't used by something else you care about — it will be killed.
for port in $WEB_PORT 7060 7002 7003 7001 7004; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "Killing stale process on port $port (pid $pid)"
        kill $pid 2>/dev/null
        sleep 1
    fi
done

# Mutual TLS flags for uvicorn, and the client side of the same for curl.
tls_args() { python "$SCRIPT_DIR/charm_guard.py" tls-args "$1"; }
TLS="$SCRIPT_DIR/keys/tls"
CURL_TLS=(--cacert "$TLS/ca.crt" --cert "$TLS/Dashboard.crt" --key "$TLS/Dashboard.key")

# Function to check if a server is ready
check_server_ready() {
    local host=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    echo "Checking if server at ${host}:${port} is ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "${CURL_TLS[@]}" "https://${host}:${port}/.well-known/agent.json" > /dev/null 2>&1; then
            echo "Server at ${host}:${port} is ready (attempt ${attempt}/${max_attempts})"
            return 0
        fi

        echo "Attempt ${attempt}/${max_attempts}: Server at ${host}:${port} not ready yet, waiting..."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo "ERROR: Server at ${host}:${port} did not become ready after ${max_attempts} attempts"
    return 1
}

# Show ps for the server processes
show_agent_processes() {
    echo
    echo "Current Server Agent Processes:"
    echo "----------------------------------------"

    # Show ps header
    ps -ef | head -1

    # Read and display each PID
    for pid_file in *.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if [ -n "$pid" ]; then
                ps -ef | grep -E "^\s*[^ ]+\s+${pid}\s" || echo "# Process $pid ($pid_file) not found"
            fi
        fi
    done
}

# Start Web Dashboard first (so trace collector is available)
echo "Starting WEB DASHBOARD (webagent_server) on port $WEB_PORT..."
cd webagent
uvicorn webagent_server:app --host 127.0.0.1 --port "$WEB_PORT" > ../webagent.log 2>&1 &
echo $! > ../webagent.pid
cd ..
sleep 2
if curl -s "http://127.0.0.1:$WEB_PORT/health" > /dev/null 2>&1; then
    echo "Web dashboard is up on http://localhost:$WEB_PORT"
else
    echo "WARNING: Web dashboard may not have started properly"
fi

echo "Starting REGISTRY (svr_registry_agent) agent on port 7060..."
uvicorn svr_registry_agent:app --host 127.0.0.1 --port 7060 $(tls_args RegistryAgent) > registry_agent.log 2>&1 &
echo $! > registry_agent.pid

if check_server_ready "127.0.0.1" "7060"; then
    echo "Registry is up, starting next server..."
fi

echo "Starting CHAT (svr_chat_agent) agent on port 7002"
uvicorn svr_chat_agent:app --host 127.0.0.1 --port 7002 $(tls_args ChatAgent) > chat_agent.log 2>&1 &
echo $! > chat_agent.pid

if check_server_ready "127.0.0.1" "7002"; then
    echo "Chat agent is up!"
fi

echo "Starting SMS (svr_sms_agent) agent on port 7003..."
uvicorn svr_sms_agent:app --host 127.0.0.1 --port 7003 $(tls_args SMSAgent) > sms_agent.log 2>&1 &
echo $! > sms_agent.pid

if check_server_ready "127.0.0.1" "7003"; then
    echo "SMS agent is up!"
fi

echo "Starting EMAIL (svr_email_agent) agent on port 7001..."
uvicorn svr_email_agent:app --host 127.0.0.1 --port 7001 $(tls_args EmailAgent) > email_agent.log 2>&1 &
echo $! > email_agent.pid

if check_server_ready "127.0.0.1" "7001"; then
    echo "Email agent is up!"
fi

echo "Starting WEB (svr_web_agent) agent on port 7004..."
uvicorn svr_web_agent:app --host 127.0.0.1 --port 7004 $(tls_args WebAgent) > web_agent.log 2>&1 &
echo $! > web_agent.pid

if check_server_ready "127.0.0.1" "7004"; then
    echo "Web agent is up!"
fi

echo ""
echo "All agents started successfully."
echo "Dashboard: http://localhost:$WEB_PORT"
echo ""

# Now show the processes.
show_agent_processes
