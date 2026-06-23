#!/bin/bash

trap "kill 0 2>/dev/null; wait 2>/dev/null" EXIT

echo "[backend] Starting Flask..."
./venv/bin/python -m backend.app &
BACKEND_PID=$!

echo "[backend] Waiting for port 5001..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:5001 >/dev/null 2>&1; then
        echo "[backend] Ready."
        break
    fi
    sleep 1
done
if ! curl -s http://127.0.0.1:5001 >/dev/null 2>&1; then
    echo "[backend] Failed to start"
    exit 1
fi

echo "[frontend] Starting frontend dev server..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# macOS ships bash 3.2, which lacks `wait -n` (bash 4.3+) — poll instead.
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 1
done
echo "[run] One of the servers exited."
