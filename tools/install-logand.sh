#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_SUPPORT="$HOME/Library/Application Support/Logan"
BIN_DIR="$APP_SUPPORT/bin"
LOG_DIR="$HOME/Library/Logs/Logan"
PLIST="$HOME/Library/LaunchAgents/dev.logan.logand.plist"
LABEL="dev.logan.logand"
PORT="${LOGAN_DAEMON_PORT:-11435}"
MODEL_DIR="${LOGAN_MODEL_DIR:-$HOME/models}"
HOT_BYTES="${LOGAN_HOT_PREFIX_CACHE_BYTES:-536870912}"

cd "$ROOT"
cargo build --release -p logan-chat --bin logand

mkdir -p "$BIN_DIR" "$LOG_DIR" "$(dirname "$PLIST")"
install -m 0755 "$ROOT/target/release/logand" "$BIN_DIR/logand"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN_DIR/logand</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>$PORT</string>
    <string>--model-dir</string>
    <string>$MODEL_DIR</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>$HOME</string>
    <key>LOGAN_MODEL_DIR</key>
    <string>$MODEL_DIR</string>
    <key>LOGAN_HOT_PREFIX_CACHE_BYTES</key>
    <string>$HOT_BYTES</string>
    <key>RUST_BACKTRACE</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>3</integer>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/logand.out.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/logand.err.log</string>
</dict>
</plist>
EOF

DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$LABEL"

printf 'Logan daemon installed and started.\nDashboard: http://127.0.0.1:%s/\nLogs: %s\n' "$PORT" "$LOG_DIR"
