#!/bin/bash
set -euo pipefail

LABEL="dev.logan.logand"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
APP_SUPPORT="$HOME/Library/Application Support/Logan"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
rm -f "$PLIST"
rm -f "$APP_SUPPORT/bin/logand"

echo "Logan daemon removed. Prompt caches and logs were left intact."
