#!/usr/bin/env bash
# Launches the Rory desktop widget. Assumes .venv exists (see README setup).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python -m rory.ui.widget
