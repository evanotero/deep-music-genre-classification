#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -d "$DIR/.venv" ]; then
    source "$DIR/.venv/bin/activate"
fi
exec python "$DIR/server.py" "$@"
