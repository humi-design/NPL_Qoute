#!/bin/bash
# Bring the database up to date with the current code.
#
# Finds the application's virtualenv for you, so there is no separate
# activation step to get wrong, and so it works from a cron job. Safe to run
# repeatedly: it only creates missing tables and adds missing columns.
#
# Usage:
#     ./run_upgrade.sh
#
# Run it from anywhere; it locates the application root itself.

cd "$(dirname "$0")" || exit 1
APP_DIR="$(basename "$PWD")"
PARENT_DIR="$(basename "$(dirname "$PWD")")"

find_python() {
    # 1. An already-activated virtualenv.
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        printf '%s' "$VIRTUAL_ENV/bin/python"
        return
    fi
    # 2. A virtualenv inside the project (local development).
    for candidate in "$PWD/.venv/bin/python" "$PWD/venv/bin/python"; do
        if [ -x "$candidate" ]; then
            printf '%s' "$candidate"
            return
        fi
    done
    # 3. cPanel layouts: ~/virtualenv/<app>/<python>/bin/python and
    #    ~/virtualenv/<domain>/<app>/<python>/bin/python
    for candidate in "$HOME/virtualenv/$PARENT_DIR/$APP_DIR"/*/bin/python \
                     "$HOME/virtualenv/$APP_DIR"/*/bin/python \
                     "$HOME"/virtualenv/*/"$APP_DIR"/*/bin/python; do
        if [ -x "$candidate" ]; then
            printf '%s' "$candidate"
            return
        fi
    done
    # 4. Whatever Python is on the PATH.
    command -v python3 || command -v python
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ]; then
    echo "Could not find a Python interpreter."
    echo "Activate the application's virtualenv first, then re-run this script."
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Application root: $PWD"
echo

"$PYTHON" upgrade_db.py
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Restart the application in cPanel (Setup Python App) if it is running."
else
    echo "Upgrade failed (exit $STATUS). The message above says why."
fi
exit $STATUS
