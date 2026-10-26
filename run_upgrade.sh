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
    # Every candidate is validated by actually executing it. On CloudLinux and
    # cPanel, virtualenv's bin/python is a *bash wrapper* around the real
    # interpreter, not a binary; if anything ever hands that wrapper to Python
    # as the script to run, Python parses the shell script and dies with
    # "SyntaxError: invalid syntax" on the `[[ $EUID -eq 0 ]]` line. Testing the
    # candidate here means a wrapper that cannot run is skipped, not trusted.
    _is_python() {
        [ -x "$1" ] && "$1" -c 'import sys' >/dev/null 2>&1
    }

    # 1. An already-activated virtualenv.
    if [ -n "${VIRTUAL_ENV:-}" ] && _is_python "$VIRTUAL_ENV/bin/python"; then
        printf '%s' "$VIRTUAL_ENV/bin/python"
        return
    fi
    # 2. A virtualenv inside the project (local development).
    for candidate in "$PWD/.venv/bin/python3" "$PWD/.venv/bin/python" \
                     "$PWD/venv/bin/python3" "$PWD/venv/bin/python"; do
        if _is_python "$candidate"; then
            printf '%s' "$candidate"
            return
        fi
    done
    # 3. cPanel layouts: ~/virtualenv/<app>/<python>/bin/python and
    #    ~/virtualenv/<domain>/<app>/<python>/bin/python
    for candidate in "$HOME/virtualenv/$PARENT_DIR/$APP_DIR"/*/bin/python \
                     "$HOME/virtualenv/$APP_DIR"/*/bin/python \
                     "$HOME"/virtualenv/*/"$APP_DIR"/*/bin/python; do
        if _is_python "$candidate"; then
            printf '%s' "$candidate"
            return
        fi
    done
    # 4. Whatever Python is on the PATH.
    for candidate in python3 python; do
        resolved="$(command -v "$candidate" 2>/dev/null)" || continue
        if _is_python "$resolved"; then
            printf '%s' "$resolved"
            return
        fi
    done
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ]; then
    echo "Could not find a working Python interpreter."
    echo "Activate the application's virtualenv first, then re-run this script:"
    echo "    source \$HOME/virtualenv/$(dirname "$PWD" | xargs basename)/$APP_DIR/3.9/bin/activate"
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Application root: $PWD"
echo

# Pass exactly one script path. Duplicating it (e.g. pasting the interpreter
# path in front of a command that already names it) makes Python try to run the
# interpreter itself as a script.
"$PYTHON" "$PWD/upgrade_db.py"
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
    echo "Done. Restart the application in cPanel (Setup Python App) if it is running."
else
    echo "Upgrade failed (exit $STATUS). The message above says why."
fi
exit $STATUS
