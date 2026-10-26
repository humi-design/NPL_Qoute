"""Guards for run_upgrade.sh, the operator-facing upgrade wrapper.

The wrapper exists because the documented two-line upgrade command
(``source .../activate`` then ``python upgrade_db.py``) is easy to paste as one
line, and because operators should not have to work out which interpreter to
call. It must therefore (a) always hand Python exactly one script path, and
(b) never trust an interpreter it has not executed.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, 'run_upgrade.sh')

pytestmark = pytest.mark.skipif(
    shutil.which('bash') is None or sys.platform == 'win32',
    reason='bash is required')


def _read():
    with open(SCRIPT) as handle:
        return handle.read()


def _clean_env(home):
    """Environment with DB settings stripped.

    Other test modules set DATABASE_URL process-wide at import time, and
    load_dotenv() deliberately will not override an already-set environment
    variable. Inheriting it would mask whether the wrapper really read the
    project's .env.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ('DATABASE_URL', 'DB_HOST', 'DB_NAME', 'DB_USER',
                        'DB_PASSWORD', 'DB_PORT', 'AUTO_UPGRADE_DB',
                        'FLASK_ENV', 'SECRET_KEY')}
    env['HOME'] = str(home)
    return env


def test_wrapper_is_executable():
    assert os.access(SCRIPT, os.X_OK), 'run_upgrade.sh must be executable'


def test_wrapper_parses_as_bash():
    result = subprocess.run(['bash', '-n', SCRIPT],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wrapper_invokes_python_with_exactly_one_script_path():
    """`"$PYTHON" "$PWD/upgrade_db.py"` — never a bare `"$PYTHON" upgrade_db.py`
    plus a duplicated interpreter path, which makes Python parse the
    interpreter itself as a script."""
    source = _read()
    assert '"$PYTHON" "$PWD/upgrade_db.py"' in source, \
        'the upgrade must be invoked with an absolute script path'


def test_wrapper_validates_interpreters_before_use():
    """On CloudLinux/cPanel, venv bin/python is a bash wrapper, not a binary.

    Running `venv/bin/python venv/bin/python upgrade_db.py` makes Python read
    that shell script and fail with "SyntaxError: invalid syntax" on its
    `[[ $EUID -eq 0 ]]` line. The wrapper must execute each candidate before
    trusting it.
    """
    source = _read()
    assert '_is_python()' in source, 'candidates must be executed before use'
    assert '"$1" -c' in source, 'the check must actually run the interpreter'
    # Every candidate path must go through the check.
    assert 'if _is_python "$candidate"' in source
    assert '_is_python "$VIRTUAL_ENV/bin/python"' in source


def test_wrapper_finds_a_working_interpreter(tmp_path):
    """End to end: with a venv containing a bash-wrapper `python`, the wrapper
    must still pick a real interpreter and complete the upgrade."""
    app = tmp_path / 'nirmalprecision.com' / 'RFQ'
    shutil.copytree(REPO_ROOT, app,
                    ignore=shutil.ignore_patterns('.git', '.venv', '.env',
                                                  '__pycache__', '*.pyc'))
    (app / '.env').write_text(
        f'DATABASE_URL=sqlite:///{tmp_path / "w.db"}\nSECRET_KEY=wrapper\n')

    venv_bin = tmp_path / 'virtualenv' / 'nirmalprecision.com' / 'RFQ' / \
        '3.9' / 'bin'
    venv_bin.mkdir(parents=True)
    wrapper = venv_bin / 'python'
    wrapper.write_text(
        '#!/bin/bash\n'
        'EUID_CHECK=0\n'
        'if [[ $EUID -eq 0 ]]; then\n'
        '    EUID_CHECK=1\n'
        'fi\n'
        f'exec {sys.executable} "$@"\n')
    wrapper.chmod(0o755)

    result = subprocess.run(
        ['bash', str(app / 'run_upgrade.sh')],
        capture_output=True, text=True, cwd=str(tmp_path),
        env=_clean_env(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SyntaxError' not in result.stderr, result.stderr
    # It must have upgraded the database the .env pointed at.
    assert 'table' in result.stdout.lower(), result.stdout
    assert 'Traceback' not in result.stderr, result.stderr


def test_wrapper_anchors_env_to_project_root(tmp_path):
    """Running from another directory must still read the app's own .env.

    Passenger and cron do not guarantee the working directory is the
    application root; reading the wrong .env would silently target a different
    database.
    """
    app = tmp_path / 'nirmalprecision.com' / 'RFQ'
    shutil.copytree(REPO_ROOT, app,
                    ignore=shutil.ignore_patterns('.git', '.venv', '.env',
                                                  '__pycache__', '*.pyc'))
    db = tmp_path / 'anchored.db'
    (app / '.env').write_text(
        f'DATABASE_URL=sqlite:///{db}\nSECRET_KEY=anchored\n')
    venv_bin = tmp_path / 'virtualenv' / 'nirmalprecision.com' / 'RFQ' / \
        '3.9' / 'bin'
    venv_bin.mkdir(parents=True)
    wrapper = venv_bin / 'python'
    wrapper.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    wrapper.chmod(0o755)

    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    result = subprocess.run(
        ['bash', str(app / 'run_upgrade.sh')],
        capture_output=True, text=True, cwd=str(elsewhere),
        env=_clean_env(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert str(db) in result.stdout, result.stdout
    assert db.exists(), 'the .env from the project root was not used'
