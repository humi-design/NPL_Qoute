"""Live MySQL/MariaDB migration test (opt-in).

The static guard in ``test_schema_consistency.py`` reads the SQL files as text.
This test goes further: it runs the real ``schema.sql`` / ``migrate.sql``
against a live MariaDB and asserts the resulting database can serve every model
query. That is what caught ``process_template_steps.sequence`` and
``quotation_operations.process_id`` being absent from both files, which broke a
fresh install at runtime with "Unknown column".

It is skipped unless ``NPL_TEST_MYSQL_URL`` points at a throwaway server, e.g.

    docker run -d --name npl-mariadb -e MARIADB_ROOT_PASSWORD=rootpw \\
        -p 13306:3306 mariadb:11.4
    NPL_TEST_MYSQL_URL='mysql+pymysql://root:rootpw@127.0.0.1:13306' \\
        .venv/bin/python -m pytest tests/test_migration_mysql.py

The URL is a base URL (no database name); the test creates and drops its own
databases, so it never touches real data. Application imports happen in a
subprocess with ``DATABASE_URL`` already set, because ``app.config`` resolves
the URI at import time.
"""
import json
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.environ.get('NPL_TEST_MYSQL_URL')

pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason='set NPL_TEST_MYSQL_URL to run the live MySQL migration test')


# Runs inside a subprocess: compare the live schema against the model metadata.
_DRIFT_PROBE = '''
import json
from app.extensions import create_app, db
from sqlalchemy import inspect
app = create_app("development")
with app.app_context():
    insp = inspect(db.engine)
    tables = set(insp.get_table_names())
    cols = {t: {c["name"] for c in insp.get_columns(t)} for t in tables}
    missing = []
    for t in db.metadata.sorted_tables:
        if t.name not in tables:
            missing.append("TABLE " + t.name)
            continue
        for c in sorted({c.name for c in t.columns} - cols[t.name]):
            missing.append(t.name + "." + c)
print("RESULT:" + json.dumps(missing))
'''


# Runs inside a subprocess: the real failure mode, an ORM query on a model whose
# column is absent raises OperationalError "Unknown column".
_ORM_PROBE = '''
import json
from app.extensions import create_app, db
from app.models import ProcessTemplateStep, QuotationOperation
app = create_app("development")
with app.app_context():
    counts = {"process_template_steps": ProcessTemplateStep.query.count(),
              "quotation_operations": QuotationOperation.query.count()}
    broken = []
    for table in db.metadata.sorted_tables:
        try:
            db.session.execute(table.select().limit(1)).fetchall()
        except Exception as exc:
            broken.append(table.name + ": " + str(exc).splitlines()[0][:80])
    db.engine.dispose()
print("RESULT:" + json.dumps({"counts": counts, "broken": broken}))
'''


def _mysql_cmd(database):
    user = re.search(r'://([^:]+):', BASE_URL).group(1)
    password = re.search(r'://[^:]+:([^@]+)@', BASE_URL).group(1)
    host = re.search(r'@([^:/]+)', BASE_URL).group(1)
    port = re.search(r'@[^:/]+:(\d+)', BASE_URL)
    cmd = ['mysql', '-u', user, f'-p{password}', '-h', host]
    if port:
        cmd += ['-P', port.group(1)]
    return cmd + [database]


def _run_sql(database, path):
    """Feed a .sql file to the server, stopping on the first error.

    ``schema.sql`` begins with ``CREATE DATABASE ... humijxhw_npl_web`` and
    ``USE humijxhw_npl_web`` for the phpMyAdmin workflow. Those lines are
    neutralised here so the script is applied to the throwaway test database
    instead of the real one.
    """
    sql = open(path).read()
    sql = re.sub(r'^\s*(CREATE\s+DATABASE|USE)\b.*$', '', sql,
                 flags=re.I | re.M)
    proc = subprocess.run(_mysql_cmd(database), input=sql,
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f'{os.path.basename(path)} failed:\n{proc.stdout}\n{proc.stderr}')


def _recreate(database):
    import sqlalchemy
    # Connect without selecting the target database - it does not exist yet.
    engine = sqlalchemy.create_engine(BASE_URL, isolation_level='AUTOCOMMIT')
    with engine.connect() as conn:
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS {database}')
        conn.exec_driver_sql(f'CREATE DATABASE {database}')
    engine.dispose()


def _probe(database, code):
    """Run a probe script with DATABASE_URL set before the app is imported."""
    env = dict(os.environ, DATABASE_URL=f'{BASE_URL}/{database}')
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                          capture_output=True, text=True, env=env)
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT:')]
    assert marker, f'probe produced no result:\n{proc.stdout}\n{proc.stderr}'
    return json.loads(marker[-1][len('RESULT:'):])


def _assert_no_drift(database):
    missing = _probe(database, _DRIFT_PROBE)
    assert not missing, f'{database}: model objects missing from SQL files: {missing}'


def test_schema_sql_is_repeatable():
    """schema.sql must apply cleanly twice, then migrate.sql reaches parity.

    schema.sql is the legacy base schema plus seed rows; migrate.sql is the
    authoritative upgrader that adds every model column. The documented install
    runs them in that order, so parity is asserted after both.
    """
    _recreate('npl_schema_check')
    schema = os.path.join(REPO_ROOT, 'schema.sql')
    _run_sql('npl_schema_check', schema)   # fresh install
    _run_sql('npl_schema_check', schema)   # re-run must not fail on seeds
    _run_sql('npl_schema_check', os.path.join(REPO_ROOT, 'migrate.sql'))
    _assert_no_drift('npl_schema_check')


def test_migrate_sql_is_idempotent_and_matches_models():
    _recreate('npl_migrate_check')
    path = os.path.join(REPO_ROOT, 'migrate.sql')
    _run_sql('npl_migrate_check', path)   # fresh install
    _run_sql('npl_migrate_check', path)   # second run must be a no-op
    _run_sql('npl_migrate_check', path)   # third run too
    _assert_no_drift('npl_migrate_check')


def test_orm_queries_work_after_migration():
    _recreate('npl_orm_check')
    _run_sql('npl_orm_check', os.path.join(REPO_ROOT, 'migrate.sql'))
    result = _probe('npl_orm_check', _ORM_PROBE)
    assert result['counts'] == {'process_template_steps': 0,
                                'quotation_operations': 0}
    assert not result['broken'], (
        'tables not queryable after migration: ' + '; '.join(result['broken']))
