"""Schema consistency guard.

The application runs on the SQLAlchemy models, but a fresh install is created
from ``schema.sql`` and an existing database is upgraded by ``migrate.sql``.
Those two files have drifted from the models before (missing ``currencies.is_base``,
``allowances.allowance_code``/``value``, ``quotation_operations`` costing columns),
which silently breaks a fresh MySQL install at runtime.

These tests fail loudly if a model column is not covered by ``schema.sql`` or by
an idempotent ``ALTER TABLE ... ADD COLUMN`` in ``migrate.sql``.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_columns(app):
    from sqlalchemy import inspect as sqla_inspect
    with app.app_context():
        db.create_all()
        insp = sqla_inspect(db.engine)
        return {t: {c['name'] for c in insp.get_columns(t)}
                for t in insp.get_table_names()}


def _parse_create_tables(path):
    sql = open(path).read()
    tables = {}
    for m in re.finditer(
            r'CREATE TABLE(?: IF NOT EXISTS)?\s+`?(\w+)`?\s*\((.*?)\n\)\s*;',
            sql, re.S | re.I):
        name = m.group(1)
        cols = set()
        for line in m.group(2).split('\n'):
            line = line.strip()
            if line.upper().startswith(('PRIMARY KEY', 'UNIQUE', 'KEY', 'INDEX',
                                        'CONSTRAINT', 'FOREIGN', 'CHECK')):
                continue
            cm = re.match(r'`?(\w+)`?\s+[A-Za-z]', line)
            if cm:
                cols.add(cm.group(1))
        tables[name] = cols
    return tables


def _migrate_added_columns(sql):
    """Map table -> set of columns added by ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``.

    The check must be scoped per table. Searching the whole file lets an
    unrelated table satisfy the requirement: ``ALTER TABLE rfq_items ... ADD
    COLUMN sequence`` once hid the fact that ``process_template_steps.sequence``
    was never added, and ``vendor_work_orders.process_id`` hid the missing
    ``quotation_operations.process_id``.
    """
    added = {}
    for m in re.finditer(
            r'ALTER TABLE\s+`?(\w+)`?\s+ADD COLUMN IF NOT EXISTS\s+`?(\w+)`?',
            sql, re.I):
        added.setdefault(m.group(1), set()).add(m.group(2))
    return added


@pytest.fixture(scope='module')
def schema_state():
    app = create_app('testing')
    models = _model_columns(app)
    schema = _parse_create_tables(os.path.join(REPO_ROOT, 'schema.sql'))
    migrate_sql = open(os.path.join(REPO_ROOT, 'migrate.sql')).read()
    migrate = _migrate_added_columns(migrate_sql)
    return models, schema, migrate


def test_every_model_column_is_covered_by_sql(schema_state):
    models, schema, migrate = schema_state
    missing = []
    for table in sorted(set(models) & set(schema)):
        for col in sorted(models[table] - schema[table]):
            if col not in migrate.get(table, set()):
                missing.append(f'{table}.{col}')
    assert not missing, (
        'Model columns absent from schema.sql and migrate.sql (fresh install '
        'would break): ' + ', '.join(missing))


def test_migrate_covers_tables_absent_from_schema_sql(schema_state):
    """A model table missing from schema.sql must still be created by migrate.sql."""
    models, schema, _ = schema_state
    migrate_sql = open(os.path.join(REPO_ROOT, 'migrate.sql')).read()
    created = {m.group(1) for m in re.finditer(
        r'CREATE TABLE IF NOT EXISTS\s+`?(\w+)`?', migrate_sql, re.I)}
    missing = sorted(t for t in set(models) - set(schema) if t not in created)
    assert not missing, f'Model tables in neither SQL file: {missing}'


def test_every_model_table_exists_in_schema_sql(schema_state):
    models, schema, _ = schema_state
    missing = sorted(set(models) - set(schema))
    assert not missing, f'Model tables absent from schema.sql: {missing}'


def test_sql_files_have_balanced_parentheses():
    for name in ('schema.sql', 'migrate.sql'):
        sql = open(os.path.join(REPO_ROOT, name)).read()
        assert sql.count('(') == sql.count(')'), f'unbalanced parentheses in {name}'
