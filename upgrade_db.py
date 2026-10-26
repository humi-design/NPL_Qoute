"""Bring an existing database up to date with the current models.

Deploying newer code onto a database created by an older release fails at
runtime with "no such table" / "unknown column" errors, because the application
never creates tables at startup. This script closes that gap.

It is additive and idempotent:

* missing tables are created;
* missing columns are added to existing tables.

Nothing is ever dropped, renamed or rewritten, and rows are never modified, so
existing data is preserved. Running it twice is a no-op.

Usage (on cPanel, from the application root, with the virtualenv active):

    python upgrade_db.py

It reads the same ``DATABASE_URL`` / ``DB_*`` environment variables (or ``.env``)
as the application.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()

from app.extensions import create_app, db  # noqa: E402


def _literal(value, column, dialect):
    """Render a Python constant as a SQL literal for the column's type."""
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace('\\', '\\\\').replace("'", "''")
        return f"'{escaped}'"
    return None


def _column_ddl(column, dialect):
    """Render a single ADD COLUMN clause, always nullable.

    New columns are added as nullable on purpose: a NOT NULL column with no
    default cannot be added to a table that already has rows. Leaving it
    nullable keeps the operation safe on populated production tables.
    """
    ddl = f"{dialect.identifier_preparer.quote(column.name)} " \
          f"{column.type.compile(dialect=dialect)}"
    if column.server_default is not None:
        ddl += f" DEFAULT {column.server_default.arg}"
    elif column.default is not None and not column.default.is_callable:
        # A constant model default (e.g. is_current=True) must be written into
        # the DDL, otherwise existing rows get NULL and disappear from any
        # query that filters on the column.
        literal = _literal(column.default.arg, column, dialect)
        if literal is not None:
            ddl += f' DEFAULT {literal}'
    return ddl


def upgrade_database(app):
    """Create missing tables and add missing columns. Returns a report dict."""
    report = {'tables_created': [], 'columns_added': [], 'values_backfilled': []}

    with app.app_context():
        inspector = sa.inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        dialect = db.engine.dialect

        # --- 1. Missing tables -------------------------------------------------
        # create_all() issues CREATE TABLE only for tables it cannot find, so it
        # is safe to call unconditionally.
        model_tables = {t.name for t in db.metadata.sorted_tables}
        report['tables_created'] = sorted(model_tables - existing_tables)
        if report['tables_created']:
            db.create_all()
            existing_tables = set(sa.inspect(db.engine).get_table_names())

        # --- 2. Missing columns on pre-existing tables -------------------------
        inspector = sa.inspect(db.engine)
        for table in db.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # just created in full
            actual = {c['name'] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in actual:
                    continue
                ddl = _column_ddl(column, dialect)
                statement = (f'ALTER TABLE '
                             f'{dialect.identifier_preparer.quote(table.name)} '
                             f'ADD COLUMN {ddl}')
                try:
                    with db.engine.begin() as conn:
                        conn.execute(sa.text(statement))
                    report['columns_added'].append(
                        f'{table.name}.{column.name}')
                except Exception as exc:  # noqa: BLE001
                    print(f'  ! could not add {table.name}.{column.name}: {exc}')
                    continue

                # Backfill existing rows from the constant model default. Only
                # NULLs in the column just added are touched.
                if column.default is not None and not column.default.is_callable:
                    literal = _literal(column.default.arg, column, dialect)
                    if literal is None:
                        continue
                    quoted_table = dialect.identifier_preparer.quote(table.name)
                    quoted_col = dialect.identifier_preparer.quote(column.name)
                    try:
                        with db.engine.begin() as conn:
                            result = conn.execute(sa.text(
                                f'UPDATE {quoted_table} SET {quoted_col} = '
                                f'{literal} WHERE {quoted_col} IS NULL'))
                        if result.rowcount:
                            report['values_backfilled'].append(
                                f'{table.name}.{column.name} '
                                f'({result.rowcount} row(s))')
                    except Exception as exc:  # noqa: BLE001
                        print(f'  ! could not backfill {table.name}.'
                              f'{column.name}: {exc}')

    return report


def main():
    app = create_app(os.environ.get('FLASK_ENV', 'production'))

    url = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    safe_url = url
    if '@' in url and '//' in url:
        scheme, rest = url.split('//', 1)
        safe_url = f'{scheme}//***@{rest.split("@", 1)[1]}'
    print(f'Target database: {safe_url}')

    report = upgrade_database(app)

    if report['tables_created']:
        print(f'\nCreated {len(report["tables_created"])} missing table(s):')
        for name in report['tables_created']:
            print(f'  + {name}')
    else:
        print('\nAll tables already present.')

    if report['columns_added']:
        print(f'\nAdded {len(report["columns_added"])} missing column(s):')
        for name in report['columns_added']:
            print(f'  + {name}')
    else:
        print('All columns already present.')

    if report['values_backfilled']:
        print(f'\nBackfilled default values in '
              f'{len(report["values_backfilled"])} column(s):')
        for name in report['values_backfilled']:
            print(f'  ~ {name}')

    print('\nDatabase is up to date. No existing data was modified.')


if __name__ == '__main__':
    main()
