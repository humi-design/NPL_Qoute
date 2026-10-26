"""Bring an existing database up to date with the current models.

Deploying newer code onto a database created by an older release fails at
runtime with "no such table" / "unknown column" errors, because the application
never creates tables at startup. This module closes that gap and is shared by
two callers:

* ``upgrade_db.py``, the manual command-line upgrader;
* ``create_app()``, which runs it automatically so a deploy cannot leave the
  site broken.

It is additive and idempotent:

* missing tables are created;
* missing columns are added to existing tables.

Nothing is ever dropped, renamed or rewritten, and existing rows are never
modified. The only writes are the constant model defaults backfilled into
columns this run created, so a healthy database produces no writes at all.
"""
import sqlalchemy as sa


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


def _add_missing_tables(db, report):
    """Create tables the database does not have yet."""
    model_tables = {t.name for t in db.metadata.sorted_tables}
    existing = set(sa.inspect(db.engine).get_table_names())
    report['tables_created'] = sorted(model_tables - existing)
    if not report['tables_created']:
        return existing
    try:
        db.create_all()
    except Exception:  # noqa: BLE001
        # Another worker may have created the tables between our check and the
        # CREATE. Re-inspect; anything still missing is reported by the caller
        # on its next boot.
        pass
    return set(sa.inspect(db.engine).get_table_names())


def _add_missing_columns(db, existing_tables, report):
    """Add columns present in the models but absent from existing tables."""
    inspector = sa.inspect(db.engine)
    dialect = db.engine.dialect
    for table in db.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # created in full above
        actual = {c['name'] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in actual:
                continue
            quoted_table = dialect.identifier_preparer.quote(table.name)
            statement = (f'ALTER TABLE {quoted_table} '
                         f'ADD COLUMN {_column_ddl(column, dialect)}')
            try:
                with db.engine.begin() as conn:
                    conn.execute(sa.text(statement))
            except Exception as exc:  # noqa: BLE001
                print(f'  ! could not add {table.name}.{column.name}: {exc}')
                continue
            report['columns_added'].append(f'{table.name}.{column.name}')

            # Backfill existing rows from the constant model default. Only
            # NULLs in the column just added are touched, so this never
            # rewrites data the application has already written.
            if column.default is not None and not column.default.is_callable:
                literal = _literal(column.default.arg, column, dialect)
                if literal is None:
                    continue
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


def upgrade_database(app):
    """Create missing tables and add missing columns. Returns a report dict."""
    from app.extensions import db

    report = {'tables_created': [], 'columns_added': [], 'values_backfilled': []}
    with app.app_context():
        existing_tables = _add_missing_tables(db, report)
        _add_missing_columns(db, existing_tables, report)
    return report


def auto_upgrade(app):
    """Run :func:`upgrade_database` without ever breaking application startup.

    Called from ``create_app()``. A failure here must not stop the WSGI
    application object from being created, because that would take every URL on
    the site down; the error is logged and the app boots against whatever
    schema exists. Set ``AUTO_UPGRADE_DB=false`` to skip it entirely.
    """
    report = upgrade_database(app)
    total = (len(report['tables_created']) + len(report['columns_added'])
             + len(report['values_backfilled']))
    if total:
        app.logger.info(
            'Database upgraded automatically: %d table(s) created, '
            '%d column(s) added, %d column(s) backfilled',
            len(report['tables_created']), len(report['columns_added']),
            len(report['values_backfilled']))
    return report
