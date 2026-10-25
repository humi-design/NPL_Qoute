"""Tests for upgrade_db.py — the additive schema upgrader.

Deploying new code onto an older database is the failure this guards: the app
never creates tables at startup, so a stale database produced
"no such table: orders" (HTTP 500) on a previously working install.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db  # noqa: E402
from app.models import Customer, User  # noqa: E402
from upgrade_db import upgrade_database  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    # Config classes evaluate SQLALCHEMY_DATABASE_URI at import time, so the
    # override has to be applied to the class before create_app() runs.
    from app.config import ProductionConfig
    ProductionConfig.SQLALCHEMY_DATABASE_URI = f'sqlite:///{tmp_path / "upgrade.db"}'
    app = create_app('production')
    yield app


def test_upgrade_creates_missing_tables_without_touching_data(app):
    """An old DB gains the new tables; existing rows are preserved."""
    with app.app_context():
        # Simulate an older release: only the master-data tables exist.
        db.metadata.tables['customers'].create(db.engine)
        db.metadata.tables['users'].create(db.engine)
        db.session.add(Customer(customer_code='CUST-1',
                                company_name='Existing Customer'))
        db.session.commit()

    report = upgrade_database(app)

    assert 'orders' in report['tables_created']
    assert 'suppliers' in report['tables_created']

    with app.app_context():
        from sqlalchemy import inspect
        tables = set(inspect(db.engine).get_table_names())
        assert 'orders' in tables and 'supplier_invoices' in tables
        # The pre-existing row must still be there, untouched.
        row = Customer.query.filter_by(customer_code='CUST-1').first()
        assert row is not None
        assert row.company_name == 'Existing Customer'


def test_upgrade_is_idempotent(app):
    """Running the upgrader twice must be a no-op the second time."""
    upgrade_database(app)
    second = upgrade_database(app)
    assert second['tables_created'] == []
    assert second['columns_added'] == []


def test_upgrade_adds_missing_columns_to_existing_table(app):
    """Columns added to a model after a table exists must be backfilled."""
    with app.app_context():
        # A 'customers' table missing later columns.
        db.session.execute(db.text(
            'CREATE TABLE customers (id INTEGER PRIMARY KEY, '
            'customer_code VARCHAR(20), company_name VARCHAR(200))'))
        db.session.execute(db.text(
            "INSERT INTO customers (customer_code, company_name) "
            "VALUES ('CUST-OLD', 'Legacy Co')"))
        db.session.commit()

    report = upgrade_database(app)

    assert any(c.startswith('customers.') for c in report['columns_added'])
    with app.app_context():
        row = db.session.execute(db.text(
            "SELECT company_name FROM customers WHERE customer_code='CUST-OLD'"
        )).fetchone()
        assert row[0] == 'Legacy Co'
