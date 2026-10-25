"""Supplier/vendor profile financial-visibility tests.

The dashboard already hides payables from users without the admin/reports
permission via ``_may_see_financials``. The supplier and vendor profiles did
not: every logged-in role could read invoice totals, outstanding balances and
ageing buckets. These tests pin the gate so a read_only or sales account sees
operational data but no money.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from datetime import date

from app.models import (User, Supplier, Vendor, SupplierInvoice,
                        VendorInvoice, SupplierPurchaseOrder,
                        SupplierPurchaseOrderLine, Material)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        for username, role in (('admin', 'admin'), ('accounts', 'accounts'),
                               ('viewer', 'read_only')):
            db.session.add(User(
                username=username, email=f'{username}@test.com',
                full_name=username.title(), role=role,
                password_hash=bcrypt.generate_password_hash('pw123456').decode()))
        sup = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel')
        ven = Vendor(vendor_code='VND-0001', vendor_name='Sharp Tools')
        db.session.add_all([sup, ven])
        db.session.add(Material(material_code='M1', material_name='Steel',
                                density=7.85))
        db.session.commit()
        db.session.add(SupplierInvoice(
            invoice_number='INV-SECRET', supplier_id=sup.id, total_amount=987654,
            invoice_date=date.today(), status='posted', currency='INR'))
        db.session.add(VendorInvoice(
            invoice_number='VINV-SECRET', vendor_id=ven.id, total_amount=555555,
            invoice_date=date.today(), status='posted', currency='INR'))
        po = SupplierPurchaseOrder(po_number='PO-0001', supplier_id=sup.id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        db.session.add(SupplierPurchaseOrderLine(
            po_id=po.id, material_id=1, quantity_ordered=100, unit='kg', rate=50))
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def _login(app, username):
    c = app.test_client()
    c.post('/auth/login', data={'username': username, 'password': 'pw123456'},
           follow_redirects=True)
    return c


def test_supplier_profile_hides_financials_from_read_only(app):
    body = _login(app, 'viewer').get('/suppliers/1').get_data(as_text=True)
    assert '987,654' not in body          # invoiced amount
    assert 'INV-SECRET' not in body       # invoice number
    assert 'Ageing' not in body
    # Operational data stays visible.
    assert 'Purchase Orders (1)' in body
    assert 'PO-0001' in body


def test_supplier_profile_shows_financials_to_accounts(app):
    body = _login(app, 'accounts').get('/suppliers/1').get_data(as_text=True)
    assert '987,654' in body
    assert 'INV-SECRET' in body
    assert 'Ageing' in body


def test_supplier_profile_shows_financials_to_admin(app):
    body = _login(app, 'admin').get('/suppliers/1').get_data(as_text=True)
    assert '987,654' in body
    assert 'Ageing' in body


def test_vendor_profile_hides_financials_from_read_only(app):
    body = _login(app, 'viewer').get('/vendors/1').get_data(as_text=True)
    assert '555,555' not in body
    assert 'VINV-SECRET' not in body
    assert 'Ageing' not in body
    assert 'Job-Work Charges' not in body
    # Operational data stays visible.
    assert 'Quantity Reconciliation' in body


def test_vendor_profile_shows_financials_to_accounts(app):
    body = _login(app, 'accounts').get('/vendors/1').get_data(as_text=True)
    assert '555,555' in body
    assert 'VINV-SECRET' in body
    assert 'Ageing' in body
    assert 'Job-Work Charges' in body


def test_financial_gate_matches_dashboard_permission(app):
    """The profile gate must agree with the dashboard gate for every role."""
    from app.utils import _may_see_financials
    with app.app_context():
        for username, expected in (('admin', True), ('accounts', True),
                                   ('viewer', False)):
            user = User.query.filter_by(username=username).first()
            assert _may_see_financials(user) is expected, username
