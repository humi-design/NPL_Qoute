"""URL render sweep.

Walks every GET route registered in the app against a seeded database and
asserts none of them return a server error. This catches template and route
regressions that unit tests miss. Read-only by design: only GET routes are
exercised.
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (User, Supplier, SupplierType, Material, SupplierMaterial,
                        Vendor, VendorWorkOrder, Customer, SupplierPurchaseOrder,
                        SupplierPurchaseOrderLine, SupplierOrderReceipt,
                        SupplierInvoice, SupplierPayment, VendorInvoice,
                        VendorPayment)


@pytest.fixture()
def app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        _seed()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _seed():
    user = User(username='admin', email='admin@test.com', full_name='Admin',
                role='admin',
                password_hash=bcrypt.generate_password_hash('admin123').decode())
    db.session.add(user)
    db.session.flush()

    st = SupplierType(type_code='SUPT-001', type_name='Raw Material')
    db.session.add(st)
    db.session.flush()

    sup = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel',
                   supplier_type_id=st.id, lead_time=7, payment_terms='Net 30')
    db.session.add(sup)
    db.session.flush()

    mat = Material(material_code='MAT-0001', material_name='SS304', density=8000,
                   supplier_id=sup.id)
    db.session.add(mat)
    db.session.flush()
    db.session.add(SupplierMaterial(supplier_id=sup.id, material_id=mat.id,
                                    grade='AISI 304', rate=210, lead_time=5,
                                    is_preferred=True))

    cust = Customer(customer_code='CUST-0001', company_name='Acme')
    db.session.add(cust)

    vend = Vendor(vendor_code='VND-0001', vendor_name='Alpha Heat Treat')
    db.session.add(vend)
    db.session.flush()
    db.session.add(VendorWorkOrder(work_order_number='WO-2026-0001', vendor_id=vend.id,
                                   title='HT batch', quantity_ordered=100,
                                   quantity_received=60, status='in_progress'))

    po = SupplierPurchaseOrder(po_number='PO-2026-0001', supplier_id=sup.id,
                               order_date=date.today(),
                               expected_date=date.today() + timedelta(days=5),
                               status='sent')
    db.session.add(po)
    db.session.flush()
    line = SupplierPurchaseOrderLine(po_id=po.id, material_id=mat.id,
                                     quantity_ordered=1000, unit='kg', rate=210)
    db.session.add(line)
    db.session.flush()
    rec = SupplierOrderReceipt(receipt_number='GRN-2026-0001', po_id=po.id,
                               supplier_id=sup.id, receipt_date=date.today())
    db.session.add(rec)
    db.session.flush()

    inv = SupplierInvoice(invoice_number='INV-001', supplier_id=sup.id,
                          invoice_date=date.today(),
                          due_date=date.today() - timedelta(days=40),
                          total_amount=50000, status='posted')
    db.session.add(inv)
    db.session.flush()
    db.session.add(SupplierPayment(payment_number='PAY-2026-0001', supplier_id=sup.id,
                                   payment_date=date.today(), amount=20000))

    vinv = VendorInvoice(invoice_number='VINV-001', vendor_id=vend.id,
                         invoice_date=date.today(), total_amount=9000, status='posted')
    db.session.add(vinv)
    db.session.flush()
    db.session.add(VendorPayment(payment_number='VPAY-2026-0001', vendor_id=vend.id,
                                 payment_date=date.today(), amount=4000))
    db.session.commit()


def test_all_get_routes_render(client, app):
    client.post('/auth/login', data={'username': 'admin', 'password': 'admin123'},
                follow_redirects=True)

    with app.app_context():
        sup_id = Supplier.query.first().id
        mat_id = Material.query.first().id
        vend_id = Vendor.query.first().id
        po_id = SupplierPurchaseOrder.query.first().id
        rec_id = SupplierOrderReceipt.query.first().id
        wo_id = VendorWorkOrder.query.first().id
        inv_id = SupplierInvoice.query.first().id
        vinv_id = VendorInvoice.query.first().id

    # Substitute concrete ids into routes with <int:id> / <int:...> params
    params = {
        'id': [sup_id, mat_id, vend_id, po_id, rec_id, wo_id, inv_id, vinv_id, 1],
        'supplier_id': [sup_id], 'material_id': [mat_id], 'vendor_id': [vend_id],
        'line_id': [1], 'po_id': [po_id],
    }

    checked, failures = 0, []
    seen = set()
    for rule in app.url_map.iter_rules():
        if 'GET' not in rule.methods or rule.endpoint == 'static':
            continue
        if rule.endpoint.startswith(('auth.logout',)):
            continue
        url = str(rule)
        for arg in rule.arguments:
            if arg in params:
                url = url.replace(f'<int:{arg}>', str(params[arg][0]))
                url = url.replace(f'<{arg}>', str(params[arg][0]))
            else:
                url = None
                break
        if url is None or '<' in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = client.get(url)
        except Exception as exc:  # noqa: BLE001
            failures.append((url, f'EXC {type(exc).__name__}: {exc}'))
            continue
        checked += 1
        if resp.status_code >= 500:
            failures.append((url, resp.status_code))

    assert not failures, f'{len(failures)} route(s) failed: {failures}'
    assert checked >= 25, f'only checked {checked} routes'
    print(f'\nURL sweep: {checked} GET routes rendered without server errors')
