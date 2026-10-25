"""Procurement & payables tests.

Exercises the real Flask app, real models, real routes and real templates
against the app's existing 'testing' config (SQLite, CSRF disabled). No mocks.

Focus areas:
  * quantity reconciliation (ordered / received / accepted / rejected / pending)
  * batch, heat, lot and certificate capture on goods receipts
  * payables integrity: allocations are counted once and capped at the payment
    balance; a payment cannot be applied to the same invoice twice
  * ageing buckets and the "not tracked vs zero" behaviour
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Supplier, Material, SupplierPurchaseOrder, SupplierPurchaseOrderLine,
    SupplierOrderReceipt, SupplierOrderReceiptLine, SupplierInvoice,
    SupplierPayment, SupplierPaymentAllocation, Vendor, VendorInvoice,
    VendorPayment, VendorPaymentAllocation,
)
from app.utils import (
    compute_supplier_procurement_summary, compute_vendor_financial_summary,
    generate_po_number, generate_receipt_number, generate_payment_number,
)


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


def login(client):
    return client.post('/auth/login',
                       data={'username': 'admin', 'password': 'admin123'},
                       follow_redirects=True)


def _seed():
    user = User(username='admin', email='admin@test.com', full_name='Admin',
                role='admin',
                password_hash=bcrypt.generate_password_hash('admin123').decode())
    db.session.add(user)
    db.session.flush()

    mat = Material(material_code='MAT-0001', material_name='SS304', density=8000)
    mat2 = Material(material_code='MAT-0002', material_name='EN8 Bar', density=7850)
    db.session.add_all([mat, mat2])

    sup = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel',
                   payment_terms='Net 30', lead_time=7)
    sup2 = Supplier(supplier_code='SUP-0002', company_name='Empty Supplier')
    db.session.add_all([sup, sup2])

    vend = Vendor(vendor_code='VND-0001', vendor_name='Alpha Heat Treat')
    db.session.add(vend)
    db.session.commit()


# ============================================================
# Purchase order quantities
# ============================================================

def test_po_quantity_reconciliation(app):
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        mat = Material.query.filter_by(material_code='MAT-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        line = SupplierPurchaseOrderLine(po_id=po.id, material_id=mat.id,
                                         quantity_ordered=1000, unit='kg', rate=210)
        db.session.add(line)
        db.session.flush()

        assert po.ordered_quantity == 1000
        assert po.pending_quantity == 1000
        assert po.received_quantity == 0
        assert po.completion_percent == 0

        receipt = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                       po_id=po.id, supplier_id=sup.id,
                                       receipt_date=date.today())
        db.session.add(receipt)
        db.session.flush()
        db.session.add(SupplierOrderReceiptLine(
            receipt_id=receipt.id, po_line_id=line.id, material_id=mat.id,
            quantity_received=600, quantity_rejected=50, batch_number='B-1',
            heat_number='H-1', inspection_status='passed'))
        db.session.commit()

        # received derives from receipt lines; accepted = received - rejected
        assert po.received_quantity == 600
        assert po.rejected_quantity == 50
        assert po.accepted_quantity == 550
        assert po.pending_quantity == 400
        assert po.completion_percent == 60.0


def test_overdue_po_detection(app):
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today() - timedelta(days=20),
                                   expected_date=date.today() - timedelta(days=5),
                                   status='sent')
        db.session.add(po)
        db.session.flush()
        db.session.add(SupplierPurchaseOrderLine(po_id=po.id, quantity_ordered=100,
                                                 unit='kg', rate=10))
        db.session.commit()

        assert po.is_overdue is True
        assert po.days_overdue == 5

        # A received PO is no longer overdue
        po.status = 'received'
        db.session.commit()
        assert po.is_overdue is False


def test_receipt_line_rejected_exceeds_received_is_blocked(app, client):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        mat = Material.query.filter_by(material_code='MAT-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        line = SupplierPurchaseOrderLine(po_id=po.id, material_id=mat.id,
                                         quantity_ordered=100, unit='kg', rate=10)
        db.session.add(line)
        receipt = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                       po_id=po.id, supplier_id=sup.id,
                                       receipt_date=date.today())
        db.session.add(receipt)
        db.session.commit()
        rec_id, line_id = receipt.id, line.id

    resp = client.post(f'/procurement/receipts/{rec_id}/line', data={
        'po_line_id': line_id, 'quantity_received': 10, 'quantity_rejected': 99,
        'inspection_status': 'failed'}, follow_redirects=True)
    assert 'cannot exceed received' in resp.get_data(as_text=True).lower()

    with app.app_context():
        assert SupplierOrderReceiptLine.query.count() == 0


# ============================================================
# Batch / heat / lot / certificate capture
# ============================================================

def test_receipt_line_captures_batch_heat_lot_and_certificate(app, client):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        mat = Material.query.filter_by(material_code='MAT-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        line = SupplierPurchaseOrderLine(po_id=po.id, material_id=mat.id,
                                         quantity_ordered=500, unit='kg', rate=100)
        db.session.add(line)
        receipt = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                       po_id=po.id, supplier_id=sup.id,
                                       receipt_date=date.today())
        db.session.add(receipt)
        db.session.commit()
        rec_id, line_id = receipt.id, line.id

    client.post(f'/procurement/receipts/{rec_id}/line', data={
        'po_line_id': line_id, 'quantity_received': 500, 'quantity_rejected': 0,
        'batch_number': 'B-2024-01', 'heat_number': 'H-777', 'lot_number': 'L-9',
        'certificate_ref': 'MC-2024-99', 'inspection_status': 'passed'},
        follow_redirects=True)

    with app.app_context():
        rl = SupplierOrderReceiptLine.query.first()
        assert rl.batch_number == 'B-2024-01'
        assert rl.heat_number == 'H-777'
        assert rl.lot_number == 'L-9'
        assert rl.certificate_ref == 'MC-2024-99'
        assert rl.inspection_status == 'passed'
        # Fully received PO auto-advances status
        po = db.session.get(SupplierPurchaseOrder, rl.receipt.purchase_order.id)
        assert po.status == 'received'


# ============================================================
# Payables integrity
# ============================================================

def test_payment_allocation_is_counted_once(app, client):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        inv = SupplierInvoice(invoice_number='INV-001', supplier_id=sup.id,
                              invoice_date=date.today(),
                              due_date=date.today() - timedelta(days=45),
                              total_amount=120000, status='posted')
        db.session.add(inv)
        pay = SupplierPayment(payment_number=generate_payment_number(),
                              supplier_id=sup.id, payment_date=date.today(),
                              amount=50000, method='bank')
        db.session.add(pay)
        db.session.commit()
        inv_id, pay_id = inv.id, pay.id
        assert inv.outstanding_amount == 120000
        assert pay.unallocated_amount == 50000

    client.post(f'/procurement/supplier-payments/{pay_id}/allocate',
                data={'invoice_id': inv_id, 'amount': 50000}, follow_redirects=True)

    with app.app_context():
        inv = db.session.get(SupplierInvoice, inv_id)
        pay = db.session.get(SupplierPayment, pay_id)
        assert inv.paid_amount == 50000
        assert inv.outstanding_amount == 70000
        assert inv.ageing_bucket == '31-60'
        assert pay.unallocated_amount == 0

    # Same payment cannot be applied to the same invoice twice
    resp = client.post(f'/procurement/supplier-payments/{pay_id}/allocate',
                       data={'invoice_id': inv_id, 'amount': 1},
                       follow_redirects=True)
    assert 'already allocated' in resp.get_data(as_text=True).lower()

    with app.app_context():
        inv = db.session.get(SupplierInvoice, inv_id)
        assert inv.paid_amount == 50000  # unchanged, not double counted


def test_allocation_cannot_exceed_payment_balance(app, client):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        inv = SupplierInvoice(invoice_number='INV-002', supplier_id=sup.id,
                              invoice_date=date.today(), total_amount=100000,
                              status='posted')
        db.session.add(inv)
        pay = SupplierPayment(payment_number=generate_payment_number(),
                              supplier_id=sup.id, payment_date=date.today(),
                              amount=1000)
        db.session.add(pay)
        db.session.commit()
        inv_id, pay_id = inv.id, pay.id

    resp = client.post(f'/procurement/supplier-payments/{pay_id}/allocate',
                       data={'invoice_id': inv_id, 'amount': 999999},
                       follow_redirects=True)
    assert 'exceeds the payment balance' in resp.get_data(as_text=True).lower()

    with app.app_context():
        assert SupplierPaymentAllocation.query.count() == 0


def test_ageing_buckets(app):
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        specs = [
            ('not_due', 0), ('0-30', 10), ('31-60', 45), ('61-90', 75), ('90+', 200),
        ]
        for name, days in specs:
            db.session.add(SupplierInvoice(
                invoice_number=f'INV-{name}', supplier_id=sup.id,
                invoice_date=date.today() - timedelta(days=days + 30),
                due_date=date.today() - timedelta(days=days),
                total_amount=1000, status='posted'))
        db.session.commit()

        summ = compute_supplier_procurement_summary(sup)
        for name, _ in specs:
            assert summ['ageing'][name] == 1000, name
        assert summ['ageing']['total'] == 5000
        assert summ['outstanding_amount'] == 5000


def test_not_tracked_vs_zero(app):
    """A supplier with no procurement activity reports zero counts but the
    has_procurement_data flag stays False so the UI can say 'not tracked'."""
    with app.app_context():
        sup2 = Supplier.query.filter_by(supplier_code='SUP-0002').first()
        summ = compute_supplier_procurement_summary(sup2)
        assert summ['po_count'] == 0
        assert summ['ordered_quantity'] == 0
        assert summ['outstanding_amount'] == 0
        assert summ['on_time_percent'] is None      # not measurable, not 0%
        assert summ['has_procurement_data'] is False

        # A supplier with real data flips the flag
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        db.session.add(SupplierPurchaseOrder(po_number=generate_po_number(),
                                             supplier_id=sup.id,
                                             order_date=date.today(), status='draft'))
        db.session.commit()
        assert compute_supplier_procurement_summary(sup)['has_procurement_data'] is True


def test_on_time_delivery_metric(app):
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today() - timedelta(days=10),
                                   expected_date=date.today() - timedelta(days=2),
                                   status='sent')
        db.session.add(po)
        db.session.flush()
        db.session.add(SupplierPurchaseOrderLine(po_id=po.id, quantity_ordered=100,
                                                 unit='kg', rate=1))
        # one on-time receipt, one late receipt
        on_time = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                       po_id=po.id, supplier_id=sup.id,
                                       receipt_date=date.today() - timedelta(days=3))
        db.session.add(on_time)
        db.session.flush()  # real routes commit each receipt, so the next number advances
        late = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                    po_id=po.id, supplier_id=sup.id,
                                    receipt_date=date.today() - timedelta(days=1))
        db.session.add(late)
        db.session.commit()

        summ = compute_supplier_procurement_summary(sup)
        assert summ['deliveries_measured'] == 2
        assert summ['deliveries_on_time'] == 1
        assert summ['on_time_percent'] == 50.0


# ============================================================
# Vendor (job-work) payables
# ============================================================

def test_vendor_financial_summary(app, client):
    login(client)
    with app.app_context():
        vend = Vendor.query.filter_by(vendor_code='VND-0001').first()
        inv = VendorInvoice(invoice_number='VINV-1', vendor_id=vend.id,
                            invoice_date=date.today(),
                            due_date=date.today() - timedelta(days=10),
                            total_amount=8000, status='posted')
        db.session.add(inv)
        pay = VendorPayment(payment_number=generate_payment_number(
                                model=VendorPayment, prefix='VPAY'),
                            vendor_id=vend.id, payment_date=date.today(), amount=3000)
        db.session.add(pay)
        db.session.commit()
        inv_id, pay_id, vend_id = inv.id, pay.id, vend.id

    client.post(f'/procurement/vendor-payments/{pay_id}/allocate',
                data={'invoice_id': inv_id, 'amount': 3000}, follow_redirects=True)

    with app.app_context():
        vend = db.session.get(Vendor, vend_id)
        summ = compute_vendor_financial_summary(vend)
        assert summ['invoice_count'] == 1
        assert summ['invoiced_amount'] == 8000
        assert summ['paid_on_invoices'] == 3000
        assert summ['outstanding_amount'] == 5000
        assert summ['ageing']['0-30'] == 5000
        assert summ['has_financial_data'] is True


# ============================================================
# Route / permission / render tests
# ============================================================

def test_procurement_requires_login(client):
    resp = client.get('/procurement/purchase-orders')
    assert resp.status_code == 302
    assert '/auth/login' in resp.headers['Location']


def test_procurement_pages_render(client):
    login(client)
    for url in ['/procurement/', '/procurement/purchase-orders',
                '/procurement/supplier-invoices', '/procurement/supplier-payments']:
        resp = client.get(url)
        assert resp.status_code == 200, url


def test_create_po_via_route(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    resp = client.post('/procurement/purchase-orders/new', data={
        'supplier_id': sup_id, 'order_date': date.today().isoformat(),
        'expected_date': (date.today() + timedelta(days=7)).isoformat(),
        'status': 'sent', 'currency': 'INR', 'payment_terms': 'Net 30',
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert '/procurement/purchase-orders/' in resp.headers['Location']

    with app.app_context():
        po = SupplierPurchaseOrder.query.first()
        assert po is not None
        assert po.supplier_id == sup_id
        assert po.status == 'sent'


def test_po_without_supplier_is_rejected(client, app):
    login(client)
    resp = client.post('/procurement/purchase-orders/new', data={
        'order_date': date.today().isoformat(), 'status': 'draft',
        'currency': 'INR',
    }, follow_redirects=True)
    assert 'select a supplier' in resp.get_data(as_text=True).lower()
    with app.app_context():
        assert SupplierPurchaseOrder.query.count() == 0


def test_read_only_cannot_write(client, app):
    with app.app_context():
        ro = User(username='viewer', email='v@test.com', full_name='V',
                  role='read_only',
                  password_hash=bcrypt.generate_password_hash('view123').decode())
        db.session.add(ro)
        db.session.commit()

    client.post('/auth/login', data={'username': 'viewer', 'password': 'view123'},
                follow_redirects=True)
    resp = client.post('/procurement/purchase-orders/new', data={
        'order_date': date.today().isoformat(), 'status': 'draft',
        'currency': 'INR'}, follow_redirects=True)
    assert 'permission' in resp.get_data(as_text=True).lower()
    with app.app_context():
        assert SupplierPurchaseOrder.query.count() == 0


def test_delete_po_with_receipt_is_blocked(client, app):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup.id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        db.session.add(SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                            po_id=po.id, supplier_id=sup.id,
                                            receipt_date=date.today()))
        db.session.commit()
        po_id = po.id

    resp = client.post(f'/procurement/purchase-orders/{po_id}/delete',
                       follow_redirects=True)
    assert 'cannot delete' in resp.get_data(as_text=True).lower()
    with app.app_context():
        assert SupplierPurchaseOrder.query.count() == 1


def test_supplier_profile_shows_real_procurement_data(client, app):
    """The supplier profile must surface real PO / batch / invoice figures."""
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        mat = Material.query.filter_by(material_code='MAT-0001').first()
        sup_id = sup.id
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup_id,
                                   order_date=date.today(),
                                   expected_date=date.today() - timedelta(days=3),
                                   status='sent')
        db.session.add(po)
        db.session.flush()
        line = SupplierPurchaseOrderLine(po_id=po.id, material_id=mat.id,
                                         quantity_ordered=1000, unit='kg', rate=210)
        db.session.add(line)
        db.session.flush()
        rec = SupplierOrderReceipt(receipt_number=generate_receipt_number(),
                                   po_id=po.id, supplier_id=sup_id,
                                   receipt_date=date.today())
        db.session.add(rec)
        db.session.flush()
        db.session.add(SupplierOrderReceiptLine(
            receipt_id=rec.id, po_line_id=line.id, material_id=mat.id,
            quantity_received=600, quantity_rejected=50, batch_number='BATCH-XYZ',
            heat_number='HEAT-42', inspection_status='passed'))
        db.session.add(SupplierInvoice(invoice_number='INV-PROF', supplier_id=sup_id,
                                       invoice_date=date.today(),
                                       due_date=date.today() - timedelta(days=10),
                                       total_amount=123456, status='posted'))
        db.session.commit()

    body = client.get(f'/suppliers/{sup_id}').get_data(as_text=True)
    assert 'Purchase Orders (1)' in body
    assert 'BATCH-XYZ' in body and 'HEAT-42' in body
    assert 'INV-PROF' in body
    assert '123,456' in body          # invoiced amount rendered
    assert 'Outstanding' in body
    assert 'Ageing' in body
    assert 'Material Batches' in body


def test_vendor_profile_shows_real_financial_data(client, app):
    login(client)
    with app.app_context():
        vend = Vendor.query.filter_by(vendor_code='VND-0001').first()
        vend_id = vend.id
        inv = VendorInvoice(invoice_number='VINV-PROF', vendor_id=vend_id,
                            invoice_date=date.today(),
                            due_date=date.today() - timedelta(days=5),
                            total_amount=77777, status='posted')
        db.session.add(inv)
        db.session.flush()
        pay = VendorPayment(payment_number='VPAY-2026-0009', vendor_id=vend_id,
                            payment_date=date.today(), amount=10000)
        db.session.add(pay)
        db.session.commit()
        inv_id, pay_id = inv.id, pay.id

    client.post(f'/procurement/vendor-payments/{pay_id}/allocate',
                data={'invoice_id': inv_id, 'amount': 10000}, follow_redirects=True)

    body = client.get(f'/vendors/{vend_id}').get_data(as_text=True)
    assert 'VINV-PROF' in body
    assert '77,777' in body
    assert 'Outstanding' in body
    assert 'Ageing' in body


def test_api_supplier_exposes_procurement(client, app):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        sup_id = sup.id
        db.session.add(SupplierPurchaseOrder(po_number=generate_po_number(),
                                             supplier_id=sup_id,
                                             order_date=date.today(), status='sent'))
        db.session.commit()

    data = client.get(f'/api/suppliers/{sup_id}').get_json()
    assert 'procurement' in data
    assert data['procurement']['po_count'] == 1
    assert data['procurement']['has_procurement_data'] is True
    assert len(data['purchase_orders']) == 1
    assert 'invoices' in data and 'batches' in data


# ============================================================
# List conventions: pagination, sort whitelist, export
# ============================================================

def _make_pos(sup_id, n, status='sent'):
    pos = []
    for i in range(n):
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup_id,
                                   order_date=date.today() - timedelta(days=i),
                                   status=status)
        db.session.add(po)
        db.session.flush()
        pos.append(po)
    db.session.commit()
    return pos


def test_po_list_paginates(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        _make_pos(sup_id, 30)

    body = client.get('/procurement/purchase-orders?per_page=10').get_data(as_text=True)
    assert 'Showing 10 of 30 purchase orders' in body

    body2 = client.get('/procurement/purchase-orders?per_page=10&page=3').get_data(as_text=True)
    assert 'Showing 10 of 30 purchase orders' in body2

    # invalid per_page falls back to the 25 default
    body3 = client.get('/procurement/purchase-orders?per_page=999').get_data(as_text=True)
    assert 'Showing 25 of 30 purchase orders' in body3


def test_po_list_sort_whitelist_is_safe(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        _make_pos(sup_id, 3)

    # A hostile sort value must not error; it falls back to the default column.
    for sort in ['number', 'supplier', 'expected', 'status', 'id; DROP TABLE users']:
        resp = client.get(f'/procurement/purchase-orders?sort={sort}&dir=asc')
        assert resp.status_code == 200, sort

    # A real sort still works and keeps the page rendering
    resp = client.get('/procurement/purchase-orders?sort=number&dir=asc')
    assert 'Purchase Orders' in resp.get_data(as_text=True)


def test_po_list_search_and_status_filter(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        _make_pos(sup_id, 2, status='sent')
        _make_pos(sup_id, 1, status='draft')

    def row_count(body):
        # Each rendered row carries an onclick that navigates to the PO detail.
        return body.count("onclick=\"window.location='/procurement/purchase-orders/")

    sent = client.get('/procurement/purchase-orders?status=sent').get_data(as_text=True)
    assert row_count(sent) == 2
    draft = client.get('/procurement/purchase-orders?status=draft').get_data(as_text=True)
    assert row_count(draft) == 1
    # The seeded supplier is "Bharat ..." so a name search matches all three.
    searched = client.get('/procurement/purchase-orders?search=Bharat').get_data(as_text=True)
    assert row_count(searched) == 3
    # A non-matching search yields the empty state.
    none = client.get('/procurement/purchase-orders?search=ZZZ-nope').get_data(as_text=True)
    assert row_count(none) == 0
    assert 'No purchase orders found' in none


def test_invoice_and_payment_lists_paginate(client, app):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        sup_id = sup.id
        for i in range(12):
            db.session.add(SupplierInvoice(invoice_number=f'INV-{i:03d}',
                                           supplier_id=sup_id,
                                           invoice_date=date.today(),
                                           total_amount=1000, status='posted'))
            db.session.add(SupplierPayment(payment_number=f'PAY-2026-{i:04d}',
                                           supplier_id=sup_id,
                                           payment_date=date.today(), amount=100))
        db.session.commit()

    inv = client.get('/procurement/supplier-invoices?per_page=10').get_data(as_text=True)
    assert 'Showing 10 of 12 invoices' in inv
    pay = client.get('/procurement/supplier-payments?per_page=10').get_data(as_text=True)
    assert 'Showing 10 of 12 payments' in pay


def test_exports_return_excel(client, app):
    login(client)
    with app.app_context():
        sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        sup_id = sup.id
        po = SupplierPurchaseOrder(po_number=generate_po_number(), supplier_id=sup_id,
                                   order_date=date.today(), status='sent')
        db.session.add(po)
        db.session.flush()
        db.session.add(SupplierPurchaseOrderLine(po_id=po.id, quantity_ordered=100,
                                                 unit='kg', rate=10))
        db.session.add(SupplierInvoice(invoice_number='INV-X', supplier_id=sup_id,
                                       invoice_date=date.today(), total_amount=500,
                                       status='posted'))
        db.session.add(SupplierPayment(payment_number='PAY-2026-0001', supplier_id=sup_id,
                                       payment_date=date.today(), amount=200))
        db.session.commit()

    for url in ['/procurement/purchase-orders/export',
                '/procurement/supplier-invoices/export',
                '/procurement/supplier-payments/export']:
        resp = client.get(url)
        assert resp.status_code == 200, url
        assert 'spreadsheetml' in resp.headers['Content-Type'], url
        # XLSX files are ZIP archives -> start with the PK magic bytes
        assert resp.data[:2] == b'PK', url


def test_export_requires_login(client):
    resp = client.get('/procurement/purchase-orders/export')
    assert resp.status_code == 302
    assert '/auth/login' in resp.headers['Location']


# ============================================================
# Edit routes
# ============================================================

def _make_invoice(sup_id, number='INV-EDIT', total=10000):
    inv = SupplierInvoice(invoice_number=number, supplier_id=sup_id,
                          invoice_date=date.today(), total_amount=total,
                          status='posted')
    db.session.add(inv)
    db.session.commit()
    return inv.id


def test_supplier_invoice_edit(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        inv_id = _make_invoice(sup_id)

    form = client.get(f'/procurement/supplier-invoices/{inv_id}/edit')
    assert form.status_code == 200
    assert 'Edit Invoice' in form.get_data(as_text=True)

    resp = client.post(f'/procurement/supplier-invoices/{inv_id}/edit', data={
        'invoice_number': 'INV-EDIT-2', 'invoice_date': str(date.today()),
        'due_date': '', 'po_id': '', 'subtotal': 9000, 'tax_amount': 1620,
        'total_amount': 10620, 'currency': 'INR', 'status': 'posted', 'notes': 'edited',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with app.app_context():
        inv = SupplierInvoice.query.get(inv_id)
        assert inv.invoice_number == 'INV-EDIT-2'
        assert inv.total_amount == 10620


def test_supplier_invoice_edit_blocks_total_below_paid(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        inv_id = _make_invoice(sup_id, number='INV-PAID', total=10000)
        pay = SupplierPayment(payment_number=generate_payment_number(),
                              supplier_id=sup_id, payment_date=date.today(),
                              amount=8000)
        db.session.add(pay)
        db.session.flush()
        db.session.add(SupplierPaymentAllocation(payment_id=pay.id,
                                                 invoice_id=inv_id, amount=8000))
        db.session.commit()

    resp = client.post(f'/procurement/supplier-invoices/{inv_id}/edit', data={
        'invoice_number': 'INV-PAID', 'invoice_date': str(date.today()),
        'due_date': '', 'po_id': '', 'subtotal': 0, 'tax_amount': 0,
        'total_amount': 5000, 'currency': 'INR', 'status': 'posted', 'notes': '',
    }, follow_redirects=True)
    assert 'cannot be less than the amount already paid' in resp.get_data(as_text=True)

    with app.app_context():
        assert SupplierInvoice.query.get(inv_id).total_amount == 10000


def test_supplier_payment_edit_blocks_amount_below_allocated(client, app):
    login(client)
    with app.app_context():
        sup_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        inv_id = _make_invoice(sup_id, number='INV-P', total=10000)
        pay = SupplierPayment(payment_number=generate_payment_number(),
                              supplier_id=sup_id, payment_date=date.today(),
                              amount=6000)
        db.session.add(pay)
        db.session.flush()
        pay_id = pay.id
        db.session.add(SupplierPaymentAllocation(payment_id=pay_id,
                                                 invoice_id=inv_id, amount=6000))
        db.session.commit()

    # Reducing below the allocated amount is rejected...
    blocked = client.post(f'/procurement/supplier-payments/{pay_id}/edit', data={
        'payment_date': str(date.today()), 'amount': 1000, 'currency': 'INR',
        'method': 'bank', 'reference': '', 'notes': '',
    }, follow_redirects=True)
    assert 'cannot be less than the amount already allocated' in blocked.get_data(as_text=True)

    # ...but a valid increase succeeds.
    ok = client.post(f'/procurement/supplier-payments/{pay_id}/edit', data={
        'payment_date': str(date.today()), 'amount': 7500, 'currency': 'INR',
        'method': 'bank', 'reference': 'NEFT-9', 'notes': '',
    }, follow_redirects=True)
    assert ok.status_code == 200
    with app.app_context():
        p = SupplierPayment.query.get(pay_id)
        assert p.amount == 7500
        assert p.unallocated_amount == 1500


def test_vendor_invoice_and_payment_edit(client, app):
    login(client)
    with app.app_context():
        vend_id = Vendor.query.first().id
        vinv = VendorInvoice(invoice_number='VINV-E', vendor_id=vend_id,
                             invoice_date=date.today(), total_amount=5000,
                             status='posted')
        db.session.add(vinv)
        db.session.flush()
        vinv_id = vinv.id
        vpay = VendorPayment(payment_number=generate_payment_number(
            model=VendorPayment, prefix='VPAY'), vendor_id=vend_id,
            payment_date=date.today(), amount=2000)
        db.session.add(vpay)
        db.session.commit()
        vpay_id = vpay.id

    resp = client.post(f'/procurement/vendor-invoices/{vinv_id}/edit', data={
        'invoice_number': 'VINV-E2', 'invoice_date': str(date.today()),
        'due_date': '', 'work_order_id': '', 'subtotal': 0, 'tax_amount': 0,
        'total_amount': 7500, 'currency': 'INR', 'status': 'posted', 'notes': '',
    }, follow_redirects=True)
    assert resp.status_code == 200
    with app.app_context():
        assert VendorInvoice.query.get(vinv_id).invoice_number == 'VINV-E2'
        assert VendorInvoice.query.get(vinv_id).total_amount == 7500

    resp2 = client.post(f'/procurement/vendor-payments/{vpay_id}/edit', data={
        'payment_date': str(date.today()), 'amount': 3000, 'currency': 'INR',
        'method': 'cash', 'reference': '', 'notes': '',
    }, follow_redirects=True)
    assert resp2.status_code == 200
    with app.app_context():
        assert VendorPayment.query.get(vpay_id).amount == 3000
