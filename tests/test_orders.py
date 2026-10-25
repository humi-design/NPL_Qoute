"""Orders & manufacturing tracking tests.

Exercises the real Flask app, real models, real routes and real templates
against the app's existing 'testing' config (SQLite, CSRF disabled). No mocks.

Focus areas:
  * order register views (current / past / all) and status summaries
  * order-specific manufacturing route with partial deliveries
  * operation ledgers: issued / returned / accepted / rejected / rework / scrap
    all reconcile, and cannot exceed what was issued
  * raw material tracker: allocation, issue, return, scrap, shortage balances
  * over-allocation and over-issue are blocked
  * duplicate movements (same reference) are rejected, not double counted
  * inspections (incoming / in-process / final) and packaging -> completion
  * a traceable timeline from material procurement to dispatch
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Customer, Material, Vendor, Process, Product, Machine,
    Order, OrderOperation, OperationTransaction, MaterialLot, MaterialMovement,
    Inspection, PackagingRecord, Supplier,
)
from app.utils import (
    generate_order_number, generate_inspection_number, generate_lot_number,
    compute_order_summary, compute_material_requirement, build_order_timeline,
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

    cust = Customer(customer_code='CUST-0001', company_name='Acme Fasteners')
    mat = Material(material_code='MAT-0001', material_name='SS304', density=8000)
    vend = Vendor(vendor_code='VND-0001', vendor_name='Alpha Heat Treat')
    proc = Process(process_code='PRC-0001', process_name='Heat Treatment',
                   process_type='vendor')
    prod = Product(internal_code='PRD-00001', product_name='Hex Bolt M10')
    mach = Machine(machine_code='MCH-0001', machine_name='CNC-1', hourly_rate=500)
    sup = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel')
    db.session.add_all([cust, mat, vend, proc, prod, mach, sup])
    db.session.commit()


def _make_order(app, qty=100, due_days=10):
    """Create an order directly through the model layer."""
    order = Order(
        order_number=generate_order_number(),
        customer_id=Customer.query.first().id,
        part_description='Hex Bolt M10', drawing_number='DWG-1', revision='A',
        material_grade='SS304', quantity_ordered=qty, unit='pcs', rate=12.5,
        order_date=date.today(), due_date=date.today() + timedelta(days=due_days),
        status='confirmed', priority='normal',
    )
    db.session.add(order)
    db.session.commit()
    return order


def _make_op(order, sequence, name, planned=100, is_vendor=False, is_final=False,
             vendor_id=None):
    op = OrderOperation(order_id=order.id, sequence=sequence, operation_name=name,
                        quantity_planned=planned, is_vendor=is_vendor,
                        vendor_id=vendor_id, is_final=is_final)
    db.session.add(op)
    db.session.commit()
    return op


# ============================================================
# Order register
# ============================================================

def test_order_number_generation_is_sequential(app):
    o1 = _make_order(app)
    o2 = _make_order(app)
    assert o1.order_number.startswith('SO-')
    assert o1.order_number != o2.order_number


def test_current_past_all_views(app, client):
    login(client)
    open_order = _make_order(app, qty=100)
    done = _make_order(app, qty=50)
    done.status = 'completed'
    db.session.commit()

    # Current shows only the open order
    resp = client.get('/orders/?view=current')
    assert resp.status_code == 200
    assert open_order.order_number.encode() in resp.data
    assert done.order_number.encode() not in resp.data

    # Past shows only the completed order
    resp = client.get('/orders/?view=past')
    assert done.order_number.encode() in resp.data
    assert open_order.order_number.encode() not in resp.data

    # All shows both
    resp = client.get('/orders/?view=all')
    assert open_order.order_number.encode() in resp.data
    assert done.order_number.encode() in resp.data


def test_order_search_and_status_summary(app, client):
    login(client)
    _make_order(app, qty=100)
    _make_order(app, qty=200)
    resp = client.get('/orders/?view=all&search=Hex')
    assert resp.status_code == 200
    assert b'Hex Bolt' in resp.data


def test_order_requires_login(client):
    resp = client.get('/orders/', follow_redirects=False)
    assert resp.status_code in (302, 401)


# ============================================================
# Operation ledger / partial deliveries
# ============================================================

def test_partial_issue_and_return_reconcile(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Turning', planned=100)

    # Two partial deliveries to the vendor, then a partial return.
    db.session.add_all([
        OperationTransaction(order_operation_id=op.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=60,
                             reference='DC-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=40,
                             reference='DC-2'),
        OperationTransaction(order_operation_id=op.id, transaction_type='return',
                             transaction_date=date.today(), quantity=70,
                             reference='RET-1'),
    ])
    db.session.commit()

    assert op.issued_quantity == 100
    assert op.returned_quantity == 70
    assert op.pending_quantity == 30  # 100 issued - 70 returned - 0 rej - 0 scrap


def test_accepted_rejected_rework_scrap_derive(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Grinding', planned=100)
    db.session.add(OperationTransaction(
        order_operation_id=op.id, transaction_type='issue',
        transaction_date=date.today(), quantity=100, reference='DC-1'))
    db.session.add_all([
        OperationTransaction(order_operation_id=op.id, transaction_type='return',
                             transaction_date=date.today(), quantity=100,
                             reference='RET-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='accept',
                             transaction_date=date.today(), quantity=85,
                             reference='ACC-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='reject',
                             transaction_date=date.today(), quantity=10,
                             reference='REJ-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='rework',
                             transaction_date=date.today(), quantity=5,
                             reference='RWK-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='scrap',
                             transaction_date=date.today(), quantity=5,
                             reference='SCR-1'),
    ])
    db.session.commit()

    assert op.accepted_quantity == 85
    assert op.rejected_quantity == 10
    assert op.rework_quantity == 5
    assert op.scrapped_quantity == 5


def test_return_cannot_exceed_issued_via_route(app, client):
    login(client)
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Turning', planned=100)
    client.post(f'/orders/operation/{op.id}/transaction', data={
        'transaction_type': 'issue', 'transaction_date': str(date.today()),
        'quantity': 50, 'reference': 'DC-1'}, follow_redirects=True)

    # Try to return 80 when only 50 were issued.
    resp = client.post(f'/orders/operation/{op.id}/transaction', data={
        'transaction_type': 'return', 'transaction_date': str(date.today()),
        'quantity': 80, 'reference': 'RET-1'}, follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(op)
    assert op.returned_quantity == 0
    assert b'outstanding' in resp.data.lower()


def test_duplicate_operation_movement_is_rejected(app, client):
    login(client)
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Turning', planned=100)
    data = {'transaction_type': 'issue', 'transaction_date': str(date.today()),
            'quantity': 50, 'reference': 'DC-DUP'}
    client.post(f'/orders/operation/{op.id}/transaction', data=data,
                follow_redirects=True)
    resp = client.post(f'/orders/operation/{op.id}/transaction', data=data,
                       follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(op)
    assert op.issued_quantity == 50  # posted once only
    assert b'already been posted' in resp.data.lower()


def test_order_progress_reflects_final_operation(app):
    order = _make_order(app, qty=100)
    op1 = _make_op(order, 1, 'Turning', planned=100)
    op2 = _make_op(order, 2, 'Heat Treatment', planned=100, is_vendor=True,
                   vendor_id=Vendor.query.first().id, is_final=True)
    db.session.add_all([
        OperationTransaction(order_operation_id=op2.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=100,
                             reference='DC-1'),
        OperationTransaction(order_operation_id=op2.id, transaction_type='accept',
                             transaction_date=date.today(), quantity=90,
                             reference='ACC-1'),
    ])
    db.session.commit()
    assert order.quantity_produced == 90
    assert order.pending_quantity == 10
    assert order.completion_percent == 90.0


def test_completion_percent_when_over_produced_is_capped(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Final', planned=100, is_final=True)
    db.session.add(OperationTransaction(
        order_operation_id=op.id, transaction_type='accept',
        transaction_date=date.today(), quantity=150, reference='ACC-1'))
    db.session.commit()
    assert order.completion_percent == 100.0


# ============================================================
# Raw material tracker & allocation
# ============================================================

def _make_lot(app, received=1000, unit='kg'):
    lot = MaterialLot(
        lot_number=generate_lot_number(),
        material_id=Material.query.first().id,
        supplier_id=Supplier.query.first().id,
        grade='AISI 304', heat_number='H-1', batch_number='B-1',
        certificate_ref='CERT-1', quantity_received=received, unit=unit,
        receipt_date=date.today(),
    )
    db.session.add(lot)
    db.session.flush()
    db.session.add(MaterialMovement(lot_id=lot.id, movement_type='receipt',
                                    movement_date=date.today(), quantity=received,
                                    reference=lot.lot_number))
    db.session.commit()
    return lot


def test_lot_balances_derive_from_movements(app):
    order = _make_order(app)
    lot = _make_lot(app, received=1000)
    db.session.add_all([
        MaterialMovement(lot_id=lot.id, movement_type='allocation',
                         movement_date=date.today(), quantity=200,
                         reference='ALLOC-1', order_id=order.id),
        MaterialMovement(lot_id=lot.id, movement_type='issue',
                         movement_date=date.today(), quantity=300,
                         reference='ISS-1', order_id=order.id),
        MaterialMovement(lot_id=lot.id, movement_type='return',
                         movement_date=date.today(), quantity=50,
                         reference='RET-1', order_id=order.id),
        MaterialMovement(lot_id=lot.id, movement_type='scrap',
                         movement_date=date.today(), quantity=20,
                         reference='SCR-1', order_id=order.id),
    ])
    db.session.commit()

    assert lot.quantity_received == 1000
    assert lot.quantity_allocated == 200
    assert lot.quantity_issued == 300
    assert lot.quantity_returned == 50
    assert lot.quantity_scrapped == 20
    # available = received - issued - scrapped
    assert lot.quantity_available == 1000 - 300 - 20
    # remaining = available - allocated
    assert lot.quantity_remaining == 680 - 200


def test_over_allocation_is_blocked(app, client):
    login(client)
    lot = _make_lot(app, received=100)
    resp = client.post(f'/orders/materials/{lot.id}/movement', data={
        'movement_type': 'allocation', 'movement_date': str(date.today()),
        'quantity': 500, 'reference': 'ALLOC-OVER'}, follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(lot)
    assert lot.quantity_allocated == 0
    assert b'free to allocate' in resp.data.lower()


def test_over_issue_is_blocked(app, client):
    login(client)
    lot = _make_lot(app, received=100)
    resp = client.post(f'/orders/materials/{lot.id}/movement', data={
        'movement_type': 'issue', 'movement_date': str(date.today()),
        'quantity': 250, 'reference': 'ISS-OVER'}, follow_redirects=True)
    db.session.refresh(lot)
    assert lot.quantity_issued == 0
    assert b'available to issue' in resp.data.lower()


def test_duplicate_material_movement_is_rejected(app, client):
    login(client)
    lot = _make_lot(app, received=100)
    data = {'movement_type': 'issue', 'movement_date': str(date.today()),
            'quantity': 40, 'reference': 'ISS-DUP'}
    client.post(f'/orders/materials/{lot.id}/movement', data=data,
                follow_redirects=True)
    resp = client.post(f'/orders/materials/{lot.id}/movement', data=data,
                       follow_redirects=True)
    db.session.refresh(lot)
    assert lot.quantity_issued == 40  # counted once
    assert b'already been posted' in resp.data.lower()


def test_shortage_is_tracked(app):
    order = _make_order(app)
    lot = _make_lot(app, received=100)
    db.session.add(MaterialMovement(lot_id=lot.id, movement_type='shortage',
                                    movement_date=date.today(), quantity=25,
                                    reference='SHORT-1', order_id=order.id))
    db.session.commit()
    assert lot.quantity_shortage == 25
    req = compute_material_requirement(order)
    assert req['shortage'] == 25


def test_material_requirement_planned_vs_issued(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Turning', planned=100)
    lot = _make_lot(app, received=500)
    db.session.add(MaterialMovement(lot_id=lot.id, movement_type='issue',
                                    movement_date=date.today(), quantity=60,
                                    reference='ISS-1', order_id=order.id,
                                    order_operation_id=op.id))
    db.session.commit()
    req = compute_material_requirement(order)
    assert req['planned'] == 100
    assert req['issued'] == 60
    assert req['outstanding'] == 40


# ============================================================
# Inspections
# ============================================================

def test_inspection_number_generation(app):
    n1 = generate_inspection_number()
    assert n1.startswith('INS-')
    db.session.add(Inspection(inspection_number=n1, order_id=_make_order(app).id,
                              inspection_type='final', inspection_date=date.today(),
                              result='pending'))
    db.session.commit()
    assert generate_inspection_number() != n1


def test_inspection_accepted_plus_rejected_cannot_exceed_inspected(app, client):
    login(client)
    order = _make_order(app)
    resp = client.post(f'/orders/{order.id}/inspection/new', data={
        'inspection_type': 'final', 'inspection_date': str(date.today()),
        'quantity_inspected': 100, 'quantity_accepted': 90,
        'quantity_rejected': 30, 'result': 'partial'}, follow_redirects=True)
    assert resp.status_code == 200
    assert Inspection.query.filter_by(order_id=order.id).count() == 0
    assert b'cannot exceed' in resp.data.lower()


def test_inspection_recorded_and_linked(app, client):
    login(client)
    order = _make_order(app)
    op = _make_op(order, 1, 'Final', planned=100, is_final=True)
    resp = client.post(f'/orders/{order.id}/inspection/new', data={
        'inspection_type': 'in_process', 'inspection_date': str(date.today()),
        'order_operation_id': str(op.id), 'tolerance': '±0.05',
        'acceptance_criteria': 'No burrs', 'measurements': '10.02, 10.03',
        'sample_size': 5, 'quantity_inspected': 100, 'quantity_accepted': 95,
        'quantity_rejected': 5, 'result': 'partial'}, follow_redirects=True)
    assert resp.status_code == 200
    insp = Inspection.query.filter_by(order_id=order.id).one()
    assert insp.inspection_type == 'in_process'
    assert insp.tolerance == '±0.05'
    assert insp.quantity_accepted == 95
    assert insp.order_operation_id == op.id


def test_inspections_list_filters(app, client):
    login(client)
    order = _make_order(app)
    db.session.add(Inspection(inspection_number=generate_inspection_number(),
                              order_id=order.id, inspection_type='final',
                              inspection_date=date.today(), result='passed'))
    db.session.commit()
    resp = client.get('/orders/inspections?type=final&result=passed')
    assert resp.status_code == 200


# ============================================================
# Packaging & completion
# ============================================================

def test_packaging_completes_order_when_full_quantity_warehoused(app, client):
    login(client)
    order = _make_order(app, qty=100)
    resp = client.post(f'/orders/{order.id}/packaging/new', data={
        'packaging_date': str(date.today()), 'packaging_type': 'box',
        'quantity_packed': 100, 'quantity_accepted': 100, 'quantity_rejected': 0,
        'warehouse_receipt_number': 'WR-1',
        'warehouse_receipt_date': str(date.today()), 'location': 'A1',
        'status': 'warehoused'}, follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(order)
    assert order.status == 'completed'
    assert order.actual_completion == date.today()
    assert order.quantity_warehoused == 100
    assert order.quantity_dispatched == 100


def test_packaging_does_not_complete_order_when_partial(app, client):
    login(client)
    order = _make_order(app, qty=100)
    client.post(f'/orders/{order.id}/packaging/new', data={
        'packaging_date': str(date.today()), 'packaging_type': 'box',
        'quantity_packed': 40, 'quantity_accepted': 40, 'quantity_rejected': 0,
        'warehouse_receipt_number': 'WR-1', 'status': 'warehoused'},
        follow_redirects=True)
    db.session.refresh(order)
    assert order.status != 'completed'
    assert order.quantity_warehoused == 40
    assert order.quantity_dispatched == 40


# ============================================================
# Timeline
# ============================================================

def test_timeline_traces_material_to_dispatch(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Heat Treatment', planned=100, is_vendor=True,
                  vendor_id=Vendor.query.first().id, is_final=True)
    lot = _make_lot(app, received=200)
    lot.order_id = order.id
    db.session.add_all([
        OperationTransaction(order_operation_id=op.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=100,
                             reference='DC-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='accept',
                             transaction_date=date.today(), quantity=95,
                             reference='ACC-1'),
        Inspection(inspection_number=generate_inspection_number(), order_id=order.id,
                   inspection_type='final', inspection_date=date.today(),
                   quantity_inspected=95, quantity_accepted=95, result='passed'),
        PackagingRecord(order_id=order.id, packaging_date=date.today(),
                        quantity_packed=95, quantity_accepted=95,
                        warehouse_receipt_number='WR-1',
                        warehouse_receipt_date=date.today(), status='warehoused'),
    ])
    db.session.commit()

    timeline = build_order_timeline(order)
    kinds = [e['kind'] for e in timeline]
    assert 'material' in kinds
    assert 'inspection' in kinds
    assert 'packaging' in kinds
    assert 'warehouse' in kinds
    # Chronological ordering
    dates = [e['sort_key'] for e in timeline]
    assert dates == sorted(dates)


def test_order_summary_matches_detail_ledgers(app):
    order = _make_order(app, qty=100)
    op = _make_op(order, 1, 'Final', planned=100, is_final=True)
    db.session.add_all([
        OperationTransaction(order_operation_id=op.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=100,
                             reference='DC-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='accept',
                             transaction_date=date.today(), quantity=92,
                             reference='ACC-1'),
        OperationTransaction(order_operation_id=op.id, transaction_type='reject',
                             transaction_date=date.today(), quantity=8,
                             reference='REJ-1'),
    ])
    db.session.commit()
    summary = compute_order_summary(order)
    assert summary['quantity_produced'] == 92
    assert summary['quantity_rejected'] == 8
    assert summary['pending_quantity'] == 8
    assert summary['operations_complete'] == 0  # 92 < 100 planned


# ============================================================
# Route-level creation
# ============================================================

def test_create_order_via_route(app, client):
    login(client)
    cust_id = Customer.query.first().id
    resp = client.post('/orders/new', data={
        'customer_id': str(cust_id), 'part_description': 'Stud M12',
        'drawing_number': 'DWG-9', 'quantity_ordered': 250, 'unit': 'pcs',
        'rate': 8, 'currency': 'INR', 'order_date': str(date.today()),
        'status': 'confirmed', 'priority': 'high'}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    order = Order.query.filter_by(part_description='Stud M12').one()
    assert order.quantity_ordered == 250
    assert order.order_number.startswith('SO-')


def test_create_order_without_customer_is_rejected(app, client):
    login(client)
    resp = client.post('/orders/new', data={
        'customer_id': '', 'quantity_ordered': 10,
        'order_date': str(date.today())}, follow_redirects=True)
    assert resp.status_code == 200
    assert Order.query.count() == 0


def test_add_operation_and_transaction_via_routes(app, client):
    login(client)
    order = _make_order(app, qty=100)
    vend_id = Vendor.query.first().id
    client.post(f'/orders/{order.id}/operation/add', data={
        'sequence': 1, 'operation_name': 'Zinc Plating', 'is_vendor': 'y',
        'vendor_id': str(vend_id), 'quantity_planned': 100,
        'status': 'pending'}, follow_redirects=True)
    op = OrderOperation.query.filter_by(order_id=order.id).one()
    assert op.is_vendor is True
    assert op.vendor_id == vend_id

    client.post(f'/orders/operation/{op.id}/transaction', data={
        'transaction_type': 'issue', 'transaction_date': str(date.today()),
        'quantity': 100, 'reference': 'DC-1'}, follow_redirects=True)
    db.session.refresh(op)
    assert op.issued_quantity == 100


def test_delete_order_blocked_when_route_exists(app, client):
    login(client)
    order = _make_order(app, qty=100)
    _make_op(order, 1, 'Turning', planned=100)
    resp = client.post(f'/orders/{order.id}/delete', follow_redirects=True)
    assert resp.status_code == 200
    assert Order.query.get(order.id) is not None
    assert b'cannot delete' in resp.data.lower()


def test_set_order_status_route(app, client):
    login(client)
    order = _make_order(app, qty=100)
    client.post(f'/orders/{order.id}/status/completed', follow_redirects=True)
    db.session.refresh(order)
    assert order.status == 'completed'
    assert order.actual_completion == date.today()


def test_order_export_returns_xlsx(app, client):
    login(client)
    _make_order(app, qty=100)
    resp = client.get('/orders/export?view=all')
    assert resp.status_code == 200
    assert resp.data[:2] == b'PK'  # xlsx is a zip archive


def test_material_lot_export_returns_xlsx(app, client):
    login(client)
    _make_lot(app, received=500)
    resp = client.get('/orders/materials/export')
    assert resp.status_code == 200
    assert resp.data[:2] == b'PK'


def test_vendor_process_export_returns_xlsx(app, client):
    login(client)
    order = _make_order(app, qty=100)
    _make_op(order, 1, 'Heat Treatment', planned=100, is_vendor=True,
             vendor_id=Vendor.query.first().id)
    resp = client.get('/orders/vendor-processes/export')
    assert resp.status_code == 200
    assert resp.data[:2] == b'PK'


def test_new_order_prefills_from_quotation(app, client):
    """A won quotation can be promoted to an order with prefilled fields."""
    login(client)
    from app.models import Quotation, QuotationItem
    cust = Customer.query.first()
    q = Quotation(quotation_number='QT-2026-0001', customer_id=cust.id,
                  quotation_date=date.today(), status='won')
    db.session.add(q)
    db.session.flush()
    db.session.add(QuotationItem(quotation_id=q.id, part_description='Flange Nut',
                                 drawing_number='DWG-Q1', material='EN8',
                                 quantity_1=500))
    db.session.commit()

    resp = client.get(f'/orders/new?quotation_id={q.id}')
    assert resp.status_code == 200
    assert b'Flange Nut' in resp.data
    assert b'DWG-Q1' in resp.data
