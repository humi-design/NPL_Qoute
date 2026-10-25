"""Inventory, costing and supplier-quote procurement tests.

Exercises the real Flask app, real models, real routes and real templates
against the app's existing 'testing' config (SQLite, CSRF disabled). No mocks.

Focus areas:
  * stock position groups raw material / WIP / subcontractor / finished goods /
    rejected / scrap, and stays derived from the existing ledgers
  * the unified stock ledger merges movements, transactions, transfers and
    adjustments with their traceable references
  * a movement can be reversed by a compensating adjustment (history is never
    edited) and cannot be reversed twice
  * negative stock / over-allocation are blocked unless an admin authorises it
  * stock transfers move stock between locations and cannot over-transfer
  * order costing compares the quotation estimate with recorded actuals and
    flags missing inputs instead of inventing them
  * supplier quote comparison scores and ranks comparable quotes, recommends
    one, and requires an explicit admin award before a PO can be raised
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Customer, Material, Vendor, Supplier, SupplierType, Product,
    Order, OrderOperation, OperationTransaction, MaterialLot, MaterialMovement,
    PackagingRecord, Quotation, QuotationItem, StockLocation, StockTransfer,
    StockAdjustment, OrderCostEntry, SupplierRFQ, SupplierQuote,
    SupplierPurchaseOrder,
)
from app.utils import (
    generate_order_number, generate_lot_number, generate_transfer_number,
    generate_adjustment_number, generate_supplier_rfq_number,
    lot_position, compute_inventory_position, build_stock_ledger,
    get_stock_policy, compute_order_cost, compare_supplier_quotes,
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


def login(client, username='admin', password='admin123'):
    return client.post('/auth/login', data={'username': username, 'password': password},
                       follow_redirects=True)


def _seed():
    admin = User(username='admin', email='admin@test.com', full_name='Admin',
                 role='admin',
                 password_hash=bcrypt.generate_password_hash('admin123').decode())
    viewer = User(username='viewer', email='viewer@test.com', full_name='Viewer',
                  role='read_only',
                  password_hash=bcrypt.generate_password_hash('view123').decode())
    db.session.add_all([admin, viewer])

    cust = Customer(customer_code='CUST-0001', company_name='Acme Fasteners')
    mat = Material(material_code='MAT-0001', material_name='SS304', density=8000)
    vend = Vendor(vendor_code='VND-0001', vendor_name='Alpha Heat Treat')
    st = SupplierType(type_code='SUPT-001', type_name='Raw Material')
    sup = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel')
    sup2 = Supplier(supplier_code='SUP-0002', company_name='Jindal Metals')
    db.session.add_all([cust, mat, vend, st, sup, sup2])
    db.session.commit()


def _make_order(qty=100, due_days=10):
    order = Order(
        order_number=generate_order_number(),
        customer_id=Customer.query.first().id,
        part_description='Hex Bolt M10', drawing_number='DWG-1', revision='A',
        material_grade='SS304', quantity_ordered=qty, unit='pcs', rate=12.5,
        order_date=date.today(), due_date=date.today() + timedelta(days=due_days),
        status='confirmed', priority='normal')
    db.session.add(order)
    db.session.commit()
    return order


def _make_lot(received=1000, order=None):
    lot = MaterialLot(lot_number=generate_lot_number(),
                      material_id=Material.query.first().id,
                      order_id=order.id if order else None,
                      grade='SS304', heat_number='HT-1', batch_number='B-1',
                      quantity_received=received, unit='kg',
                      receipt_date=date.today())
    db.session.add(lot)
    db.session.commit()
    return lot


def _make_location(code, kind):
    loc = StockLocation(code=code, name=code, kind=kind)
    db.session.add(loc)
    db.session.commit()
    return loc


# ============================================================
# Stock position
# ============================================================

def test_lot_position_derives_from_movements(app):
    lot = _make_lot(received=1000)
    db.session.add_all([
        MaterialMovement(lot_id=lot.id, movement_type='allocation',
                         movement_date=date.today(), quantity=200, reference='A1'),
        MaterialMovement(lot_id=lot.id, movement_type='issue',
                         movement_date=date.today(), quantity=300, reference='I1'),
        MaterialMovement(lot_id=lot.id, movement_type='scrap',
                         movement_date=date.today(), quantity=50, reference='S1'),
    ])
    db.session.commit()
    pos = lot_position(lot)
    assert pos['received'] == 1000
    assert pos['issued'] == 300
    assert pos['scrapped'] == 50
    assert pos['allocated'] == 200
    assert pos['available'] == 650        # 1000 - 300 - 50
    assert pos['remaining'] == 450        # available - allocated


def test_lot_position_includes_adjustments(app):
    lot = _make_lot(received=100)
    adj1 = StockAdjustment(adjustment_number=generate_adjustment_number(),
                           adjustment_date=date.today(), material_lot_id=lot.id,
                           adjustment_type='increase', quantity=20, unit='kg',
                           reason='found stock')
    db.session.add(adj1)
    db.session.flush()
    db.session.add(StockAdjustment(adjustment_number=generate_adjustment_number(),
                                   adjustment_date=date.today(), material_lot_id=lot.id,
                                   adjustment_type='write_off', quantity=5, unit='kg',
                                   reason='damaged'))
    db.session.commit()
    pos = lot_position(lot)
    assert pos['adjustment'] == 15        # +20 - 5
    assert pos['available'] == 115


def test_inventory_position_groups_all_categories(app):
    order = _make_order(qty=500)
    lot = _make_lot(received=1000, order=order)
    # In-house op with outstanding WIP, vendor op holding stock.
    op_house = OrderOperation(order_id=order.id, sequence=1,
                              operation_name='Forging', quantity_planned=500)
    op_vendor = OrderOperation(order_id=order.id, sequence=2,
                               operation_name='Plating', quantity_planned=500,
                               is_vendor=True, vendor_id=Vendor.query.first().id)
    db.session.add_all([op_house, op_vendor])
    db.session.commit()
    db.session.add_all([
        OperationTransaction(order_operation_id=op_house.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=500),
        OperationTransaction(order_operation_id=op_vendor.id, transaction_type='issue',
                             transaction_date=date.today(), quantity=300),
        PackagingRecord(order_id=order.id, packaging_date=date.today(),
                        quantity_packed=200, quantity_accepted=200,
                        status='warehoused'),
    ])
    db.session.commit()

    totals = compute_inventory_position()
    assert totals['raw_material'] == 1000
    assert totals['wip'] == 500
    assert totals['subcontractor'] == 300
    assert totals['finished_goods'] == 200


def test_stock_position_page_renders(app, client):
    login(client)
    _make_lot(received=100)
    resp = client.get('/inventory/')
    assert resp.status_code == 200
    assert b'Stock Position' in resp.data


def test_stock_position_requires_login(client):
    resp = client.get('/inventory/', follow_redirects=False)
    assert resp.status_code in (301, 302)


# ============================================================
# Stock ledger
# ============================================================

def test_stock_ledger_merges_sources(app):
    order = _make_order()
    lot = _make_lot(received=500, order=order)
    db.session.add(MaterialMovement(lot_id=lot.id, movement_type='receipt',
                                    movement_date=date.today(), quantity=500,
                                    reference=lot.lot_number))
    db.session.add(StockTransfer(
        transfer_number=generate_transfer_number(),
        from_location_id=_make_location('RM', 'raw_material').id,
        to_location_id=_make_location('WIP', 'wip').id,
        transfer_date=date.today(), quantity=100, unit='kg',
        material_lot_id=lot.id, order_id=order.id, status='received'))
    db.session.add(StockAdjustment(
        adjustment_number=generate_adjustment_number(), adjustment_date=date.today(),
        material_lot_id=lot.id, adjustment_type='decrease', quantity=5, unit='kg',
        reason='count'))
    db.session.commit()

    events = build_stock_ledger(lot=lot)
    kinds = {e['kind'] for e in events}
    assert 'movement_receipt' in kinds
    assert 'transfer' in kinds
    assert 'adjustment_decrease' in kinds


def test_stock_ledger_page_and_export(app, client):
    login(client)
    lot = _make_lot(received=100)
    db.session.add(MaterialMovement(lot_id=lot.id, movement_type='receipt',
                                    movement_date=date.today(), quantity=100,
                                    reference=lot.lot_number))
    db.session.commit()
    resp = client.get('/inventory/ledger')
    assert resp.status_code == 200
    assert lot.lot_number.encode() in resp.data

    resp = client.get('/inventory/ledger/export')
    assert resp.status_code == 200
    assert 'spreadsheet' in resp.headers['Content-Type']


# ============================================================
# Reversal via compensating adjustment
# ============================================================

def test_reverse_issue_adds_stock_back(app, client):
    login(client)
    lot = _make_lot(received=100)
    mv = MaterialMovement(lot_id=lot.id, movement_type='issue',
                          movement_date=date.today(), quantity=40, reference='I1')
    db.session.add(mv)
    db.session.commit()
    assert lot.quantity_available == 60

    resp = client.post(f'/inventory/movements/{mv.id}/reverse', follow_redirects=True)
    assert resp.status_code == 200
    adj = StockAdjustment.query.filter_by(reverses_movement_id=mv.id).first()
    assert adj is not None
    assert adj.adjustment_type == 'reversal'
    assert adj.quantity == 40
    # Stock is back to 100 because the issue is netted against the reversal.
    assert lot_position(lot)['available'] == 100


def test_movement_cannot_be_reversed_twice(app, client):
    login(client)
    lot = _make_lot(received=100)
    mv = MaterialMovement(lot_id=lot.id, movement_type='issue',
                          movement_date=date.today(), quantity=10, reference='I1')
    db.session.add(mv)
    db.session.commit()
    client.post(f'/inventory/movements/{mv.id}/reverse', follow_redirects=True)
    client.post(f'/inventory/movements/{mv.id}/reverse', follow_redirects=True)
    assert StockAdjustment.query.filter_by(reverses_movement_id=mv.id).count() == 1


def test_receipt_movement_cannot_be_reversed(app, client):
    login(client)
    lot = _make_lot(received=100)
    mv = MaterialMovement(lot_id=lot.id, movement_type='receipt',
                          movement_date=date.today(), quantity=100,
                          reference=lot.lot_number)
    db.session.add(mv)
    db.session.commit()
    client.post(f'/inventory/movements/{mv.id}/reverse', follow_redirects=True)
    assert StockAdjustment.query.filter_by(reverses_movement_id=mv.id).count() == 0


# ============================================================
# Stock locations, transfers, adjustments
# ============================================================

def test_create_location_via_route(app, client):
    login(client)
    resp = client.post('/inventory/locations/new', data={
        'code': 'STORE-RM', 'name': 'Raw Material Store', 'kind': 'raw_material',
        'is_active': 'y'}, follow_redirects=True)
    assert resp.status_code == 200
    assert StockLocation.query.filter_by(code='STORE-RM').first() is not None


def test_location_code_must_be_unique(app, client):
    login(client)
    _make_location('DUP', 'wip')
    client.post('/inventory/locations/new', data={
        'code': 'DUP', 'name': 'Another', 'kind': 'wip', 'is_active': 'y'},
        follow_redirects=True)
    assert StockLocation.query.filter_by(code='DUP').count() == 1


def test_transfer_moves_stock_between_locations(app, client):
    login(client)
    rm = _make_location('RM', 'raw_material')
    wip = _make_location('WIP', 'wip')
    lot = _make_lot(received=100)
    resp = client.post('/inventory/transfers/new', data={
        'transfer_date': date.today().isoformat(),
        'from_location_id': str(rm.id), 'to_location_id': str(wip.id),
        'material_lot_id': str(lot.id), 'quantity': '40', 'unit': 'kg',
        'status': 'received', 'reference': 'DC-1'}, follow_redirects=True)
    assert resp.status_code == 200
    tr = StockTransfer.query.first()
    assert tr is not None and tr.quantity == 40


def test_transfer_cannot_exceed_available(app, client):
    login(client)
    rm = _make_location('RM', 'raw_material')
    wip = _make_location('WIP', 'wip')
    lot = _make_lot(received=50)
    client.post('/inventory/transfers/new', data={
        'transfer_date': date.today().isoformat(),
        'from_location_id': str(rm.id), 'to_location_id': str(wip.id),
        'material_lot_id': str(lot.id), 'quantity': '500', 'unit': 'kg',
        'status': 'received'}, follow_redirects=True)
    assert StockTransfer.query.count() == 0


def test_same_location_transfer_rejected(app, client):
    login(client)
    rm = _make_location('RM', 'raw_material')
    lot = _make_lot(received=100)
    resp = client.post('/inventory/transfers/new', data={
        'transfer_date': date.today().isoformat(),
        'from_location_id': str(rm.id), 'to_location_id': str(rm.id),
        'material_lot_id': str(lot.id), 'quantity': '10', 'unit': 'kg',
        'status': 'received'}, follow_redirects=True)
    assert resp.status_code == 200
    assert StockTransfer.query.count() == 0


def test_adjustment_beyond_available_needs_admin(app, client):
    """A read-only user cannot reduce a lot below zero."""
    login(client, 'viewer', 'view123')
    lot = _make_lot(received=10)
    # read_only has no write permission at all, so nothing is created.
    client.post('/inventory/adjustments/new', data={
        'adjustment_date': date.today().isoformat(),
        'material_lot_id': str(lot.id), 'adjustment_type': 'decrease',
        'quantity': '100', 'unit': 'kg', 'reason': 'test'},
        follow_redirects=True)
    assert StockAdjustment.query.count() == 0


def test_admin_can_authorise_over_adjustment(app, client):
    login(client)
    lot = _make_lot(received=10)
    resp = client.post('/inventory/adjustments/new', data={
        'adjustment_date': date.today().isoformat(),
        'material_lot_id': str(lot.id), 'adjustment_type': 'decrease',
        'quantity': '100', 'unit': 'kg', 'reason': 'stock take correction'},
        follow_redirects=True)
    assert resp.status_code == 200
    adj = StockAdjustment.query.first()
    assert adj is not None
    assert adj.authorized_by is not None  # out-of-policy adjustment authorised


def test_stock_policy_defaults_are_safe(app):
    policy = get_stock_policy()
    assert policy['allow_negative_stock'] is False
    assert policy['allow_over_allocation'] is False


def test_location_with_history_cannot_be_deleted(app, client):
    login(client)
    rm = _make_location('RM', 'raw_material')
    wip = _make_location('WIP', 'wip')
    lot = _make_lot(received=100)
    db.session.add(StockTransfer(
        transfer_number=generate_transfer_number(),
        from_location_id=rm.id, to_location_id=wip.id, transfer_date=date.today(),
        quantity=10, unit='kg', material_lot_id=lot.id, status='received'))
    db.session.commit()
    client.post(f'/inventory/locations/{rm.id}/delete', follow_redirects=True)
    assert StockLocation.query.filter_by(code='RM').first() is not None


# ============================================================
# Costing
# ============================================================

def test_order_cost_compares_estimate_and_actual(app):
    order = _make_order(qty=100)
    quotation = Quotation(quotation_number='Q-1', customer_id=Customer.query.first().id,
                          quotation_date=date.today(), status='won')
    db.session.add(quotation)
    db.session.flush()
    item = QuotationItem(quotation_id=quotation.id, quantity_1=100,
                         material_cost=10, machining_cost=2, vendor_cost=1,
                         other_cost=0.5)
    db.session.add(item)
    db.session.flush()
    order.quotation_id = quotation.id
    order.quotation_item_id = item.id
    db.session.add_all([
        OrderCostEntry(order_id=order.id, category='material', amount=1100,
                       currency='INR', amount_base=1100, entry_date=date.today(),
                       description='material'),
        OrderCostEntry(order_id=order.id, category='machining', amount=250,
                       currency='INR', amount_base=250, entry_date=date.today(),
                       description='machining'),
    ])
    db.session.commit()

    cost = compute_order_cost(order)
    # Estimate: material 10*100=1000, machining 200, subcontracting 100, other 50.
    assert cost['estimate_total'] == 1350
    assert cost['actual_total'] == 1350
    assert cost['variance'] == 0
    assert cost['has_estimate'] and cost['has_actual']
    assert cost['unit_cost_actual'] == 13.5


def test_order_cost_flags_missing_inputs(app):
    order = _make_order(qty=100)
    db.session.add(OrderCostEntry(order_id=order.id, category='material',
                                  amount=500, currency='INR', amount_base=500,
                                  entry_date=date.today()))
    db.session.commit()
    cost = compute_order_cost(order)
    assert cost['has_actual']
    assert not cost['has_estimate']
    assert cost['missing']  # flagged, not invented


def test_cost_entry_double_count_prevented(app):
    """UNIQUE(source_type, source_id, category) stops the same document twice."""
    from sqlalchemy.exc import IntegrityError
    order = _make_order()
    db.session.add(OrderCostEntry(order_id=order.id, category='freight',
                                  amount=100, currency='INR', amount_base=100,
                                  entry_date=date.today(), source_type='manual',
                                  source_id=42))
    db.session.commit()
    db.session.add(OrderCostEntry(order_id=order.id, category='freight',
                                  amount=100, currency='INR', amount_base=100,
                                  entry_date=date.today(), source_type='manual',
                                  source_id=42))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_costing_page_and_order_cost_page_render(app, client):
    login(client)
    order = _make_order()
    db.session.add(OrderCostEntry(order_id=order.id, category='material',
                                  amount=100, currency='INR', amount_base=100,
                                  entry_date=date.today()))
    db.session.commit()
    resp = client.get('/costing/')
    assert resp.status_code == 200
    assert order.order_number.encode() in resp.data

    resp = client.get(f'/costing/order/{order.id}')
    assert resp.status_code == 200

    resp = client.get('/costing/missing')
    assert resp.status_code == 200


def test_add_cost_entry_via_route(app, client):
    login(client)
    order = _make_order()
    resp = client.post(f'/costing/order/{order.id}/entry/new', data={
        'category': 'packaging', 'amount': '250.5', 'currency': 'INR',
        'entry_date': date.today().isoformat(), 'description': 'boxes'},
        follow_redirects=True)
    assert resp.status_code == 200
    entry = OrderCostEntry.query.filter_by(order_id=order.id).first()
    assert entry is not None and entry.amount == 250.5


def test_costing_export_returns_xlsx(app, client):
    login(client)
    _make_order()
    resp = client.get('/costing/export')
    assert resp.status_code == 200
    assert 'spreadsheet' in resp.headers['Content-Type']


# ============================================================
# Supplier RFQ / quote comparison / approval
# ============================================================

def _make_rfq(qty=1000, currency='INR'):
    rfq = SupplierRFQ(rfq_number=generate_supplier_rfq_number(),
                      title='SS304 bar', material_id=Material.query.first().id,
                      quantity=qty, unit='kg', grade='SS304',
                      rfq_date=date.today(), due_date=date.today() + timedelta(days=7),
                      status='sent', currency=currency)
    db.session.add(rfq)
    db.session.commit()
    return rfq


def _add_quote(rfq, supplier, price, lead, tax=0, payment=None, delivery=None,
               quality=None):
    q = SupplierQuote(supplier_rfq_id=rfq.id, supplier_id=supplier.id,
                      quote_date=date.today(), quantity=rfq.quantity, unit='kg',
                      unit_price=price, currency='INR', tax_percent=tax,
                      lead_time_days=lead, payment_terms=payment,
                      delivery_terms=delivery, quality_rating=quality)
    db.session.add(q)
    db.session.commit()
    return q


def test_quote_comparison_ranks_and_recommends(app):
    rfq = _make_rfq()
    sup1 = Supplier.query.filter_by(supplier_code='SUP-0001').first()
    sup2 = Supplier.query.filter_by(supplier_code='SUP-0002').first()
    # sup2 is cheaper AND faster -> should be recommended.
    _add_quote(rfq, sup1, 210, 14, tax=18, payment='Net 15', delivery='Ex-works')
    _add_quote(rfq, sup2, 200, 5, tax=18, payment='Net 45', delivery='Delivered')

    comparison = compare_supplier_quotes(rfq)
    assert len(comparison['rows']) == 2
    assert comparison['recommended']['quote'].supplier_id == sup2.id
    assert comparison['rows'][0]['rank'] == 1
    assert comparison['rows'][0]['total_score'] >= comparison['rows'][1]['total_score']


def test_quote_comparison_handles_no_quotes(app):
    rfq = _make_rfq()
    comparison = compare_supplier_quotes(rfq)
    assert comparison['recommended'] is None
    assert comparison['rows'] == []


def test_quote_comparison_flags_missing_inputs(app):
    rfq = _make_rfq()
    sup1 = Supplier.query.first()
    _add_quote(rfq, sup1, 0, None)  # no price, no lead time
    comparison = compare_supplier_quotes(rfq)
    assert comparison['missing']


def test_supplier_quality_history_from_receipts(app):
    from app.models import SupplierOrderReceipt, SupplierOrderReceiptLine
    sup = Supplier.query.filter_by(supplier_code='SUP-0001').first()
    po = SupplierPurchaseOrder(po_number='PO-1', supplier_id=sup.id,
                               order_date=date.today(), status='received')
    db.session.add(po)
    db.session.flush()
    rec = SupplierOrderReceipt(receipt_number='GRN-1', po_id=po.id,
                               supplier_id=sup.id, receipt_date=date.today())
    db.session.add(rec)
    db.session.flush()
    db.session.add(SupplierOrderReceiptLine(receipt_id=rec.id,
                                            quantity_received=100,
                                            quantity_rejected=10))
    db.session.commit()
    from app.utils import supplier_quality_history
    score, note = supplier_quality_history(sup.id)
    assert score == 90.0
    assert 'rejected' in note


def test_rfq_pages_render(app, client):
    login(client)
    rfq = _make_rfq()
    _add_quote(rfq, Supplier.query.first(), 210, 7)
    assert client.get('/procurement/supplier-rfqs').status_code == 200
    assert client.get(f'/procurement/supplier-rfqs/{rfq.id}').status_code == 200
    assert client.get(f'/procurement/supplier-rfqs/{rfq.id}/compare').status_code == 200


def test_compare_export_returns_xlsx(app, client):
    login(client)
    rfq = _make_rfq()
    _add_quote(rfq, Supplier.query.first(), 210, 7)
    resp = client.get(f'/procurement/supplier-rfqs/{rfq.id}/compare/export')
    assert resp.status_code == 200
    assert 'spreadsheet' in resp.headers['Content-Type']


def test_create_rfq_and_add_quote_via_routes(app, client):
    login(client)
    sup = Supplier.query.first()
    resp = client.post('/procurement/supplier-rfqs/new', data={
        'title': 'SS304 bar', 'material_id': str(Material.query.first().id),
        'quantity': '500', 'unit': 'kg', 'grade': 'SS304',
        'rfq_date': date.today().isoformat(), 'currency': 'INR',
        'status': 'sent'}, follow_redirects=True)
    assert resp.status_code == 200
    rfq = SupplierRFQ.query.first()
    assert rfq is not None

    client.post(f'/procurement/supplier-rfqs/{rfq.id}/quote', data={
        'supplier_id': str(sup.id), 'quote_date': date.today().isoformat(),
        'quantity': '500', 'unit': 'kg', 'unit_price': '205',
        'currency': 'INR', 'tax_percent': '18', 'lead_time_days': '6',
        'status': 'received'}, follow_redirects=True)
    assert SupplierQuote.query.filter_by(supplier_rfq_id=rfq.id).count() == 1


def test_award_requires_admin(app, client):
    """A read-only user cannot award a quote."""
    rfq = _make_rfq()
    quote = _add_quote(rfq, Supplier.query.first(), 210, 7)
    login(client, 'viewer', 'view123')
    client.post(f'/procurement/supplier-rfqs/{rfq.id}/award/{quote.id}',
                follow_redirects=True)
    db.session.refresh(rfq)
    assert rfq.awarded_quote_id is None
    assert rfq.status != 'awarded'


def test_admin_award_then_create_po(app, client):
    login(client)
    rfq = _make_rfq()
    sup = Supplier.query.first()
    quote = _add_quote(rfq, sup, 210, 7, tax=18, payment='Net 30',
                       delivery='Delivered')

    resp = client.post(f'/procurement/supplier-rfqs/{rfq.id}/award/{quote.id}',
                       follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(rfq)
    assert rfq.awarded_quote_id == quote.id
    assert rfq.status == 'awarded'
    assert rfq.awarded_by is not None

    # Buyer then raises a PO from the award.
    resp = client.post(f'/procurement/supplier-rfqs/{rfq.id}/create-po',
                       follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(rfq)
    assert rfq.po_id is not None
    po = SupplierPurchaseOrder.query.get(rfq.po_id)
    assert po.supplier_id == sup.id
    assert po.lines.count() == 1
    assert po.lines.first().rate == 210


def test_cannot_create_po_before_award(app, client):
    login(client)
    rfq = _make_rfq()
    _add_quote(rfq, Supplier.query.first(), 210, 7)
    client.post(f'/procurement/supplier-rfqs/{rfq.id}/create-po',
                follow_redirects=True)
    db.session.refresh(rfq)
    assert rfq.po_id is None


def test_cannot_delete_rfq_with_po(app, client):
    login(client)
    rfq = _make_rfq()
    sup = Supplier.query.first()
    quote = _add_quote(rfq, sup, 210, 7)
    client.post(f'/procurement/supplier-rfqs/{rfq.id}/award/{quote.id}',
                follow_redirects=True)
    client.post(f'/procurement/supplier-rfqs/{rfq.id}/create-po',
                follow_redirects=True)
    client.post(f'/procurement/supplier-rfqs/{rfq.id}/delete', follow_redirects=True)
    assert SupplierRFQ.query.get(rfq.id) is not None
