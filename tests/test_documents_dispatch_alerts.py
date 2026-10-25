"""Document management, dispatch, alerts and dashboard tests.

Exercises the real Flask app, real models, real routes and real templates
against the app's existing 'testing' config (SQLite, CSRF disabled). No mocks.

Focus areas:
  * documents link to the right record, carry a category and revision, and the
    revision in force is identifiable (superseded revisions are preserved)
  * a document can actually be downloaded (the gap this phase closes) and the
    download refuses paths that escape the upload root
  * dispatch ships only what the warehouse holds, supports partial shipments,
    releases stock when a line is removed, and never marks the order complete
    early
  * alerts are derived from live records (so they cannot go stale), are not
    duplicated, and acknowledgement is per-user
  * dashboard operational metrics are real sums/counts, and financial figures
    are hidden from users without the permission
"""
import io
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Customer, Material, Vendor, Supplier, SupplierType, Product,
    Order, OrderOperation, OperationTransaction, MaterialLot, PackagingRecord,
    Attachment, Dispatch, DispatchLine, AlertAcknowledgement, Inspection,
)
from app.utils import (
    generate_order_number, generate_lot_number, generate_inspection_number,
    generate_dispatch_number, packaging_available,
    compute_order_dispatch_summary, compute_alerts,
    count_unacknowledged_alerts, compute_operational_metrics,
    compute_order_summary,
    resolve_upload_path, inline_ok,
)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['UPLOAD_FOLDER'] = 'uploads'
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
    return client.post('/auth/login', data={'username': username,
                                            'password': password},
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
    prod = Product(internal_code='PRD-0001', product_name='Hex Bolt M10')
    db.session.add_all([cust, mat, vend, st, sup, prod])
    db.session.commit()


def _make_order(qty=100, due_days=10, status='confirmed'):
    order = Order(
        order_number=generate_order_number(),
        customer_id=Customer.query.first().id,
        product_id=Product.query.first().id,
        part_description='Hex Bolt M10', drawing_number='DWG-1', revision='A',
        material_grade='SS304', quantity_ordered=qty, unit='pcs', rate=12.5,
        order_date=date.today(), due_date=date.today() + timedelta(days=due_days),
        status=status, priority='normal')
    db.session.add(order)
    db.session.commit()
    return order


def _make_packaging(order, accepted=100, status='warehoused'):
    rec = PackagingRecord(order_id=order.id, packaging_date=date.today(),
                          packaging_type='carton', quantity_packed=accepted,
                          quantity_accepted=accepted, status=status,
                          label_ref='LBL-1')
    db.session.add(rec)
    db.session.commit()
    return rec


def _make_doc(entity_type='product', entity_id=None, title='Drawing A',
              category='drawing', revision='A', is_current=True,
              file_path='uploads/test_doc.pdf'):
    doc = Attachment(
        entity_type=entity_type,
        entity_id=entity_id if entity_id is not None else Product.query.first().id,
        file_name='stored.pdf', original_name='drawing.pdf', file_type='pdf',
        file_size=1024, file_path=file_path, title=title, category=category,
        revision=revision, is_current=is_current, uploaded_by=1)
    db.session.add(doc)
    db.session.commit()
    return doc


# ============================================================
# Documents
# ============================================================

def test_document_register_renders(app, client):
    login(client)
    _make_doc(title='Bracket drawing')
    resp = client.get('/documents/')
    assert resp.status_code == 200
    assert b'Document Register' in resp.data
    assert b'Bracket drawing' in resp.data


def test_documents_require_login(client):
    resp = client.get('/documents/', follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_document_links_to_product(app):
    doc = _make_doc(entity_type='product', category='material_certificate',
                    title='Mill cert')
    from app.documents.routes import resolve_entity
    from flask import url_for
    # resolve_entity builds a URL, so it needs a request context.
    with app.test_request_context():
        record, label, url = resolve_entity(doc.entity_type, doc.entity_id)
    assert record is not None
    assert label == 'PRD-0001'
    assert url and '/products/' in url

    lot = MaterialLot(lot_number=generate_lot_number(),
                      material_id=Material.query.first().id, quantity_received=10,
                      unit='kg', receipt_date=date.today())
    db.session.add(lot); db.session.commit()
    with app.test_request_context():
        rec, lbl, u = resolve_entity('lot', lot.id)
    assert u and f'/inventory/lots/{lot.id}' in u


def test_document_upload_via_route(app, client):
    login(client)
    pid = Product.query.first().id
    data = {
        'entity_type': 'product', 'entity_id': str(pid),
        'title': 'Spec sheet', 'category': 'specification',
        'revision': 'B', 'is_current': 'y',
        'file': (io.BytesIO(b'%PDF-1.4 test'), 'spec.pdf'),
    }
    resp = client.post('/documents/new', data=data,
                       content_type='multipart/form-data',
                       follow_redirects=True)
    assert resp.status_code == 200
    doc = Attachment.query.filter_by(title='Spec sheet').first()
    assert doc is not None
    assert doc.category == 'specification'
    assert doc.revision == 'B'
    assert doc.entity_type == 'product'
    assert doc.entity_id == pid


def test_new_revision_supersedes_previous(app, client):
    login(client)
    pid = Product.query.first().id
    for rev in ('A', 'B'):
        client.post('/documents/new', data={
            'entity_type': 'product', 'entity_id': str(pid),
            'title': 'Bracket drawing', 'category': 'drawing',
            'revision': rev, 'is_current': 'y',
            'file': (io.BytesIO(b'x'), f'dwg_{rev}.pdf'),
        }, content_type='multipart/form-data', follow_redirects=True)
    docs = Attachment.query.filter_by(title='Bracket drawing').order_by(
        Attachment.id).all()
    assert len(docs) == 2
    # Revision history is preserved: the old one is kept but no longer current.
    assert docs[0].is_current is False
    assert docs[1].is_current is True
    assert docs[1].supersedes_id == docs[0].id


def test_document_download_serves_file(app, client):
    login(client)
    os.makedirs('uploads', exist_ok=True)
    with open('uploads/test_doc.pdf', 'wb') as fh:
        fh.write(b'%PDF-1.4 hello')
    doc = _make_doc(file_path='uploads/test_doc.pdf')
    resp = client.get(f'/documents/{doc.id}/download')
    assert resp.status_code == 200
    assert resp.data.startswith(b'%PDF')


def test_document_download_blocks_path_traversal(app, client):
    login(client)
    doc = _make_doc(file_path='uploads/../../etc/passwd')
    resp = client.get(f'/documents/{doc.id}/download', follow_redirects=True)
    # Refused, and the response must not leak the file contents.
    assert b'root:' not in resp.data
    assert resolve_upload_path('uploads/../../etc/passwd', 'uploads') is None


def test_resolve_upload_path_rejects_escape():
    assert resolve_upload_path('uploads/../../etc/hosts', 'uploads') is None
    assert resolve_upload_path('', 'uploads') is None
    assert resolve_upload_path('uploads/does_not_exist_xyz.pdf', 'uploads') is None


def test_inline_ok_refuses_unsafe_types():
    assert inline_ok('pdf') is True
    assert inline_ok('html') is False
    assert inline_ok('svg') is False
    assert inline_ok('step') is False


def test_document_supersede_route(app, client):
    login(client)
    doc = _make_doc()
    resp = client.post(f'/documents/{doc.id}/supersede', follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(doc)
    assert doc.is_current is False


def test_document_register_filters_by_category(app, client):
    login(client)
    _make_doc(title='A drawing', category='drawing')
    _make_doc(title='A cert', category='material_certificate')
    resp = client.get('/documents/?category=material_certificate')
    assert resp.status_code == 200
    assert b'A cert' in resp.data
    assert b'A drawing' not in resp.data


def test_documents_export_xlsx(app, client):
    login(client)
    _make_doc(title='Exportable')
    resp = client.get('/documents/export')
    assert resp.status_code == 200
    assert resp.data[:2] == b'PK'  # xlsx is a zip container


def test_documents_for_entity_page(app, client):
    login(client)
    pid = Product.query.first().id
    _make_doc(entity_type='product', entity_id=pid, title='Part doc')
    resp = client.get(f'/documents/for/product/{pid}')
    assert resp.status_code == 200
    assert b'Part doc' in resp.data


# ============================================================
# Dispatch
# ============================================================

def test_dispatch_number_generation_unique(app):
    a = generate_dispatch_number()
    db.session.add(Dispatch(dispatch_number=a, order_id=_make_order().id,
                            dispatch_date=date.today()))
    db.session.commit()
    b = generate_dispatch_number()
    assert a != b


def test_packaging_available_subtracts_shipped(app):
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)
    assert packaging_available(rec) == 100
    dispatch = Dispatch(dispatch_number=generate_dispatch_number(),
                        order_id=order.id, dispatch_date=date.today(),
                        status='dispatched')
    db.session.add(dispatch)
    db.session.flush()
    db.session.add(DispatchLine(dispatch_id=dispatch.id,
                                packaging_record_id=rec.id, quantity=40))
    db.session.commit()
    assert packaging_available(rec) == 60


def test_dispatch_route_creates_and_ships(app, client):
    login(client)
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)

    resp = client.post(f'/orders/{order.id}/dispatch/new', data={
        'dispatch_date': date.today().isoformat(), 'mode': 'surface',
        'status': 'dispatched', 'delivery_status': 'pending',
    }, follow_redirects=True)
    assert resp.status_code == 200
    dispatch = Dispatch.query.filter_by(order_id=order.id).first()
    assert dispatch is not None

    resp = client.post(f'/orders/dispatch/{dispatch.id}/line/add', data={
        'packaging_record_id': str(rec.id), 'quantity': '60',
        'quantity_rejected': '0',
    }, follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(dispatch)
    assert dispatch.quantity_shipped == 60
    # The order is only part-shipped, so it must NOT be complete.
    db.session.refresh(order)
    assert order.status != 'completed'


def test_dispatch_line_cannot_exceed_available(app, client):
    login(client)
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=50)
    dispatch = Dispatch(dispatch_number=generate_dispatch_number(),
                        order_id=order.id, dispatch_date=date.today(),
                        status='dispatched')
    db.session.add(dispatch)
    db.session.commit()

    resp = client.post(f'/orders/dispatch/{dispatch.id}/line/add', data={
        'packaging_record_id': str(rec.id), 'quantity': '999',
        'quantity_rejected': '0',
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert b'available' in resp.data.lower()
    db.session.refresh(dispatch)
    assert dispatch.quantity_shipped == 0


def test_delete_dispatch_line_releases_stock(app, client):
    login(client)
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)
    dispatch = Dispatch(dispatch_number=generate_dispatch_number(),
                        order_id=order.id, dispatch_date=date.today(),
                        status='dispatched')
    db.session.add(dispatch)
    db.session.flush()
    line = DispatchLine(dispatch_id=dispatch.id, packaging_record_id=rec.id,
                        quantity=100)
    db.session.add(line)
    db.session.flush()
    rec.status = 'dispatched'
    db.session.commit()

    resp = client.post(f'/orders/dispatch/line/{line.id}/delete',
                       follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(rec)
    assert packaging_available(rec) == 100
    assert rec.status == 'warehoused'


def test_dispatch_register_and_view_render(app, client):
    login(client)
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)
    dispatch = Dispatch(dispatch_number=generate_dispatch_number(),
                        order_id=order.id, customer_id=order.customer_id,
                        dispatch_date=date.today(), status='dispatched')
    db.session.add(dispatch)
    db.session.flush()
    db.session.add(DispatchLine(dispatch_id=dispatch.id,
                                packaging_record_id=rec.id, quantity=25))
    db.session.commit()

    assert client.get('/orders/dispatches').status_code == 200
    view = client.get(f'/orders/dispatch/{dispatch.id}')
    assert view.status_code == 200
    assert dispatch.dispatch_number.encode() in view.data
    assert client.get('/orders/dispatches/export').data[:2] == b'PK'


def test_order_dispatch_summary_partial_and_full(app):
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)
    d1 = Dispatch(dispatch_number=generate_dispatch_number(), order_id=order.id,
                  dispatch_date=date.today(), status='dispatched')
    db.session.add(d1)
    db.session.flush()
    db.session.add(DispatchLine(dispatch_id=d1.id,
                                packaging_record_id=rec.id, quantity=40))
    db.session.commit()
    s = compute_order_dispatch_summary(order)
    assert s['shipped'] == 40
    assert s['remaining'] == 60
    assert s['is_partial'] is True
    assert s['is_fully_shipped'] is False

    db.session.add(DispatchLine(dispatch_id=d1.id,
                                packaging_record_id=rec.id, quantity=60))
    db.session.commit()
    s = compute_order_dispatch_summary(order)
    assert s['shipped'] == 100
    assert s['is_fully_shipped'] is True


def test_order_summary_dispatched_reflects_shipped_not_warehoused(app):
    """The order summary's dispatched figure must be what actually left the
    warehouse, not what is merely sitting in finished-goods stock."""
    order = _make_order(qty=100)
    rec = _make_packaging(order, accepted=100)
    d = Dispatch(dispatch_number=generate_dispatch_number(), order_id=order.id,
                 dispatch_date=date.today(), status='dispatched')
    db.session.add(d)
    db.session.flush()
    db.session.add(DispatchLine(dispatch_id=d.id,
                                packaging_record_id=rec.id, quantity=40))
    db.session.commit()

    assert order.quantity_warehoused == 100
    assert order.quantity_shipped == 40
    summary = compute_order_summary(order)
    assert summary['quantity_dispatched'] == 40
    assert summary['quantity_warehoused'] == 100


def test_dispatch_requires_login(client):
    resp = client.get('/orders/dispatches', follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_read_only_user_cannot_create_dispatch(app, client):
    login(client, 'viewer', 'view123')
    order = _make_order(qty=100)
    resp = client.post(f'/orders/{order.id}/dispatch/new', data={
        'dispatch_date': date.today().isoformat(), 'mode': 'surface',
        'status': 'prepared', 'delivery_status': 'pending',
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert Dispatch.query.filter_by(order_id=order.id).first() is None


# ============================================================
# Alerts
# ============================================================

def test_overdue_order_raises_alert(app):
    _make_order(qty=100, due_days=-5)  # already past due
    alerts = compute_alerts()
    keys = [a['key'] for a in alerts]
    assert any(k.endswith(':overdue') for k in keys)


def test_alerts_are_not_duplicated(app):
    _make_order(qty=100, due_days=-5)
    alerts = compute_alerts()
    keys = [a['key'] for a in alerts]
    assert len(keys) == len(set(keys))


def test_pending_inspection_alert(app):
    order = _make_order()
    db.session.add(Inspection(inspection_number=generate_inspection_number(),
                              order_id=order.id, inspection_type='final',
                              inspection_date=date.today(), result='pending'))
    db.session.commit()
    alerts = compute_alerts()
    assert any(a['category'] == 'pending_inspection' for a in alerts)


def test_alert_disappears_when_condition_clears(app):
    order = _make_order(qty=100, due_days=-5)
    assert any(a['category'] == 'overdue_order' for a in compute_alerts())
    order.due_date = date.today() + timedelta(days=30)
    order.status = 'completed'
    db.session.commit()
    assert not any(a['category'] == 'overdue_order' for a in compute_alerts())


def test_alerts_page_and_acknowledge(app, client):
    login(client)
    order = _make_order(qty=100, due_days=-5)
    resp = client.get('/alerts/')
    assert resp.status_code == 200

    alerts = compute_alerts(user=User.query.filter_by(username='admin').first())
    key = next(a['key'] for a in alerts if a['category'] == 'overdue_order')
    resp = client.post('/alerts/ack', data={'alert_key': key},
                       follow_redirects=True)
    assert resp.status_code == 200
    admin = User.query.filter_by(username='admin').first()
    ack = AlertAcknowledgement.query.filter_by(alert_key=key,
                                               user_id=admin.id).first()
    assert ack is not None


def test_acknowledgement_is_per_user(app):
    order = _make_order(qty=100, due_days=-5)
    admin = User.query.filter_by(username='admin').first()
    viewer = User.query.filter_by(username='viewer').first()
    key = next(a['key'] for a in compute_alerts()
               if a['category'] == 'overdue_order')

    db.session.add(AlertAcknowledgement(alert_key=key, user_id=admin.id))
    db.session.commit()

    # Admin sees it acknowledged; the viewer still sees it open.
    assert count_unacknowledged_alerts(admin) < count_unacknowledged_alerts(viewer)


def test_alerts_export_xlsx(app, client):
    login(client)
    _make_order(qty=100, due_days=-5)
    resp = client.get('/alerts/export')
    assert resp.status_code == 200
    assert resp.data[:2] == b'PK'


def test_alerts_require_login(client):
    resp = client.get('/alerts/', follow_redirects=False)
    assert resp.status_code in (301, 302)


# ============================================================
# Dashboard
# ============================================================

def test_dashboard_renders_with_operations(app, client):
    login(client)
    order = _make_order(qty=100)
    _make_packaging(order, accepted=100)
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'Operations' in resp.data


def test_operational_metrics_reflect_records(app):
    order = _make_order(qty=100, due_days=-3)
    _make_packaging(order, accepted=100)
    m = compute_operational_metrics()
    assert m['total_orders'] == 1
    assert m['overdue_orders'] == 1
    assert m['finished_goods_warehoused'] == 100
    assert m['finished_goods_available'] == 100
    assert m['has_operational_data'] is True


def test_operational_metrics_hide_financials(app):
    admin = User.query.filter_by(username='admin').first()
    viewer = User.query.filter_by(username='viewer').first()
    m_admin = compute_operational_metrics(user=admin, include_financials=True)
    m_viewer = compute_operational_metrics(user=viewer, include_financials=False)
    assert m_admin['show_financials'] is True
    assert 'supplier_payable_outstanding' in m_admin
    assert m_viewer['show_financials'] is False
    assert 'supplier_payable_outstanding' not in m_viewer


def test_dashboard_financial_section_gated_in_rendered_page(app, client):
    """The rendered dashboard, not just the metric dict, hides payables from a
    read-only user and shows them to an admin."""
    login(client, 'viewer', 'view123')
    body = client.get('/').data.decode()
    assert 'Financial balances' not in body

    client.get('/auth/logout')
    login(client, 'admin', 'admin123')
    body = client.get('/').data.decode()
    assert 'Financial balances' in body


def test_operational_metrics_empty_db_is_zero_not_error(app):
    m = compute_operational_metrics()
    assert m['total_orders'] == 0
    assert m['has_operational_data'] is False
    assert m['dispatches_total'] == 0


# ============================================================
# Notifications
# ============================================================

def _make_notification(user_id, title='Something happened', read=False,
                       ref_type=None, ref_id=None):
    from app.models import Notification
    n = Notification(user_id=user_id, title=title, message='detail',
                     notification_type='test', is_read=read,
                     reference_type=ref_type, reference_id=ref_id)
    db.session.add(n)
    db.session.commit()
    return n


def test_notifications_page_renders(app, client):
    login(client)
    admin = User.query.filter_by(username='admin').first()
    _make_notification(admin.id, title='Quotation Q-1 created')
    resp = client.get('/notifications/')
    assert resp.status_code == 200
    assert b'Quotation Q-1 created' in resp.data


def test_notifications_require_login(client):
    resp = client.get('/notifications/', follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_mark_notification_read(app, client):
    login(client)
    admin = User.query.filter_by(username='admin').first()
    n = _make_notification(admin.id)
    resp = client.post(f'/notifications/{n.id}/read', follow_redirects=True)
    assert resp.status_code == 200
    db.session.refresh(n)
    assert n.is_read is True


def test_mark_all_read(app, client):
    login(client)
    admin = User.query.filter_by(username='admin').first()
    _make_notification(admin.id, title='a')
    _make_notification(admin.id, title='b')
    resp = client.post('/notifications/read-all', follow_redirects=True)
    assert resp.status_code == 200
    from app.models import Notification
    assert Notification.query.filter_by(user_id=admin.id, is_read=False).count() == 0


def test_notifications_are_private_per_user(app, client):
    login(client, 'viewer', 'view123')
    admin = User.query.filter_by(username='admin').first()
    n = _make_notification(admin.id, title='Admin only')
    # The viewer's centre must not show, read or delete the admin's notification.
    assert b'Admin only' not in client.get('/notifications/').data
    assert client.post(f'/notifications/{n.id}/read').status_code == 404
    assert client.post(f'/notifications/{n.id}/delete').status_code == 404


def test_delete_notification(app, client):
    login(client)
    from app.models import Notification
    admin = User.query.filter_by(username='admin').first()
    n = _make_notification(admin.id)
    nid = n.id
    resp = client.post(f'/notifications/{nid}/delete', follow_redirects=True)
    assert resp.status_code == 200
    assert Notification.query.get(nid) is None


def test_notify_roles_creates_one_per_matching_user(app):
    from app.utils import notify_roles
    notify_roles(['admin'], 'New order', 'detail', 'order_created',
                 reference_type='order', reference_id=1)
    db.session.commit()
    from app.models import Notification
    admin = User.query.filter_by(username='admin').first()
    viewer = User.query.filter_by(username='viewer').first()
    assert Notification.query.filter_by(user_id=admin.id).count() == 1
    assert Notification.query.filter_by(user_id=viewer.id).count() == 0


def test_order_creation_notifies_admin(app, client):
    login(client)
    from app.models import Notification
    client.post('/orders/new', data={
        'customer_id': str(Customer.query.first().id),
        'part_description': 'Hex Bolt', 'drawing_number': 'DWG-1',
        'quantity_ordered': '50', 'unit': 'pcs', 'rate': '10',
        'currency': 'INR', 'order_date': date.today().isoformat(),
        'status': 'confirmed', 'priority': 'normal',
    }, follow_redirects=True)
    admin = User.query.filter_by(username='admin').first()
    notes = Notification.query.filter_by(
        user_id=admin.id, notification_type='order_created').all()
    assert len(notes) == 1


def test_notification_link_resolves_reference(app):
    from app.notifications.routes import resolve_notification_link
    order = _make_order()
    n = _make_notification(User.query.first().id, ref_type='order',
                           ref_id=order.id)
    with app.test_request_context():
        link = resolve_notification_link(n)
    assert link and f'/orders/{order.id}' in link


def test_notification_link_none_for_unknown_reference(app):
    from app.notifications.routes import resolve_notification_link
    n = _make_notification(User.query.first().id, ref_type='nonsense',
                           ref_id=1)
    assert resolve_notification_link(n) is None
