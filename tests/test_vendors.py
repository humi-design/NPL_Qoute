"""Vendors module tests.

These exercise the real Flask app, real models and real request handlers
against an in-memory SQLite database (the app's existing 'testing' config).
No mocks: every assertion runs the production code path.
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Vendor, VendorWorkOrder, VendorQualityIssue, Process
)
from app.utils import compute_vendor_summary, generate_work_order_number


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
    """Create a small but representative dataset."""
    user = User(
        username='admin', email='admin@test.com', full_name='Admin',
        role='admin', password_hash=bcrypt.generate_password_hash('admin123').decode()
    )
    db.session.add(user)
    db.session.flush()

    v1 = Vendor(vendor_code='VND-0001', vendor_name='Alpha Heat Treat',
                contact_person='Ravi', email='alpha@test.com', lead_time=7)
    v2 = Vendor(vendor_code='VND-0002', vendor_name='Beta Plating',
                contact_person='Suresh', email='beta@test.com', lead_time=3,
                is_active=False)
    db.session.add_all([v1, v2])
    db.session.flush()

    proc = Process(process_code='PRC-0008', process_name='Heat Treatment',
                   process_type='vendor', vendor_id=v1.id, cost_type='per_piece',
                   cost_value=2.5)
    db.session.add(proc)
    db.session.flush()

    today = date.today()
    # v1: one completed (past), one open overdue, one open future
    db.session.add_all([
        VendorWorkOrder(
            work_order_number='WO-2024-0001', vendor_id=v1.id, process_id=proc.id,
            title='Heat Treatment Batch', service_name='Heat Treatment',
            quantity_ordered=500, quantity_received=500, quantity_rejected=5,
            rate=2.5, status='completed',
            order_date=today - timedelta(days=20), due_date=today - timedelta(days=5),
            completed_date=today - timedelta(days=5), created_by=user.id),
        VendorWorkOrder(
            work_order_number='WO-2024-0002', vendor_id=v1.id,
            title='Case Hardening', service_name='Heat Treatment',
            quantity_ordered=300, quantity_received=100, rate=3.0, status='in_progress',
            order_date=today - timedelta(days=10), due_date=today - timedelta(days=2),
            created_by=user.id),
        VendorWorkOrder(
            work_order_number='WO-2024-0003', vendor_id=v1.id,
            title='Annealing', service_name='Annealing',
            quantity_ordered=200, quantity_received=0, rate=1.0, status='pending',
            order_date=today, due_date=today + timedelta(days=10), created_by=user.id),
        # v2: one cancelled (past)
        VendorWorkOrder(
            work_order_number='WO-2024-0004', vendor_id=v2.id,
            title='Zinc Plating', service_name='Plating',
            quantity_ordered=1000, quantity_received=0, rate=1.5, status='cancelled',
            order_date=today - timedelta(days=15), due_date=today - timedelta(days=8),
            created_by=user.id),
    ])
    db.session.flush()

    wo = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0001').first()
    db.session.add(VendorQualityIssue(
        vendor_id=v1.id, work_order_id=wo.id, issue_date=today - timedelta(days=4),
        issue_type='rejection', severity='minor', quantity_affected=5,
        description='Hardness below spec', status='resolved',
        resolved_date=today - timedelta(days=1), reported_by=user.id))
    db.session.commit()


def login(client):
    return client.post('/auth/login', data={'username': 'admin', 'password': 'admin123'},
                       follow_redirects=True)


# ============================================================
# Model / status-rule unit tests
# ============================================================

def test_status_rules(app):
    with app.app_context():
        open_wo = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0002').first()
        done_wo = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0001').first()
        cancelled_wo = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0004').first()

        assert open_wo.is_open and not open_wo.is_completed
        assert open_wo.pending_quantity == 200
        assert open_wo.is_overdue and open_wo.days_overdue == 2

        assert done_wo.is_completed and not done_wo.is_open
        assert done_wo.pending_quantity == 0
        assert not done_wo.is_overdue  # completed work is never overdue

        assert cancelled_wo.is_cancelled and not cancelled_wo.is_open
        assert not cancelled_wo.is_overdue


def test_vendor_summary(app):
    with app.app_context():
        v1 = Vendor.query.filter_by(vendor_code='VND-0001').first()
        s = compute_vendor_summary(v1)

        assert s['assigned_work'] == 3
        assert s['current_work'] == 2       # in_progress + pending
        assert s['past_work'] == 1          # completed
        assert s['pending_quantity'] == 200 + 200  # from the two open orders
        assert s['completed_quantity'] == 500
        assert s['rejected_quantity'] == 5
        assert s['overdue_work'] == 1
        assert s['quality_issues'] == 1
        assert s['open_quality_issues'] == 0  # the one issue is resolved
        assert s['vendor_processes'] == 1


def test_work_order_number_generator(app):
    with app.app_context():
        num = generate_work_order_number()
        assert num.startswith('WO-')
        assert num.endswith('0001')


# ============================================================
# Route tests
# ============================================================

def test_vendor_index_requires_login(client):
    resp = client.get('/vendors/')
    assert resp.status_code == 302
    assert '/auth/login' in resp.headers['Location']


def test_vendor_index_lists_and_paginates(client):
    login(client)
    resp = client.get('/vendors/?per_page=10')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Alpha Heat Treat' in body
    assert 'Beta Plating' in body


def test_vendor_index_search_and_filters(client):
    login(client)
    # search matches name
    body = client.get('/vendors/?search=Alpha').get_data(as_text=True)
    assert 'Alpha Heat Treat' in body and 'Beta Plating' not in body

    # status filter
    body = client.get('/vendors/?status=inactive').get_data(as_text=True)
    assert 'Beta Plating' in body and 'Alpha Heat Treat' not in body

    # work filter: vendor with current work
    body = client.get('/vendors/?work=current').get_data(as_text=True)
    assert 'Alpha Heat Treat' in body and 'Beta Plating' not in body

    # work filter: vendor with overdue work
    body = client.get('/vendors/?work=overdue').get_data(as_text=True)
    assert 'Alpha Heat Treat' in body

    # sorting by name desc
    resp = client.get('/vendors/?sort=name&dir=desc')
    assert resp.status_code == 200


def test_vendor_view_current_past_all(client):
    login(client)
    with client.application.app_context():
        v1 = Vendor.query.filter_by(vendor_code='VND-0001').first()
        v1_id = v1.id

    current = client.get(f'/vendors/{v1_id}?work=current').get_data(as_text=True)
    assert 'WO-2024-0002' in current and 'WO-2024-0003' in current
    assert 'WO-2024-0001' not in current  # completed excluded

    past = client.get(f'/vendors/{v1_id}?work=past').get_data(as_text=True)
    assert 'WO-2024-0001' in past
    assert 'WO-2024-0002' not in past

    all_view = client.get(f'/vendors/{v1_id}?work=all').get_data(as_text=True)
    assert 'WO-2024-0001' in all_view and 'WO-2024-0002' in all_view and 'WO-2024-0003' in all_view

    # summary tiles rendered
    assert 'Assigned Work' in all_view
    assert 'Overdue Work' in all_view


def test_work_order_register_views(client):
    login(client)
    body = client.get('/vendors/work-orders?work=current').get_data(as_text=True)
    assert 'WO-2024-0002' in body
    assert 'WO-2024-0001' not in body  # completed

    body = client.get('/vendors/work-orders?work=past').get_data(as_text=True)
    assert 'WO-2024-0001' in body
    assert 'WO-2024-0002' not in body

    body = client.get('/vendors/work-orders?work=all&overdue=1').get_data(as_text=True)
    assert 'WO-2024-0002' in body
    assert 'WO-2024-0003' not in body  # not overdue


def test_work_order_register_search(client):
    login(client)
    body = client.get('/vendors/work-orders?search=Case Hardening').get_data(as_text=True)
    assert 'WO-2024-0002' in body and 'WO-2024-0003' not in body

    body = client.get('/vendors/work-orders?search=Beta Plating').get_data(as_text=True)
    assert 'WO-2024-0004' in body


def test_create_and_edit_work_order(client):
    login(client)
    with client.application.app_context():
        v1_id = Vendor.query.filter_by(vendor_code='VND-0001').first().id

    resp = client.post('/vendors/work-orders/new', data={
        'work_order_number': 'WO-2024-9999',
        'vendor_id': str(v1_id),
        'process_id': '',
        'product_id': '',
        'title': 'Test Job',
        'service_name': 'Testing',
        'quantity_ordered': '50',
        'quantity_received': '0',
        'quantity_rejected': '0',
        'unit': 'pcs',
        'rate': '4.5',
        'currency': 'INR',
        'order_date': date.today().isoformat(),
        'due_date': (date.today() + timedelta(days=5)).isoformat(),
        'status': 'pending',
        'priority': 'normal',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        wo = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-9999').first()
        assert wo is not None
        assert wo.quantity_ordered == 50
        wo_id = wo.id

    # Edit it
    resp = client.post(f'/vendors/work-orders/{wo_id}/edit', data={
        'work_order_number': 'WO-2024-9999',
        'vendor_id': str(v1_id),
        'process_id': '',
        'product_id': '',
        'title': 'Test Job Updated',
        'service_name': 'Testing',
        'quantity_ordered': '75',
        'quantity_received': '0',
        'quantity_rejected': '0',
        'unit': 'pcs',
        'rate': '4.5',
        'currency': 'INR',
        'order_date': date.today().isoformat(),
        'due_date': (date.today() + timedelta(days=5)).isoformat(),
        'status': 'in_progress',
        'priority': 'high',
    }, follow_redirects=True)
    assert resp.status_code == 200
    with client.application.app_context():
        wo = VendorWorkOrder.query.get(wo_id)
        assert wo.quantity_ordered == 75
        assert wo.status == 'in_progress'


def test_receive_auto_completes(client):
    login(client)
    with client.application.app_context():
        wo_id = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0002').first().id
        # ordered 300, received 100 -> receiving 200 should complete it

    resp = client.post(f'/vendors/work-orders/{wo_id}/receive',
                       data={'quantity': '200', 'rejected': '3'}, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        wo = VendorWorkOrder.query.get(wo_id)
        assert wo.quantity_received == 300
        assert wo.quantity_rejected == 3
        assert wo.status == 'completed'
        assert wo.completed_date == date.today()
        assert wo.pending_quantity == 0


def test_receive_partial_sets_in_progress(client):
    login(client)
    with client.application.app_context():
        wo_id = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0003').first().id

    client.post(f'/vendors/work-orders/{wo_id}/receive',
                data={'quantity': '50', 'rejected': '0'}, follow_redirects=True)

    with client.application.app_context():
        wo = VendorWorkOrder.query.get(wo_id)
        assert wo.quantity_received == 50
        assert wo.status == 'in_progress'
        assert wo.pending_quantity == 150


def test_change_status(client):
    login(client)
    with client.application.app_context():
        wo_id = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0003').first().id

    client.post(f'/vendors/work-orders/{wo_id}/status/completed', follow_redirects=True)
    with client.application.app_context():
        wo = VendorWorkOrder.query.get(wo_id)
        assert wo.status == 'completed'
        assert wo.completed_date == date.today()

    # invalid status is rejected
    client.post(f'/vendors/work-orders/{wo_id}/status/bogus', follow_redirects=True)
    with client.application.app_context():
        assert VendorWorkOrder.query.get(wo_id).status == 'completed'


def test_delete_work_order_detaches_quality_issue(client):
    login(client)
    with client.application.app_context():
        wo_id = VendorWorkOrder.query.filter_by(work_order_number='WO-2024-0001').first().id
        issue = VendorQualityIssue.query.filter_by(work_order_id=wo_id).first()
        issue_id = issue.id

    client.post(f'/vendors/work-orders/{wo_id}/delete', follow_redirects=True)

    with client.application.app_context():
        assert VendorWorkOrder.query.get(wo_id) is None
        issue = VendorQualityIssue.query.get(issue_id)
        assert issue is not None
        assert issue.work_order_id is None


def test_quality_issue_crud(client):
    login(client)
    with client.application.app_context():
        v1_id = Vendor.query.filter_by(vendor_code='VND-0001').first().id

    resp = client.post(f'/vendors/{v1_id}/quality/new', data={
        'vendor_id': str(v1_id),
        'work_order_id': '',
        'issue_date': date.today().isoformat(),
        'issue_type': 'complaint',
        'severity': 'major',
        'description': 'Late delivery',
        'quantity_affected': '10',
        'status': 'open',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        issue = VendorQualityIssue.query.filter_by(description='Late delivery').first()
        assert issue is not None
        assert issue.severity == 'major'
        issue_id = issue.id

    # Edit -> resolved sets resolved_date automatically
    client.post(f'/vendors/quality/{issue_id}/edit', data={
        'vendor_id': str(v1_id),
        'work_order_id': '',
        'issue_date': date.today().isoformat(),
        'issue_type': 'complaint',
        'severity': 'major',
        'description': 'Late delivery',
        'quantity_affected': '10',
        'status': 'resolved',
        'resolution': 'Vendor apologised',
    }, follow_redirects=True)
    with client.application.app_context():
        issue = VendorQualityIssue.query.get(issue_id)
        assert issue.status == 'resolved'
        assert issue.resolved_date == date.today()

    client.post(f'/vendors/quality/{issue_id}/delete', follow_redirects=True)
    with client.application.app_context():
        assert VendorQualityIssue.query.get(issue_id) is None


def test_vendor_delete_blocked_with_work_orders(client):
    login(client)
    with client.application.app_context():
        v1_id = Vendor.query.filter_by(vendor_code='VND-0001').first().id

    client.post(f'/vendors/{v1_id}/delete', follow_redirects=True)
    with client.application.app_context():
        assert Vendor.query.get(v1_id) is not None


def test_exports(client):
    login(client)
    resp = client.get('/vendors/export')
    assert resp.status_code == 200
    assert 'spreadsheetml' in resp.headers['Content-Type']

    resp = client.get('/vendors/work-orders/export?work=current')
    assert resp.status_code == 200
    assert 'spreadsheetml' in resp.headers['Content-Type']


def test_api_endpoints(client):
    login(client)
    resp = client.get('/api/vendors?summary=1')
    assert resp.status_code == 200
    data = resp.get_json()
    assert any(v['code'] == 'VND-0001' and 'summary' in v for v in data)

    with client.application.app_context():
        v1_id = Vendor.query.filter_by(vendor_code='VND-0001').first().id

    resp = client.get(f'/api/vendors/{v1_id}')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['summary']['assigned_work'] == 3
    assert len(data['work_orders']) == 3

    resp = client.get('/api/vendors/work-orders?view=current')
    assert resp.status_code == 200
    data = resp.get_json()
    assert all(w['status'] in ('pending', 'in_progress') for w in data)
    assert len(data) == 2


def test_create_work_order_requires_valid_vendor(client):
    login(client)
    # empty vendor -> form validation fails, nothing created
    resp = client.post('/vendors/work-orders/new', data={
        'work_order_number': 'WO-BAD-1', 'vendor_id': '', 'process_id': '', 'product_id': '',
        'title': 'Bad', 'quantity_ordered': '1', 'unit': 'pcs', 'currency': 'INR',
        'status': 'pending', 'priority': 'normal',
    }, follow_redirects=True)
    assert resp.status_code == 200

    # non-existent vendor id -> rejected by the route guard
    resp = client.post('/vendors/work-orders/new', data={
        'work_order_number': 'WO-BAD-2', 'vendor_id': '999999', 'process_id': '', 'product_id': '',
        'title': 'Bad', 'quantity_ordered': '1', 'unit': 'pcs', 'currency': 'INR',
        'status': 'pending', 'priority': 'normal',
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert 'valid vendor' in resp.get_data(as_text=True).lower()

    with client.application.app_context():
        assert VendorWorkOrder.query.filter(
            VendorWorkOrder.work_order_number.in_(['WO-BAD-1', 'WO-BAD-2'])
        ).count() == 0


def test_vendor_api_search(client):
    login(client)
    resp = client.get('/vendors/api/search?q=Alpha')
    assert resp.status_code == 200
    data = resp.get_json()
    assert any(v['code'] == 'VND-0001' for v in data)
    # too short -> empty
    assert client.get('/vendors/api/search?q=A').get_json() == []
