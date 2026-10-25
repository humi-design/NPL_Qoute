"""Write-authorization guard tests.

The newer modules (orders, procurement, inventory, documents, costing) each
enforce their own write check. The older master-data modules (customer, vendor,
material, product, process, machine, quotation, rfq, setting) did not, which let
a read-only account create, edit and delete records. A central before_request
guard in the app factory now closes that gap.

These tests prove a read-only user cannot mutate records, while still being able
to use personal state (own notifications, own profile) and pure calculators.
"""
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (User, Customer, Material, Product, Vendor, Supplier,
                        SupplierType, Notification)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        _seed()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, username, password):
    return client.post('/auth/login', data={'username': username,
                                            'password': password},
                       follow_redirects=True)


def _seed():
    admin = User(username='admin', email='admin@test.com', full_name='Admin',
                 role='admin',
                 password_hash=bcrypt.generate_password_hash('admin123').decode())
    viewer = User(username='readonly', email='ro@test.com', full_name='RO',
                  role='read_only',
                  password_hash=bcrypt.generate_password_hash('read123').decode())
    sales = User(username='sales', email='sales@test.com', full_name='Sales',
                 role='sales',
                 password_hash=bcrypt.generate_password_hash('sales123').decode())
    db.session.add_all([admin, viewer, sales])
    db.session.add_all([
        Customer(customer_code='CUST-0001', company_name='Acme'),
        Material(material_code='MAT-0001', material_name='SS304', density=8000),
        Vendor(vendor_code='VND-0001', vendor_name='Alpha'),
        Product(internal_code='PRD-0001', product_name='Hex Bolt M10'),
        SupplierType(type_code='SUPT-001', type_name='Raw Material'),
        Supplier(supplier_code='SUP-0001', company_name='Bharat Steel'),
    ])
    db.session.commit()


def _counts():
    return {
        'Customer': Customer.query.count(),
        'Material': Material.query.count(),
        'Vendor': Vendor.query.count(),
        'Product': Product.query.count(),
    }


def test_read_only_cannot_delete_master_data(app, client):
    login(client, 'readonly', 'read123')
    with app.app_context():
        ids = {'cust': Customer.query.first().id,
               'mat': Material.query.first().id,
               'vend': Vendor.query.first().id,
               'prod': Product.query.first().id}
        before = _counts()
    for url in (f"/customers/{ids['cust']}/delete",
                f"/materials/{ids['mat']}/delete",
                f"/vendors/{ids['vend']}/delete",
                f"/products/{ids['prod']}/delete"):
        client.post(url, data={})
    with app.app_context():
        assert _counts() == before, 'read-only user deleted master data'


def test_read_only_cannot_create_master_data(app, client):
    login(client, 'readonly', 'read123')
    with app.app_context():
        before = _counts()
    client.post('/customers/new', data={'customer_code': 'CUST-NEW',
                                        'company_name': 'New Co',
                                        'currency': 'INR'})
    client.post('/materials/new', data={'material_code': 'MAT-NEW',
                                        'material_name': 'EN8'})
    with app.app_context():
        assert _counts() == before, 'read-only user created master data'


def test_sales_user_keeps_write_access(app, client):
    """The guard must not over-block: a role with the write permission still
    creates records."""
    login(client, 'sales', 'sales123')
    with app.app_context():
        before = Customer.query.count()
    client.post('/customers/new', data={'customer_code': 'CUST-S1',
                                        'company_name': 'Sales Co',
                                        'currency': 'INR'})
    with app.app_context():
        assert Customer.query.count() == before + 1


def test_read_only_can_still_mark_own_notification_read(app, client):
    login(client, 'readonly', 'read123')
    with app.app_context():
        u = User.query.filter_by(username='readonly').first()
        n = Notification(user_id=u.id, title='Hi', message='x')
        db.session.add(n)
        db.session.commit()
        nid = n.id
    client.post(f'/notifications/{nid}/read')
    with app.app_context():
        assert db.session.get(Notification, nid).is_read is True


def test_read_only_can_still_use_cost_calculator(client):
    login(client, 'readonly', 'read123')
    resp = client.post('/quotations/1/cost-item',
                       data={'material_cost': 100, 'machining_cost': 50,
                             'vendor_cost': 0, 'tooling_cost': 0})
    # Not blocked by the write guard (would be a 302 redirect, not JSON).
    assert resp.status_code == 200
    assert resp.is_json
