"""Referential-integrity tests for master-data deletes.

Deleting a parent that is still referenced used to fail two ways:
  * a non-nullable FK (orders.customer_id) raised an unhandled IntegrityError,
    i.e. a 500 crash;
  * a nullable FK (supplier_materials.material_id) was silently nulled,
    corrupting history.
These tests pin the guarded behaviour: the delete is refused with a clear
message and the parent row survives.
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (User, Customer, Product, Material, Supplier,
                        SupplierMaterial, Order)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        db.session.add(User(
            username='admin', email='admin@test.com', full_name='Admin',
            role='admin',
            password_hash=bcrypt.generate_password_hash('admin123').decode()))
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client):
    client.post('/auth/login', data={'username': 'admin', 'password': 'admin123'},
                follow_redirects=True)


def _make_order(app, customer_id, product_id):
    with app.app_context():
        user = User.query.filter_by(username='admin').first()
        db.session.add(Order(
            order_number='SO-1', customer_id=customer_id, product_id=product_id,
            part_description='test', quantity_ordered=10, order_date=date.today(),
            due_date=date.today() + timedelta(days=5), status='confirmed',
            priority='normal', created_by=user.id))
        db.session.commit()


def test_customer_delete_blocked_by_order_no_crash(app, client):
    login(client)
    with app.app_context():
        c = Customer(customer_code='C1', company_name='Acme')
        p = Product(internal_code='P1', product_name='Bolt')
        db.session.add_all([c, p])
        db.session.commit()
        cid, pid = c.id, p.id
    _make_order(app, cid, pid)

    resp = client.post(f'/customers/{cid}/delete', follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200          # not a 500
    assert 'Cannot delete' in body
    with app.app_context():
        assert db.session.get(Customer, cid) is not None


def test_product_delete_blocked_by_order(app, client):
    login(client)
    with app.app_context():
        c = Customer(customer_code='C1', company_name='Acme')
        p = Product(internal_code='P1', product_name='Bolt')
        db.session.add_all([c, p])
        db.session.commit()
        cid, pid = c.id, p.id
    _make_order(app, cid, pid)

    resp = client.post(f'/products/{pid}/delete', follow_redirects=True)
    assert 'Cannot delete' in resp.get_data(as_text=True)
    with app.app_context():
        assert db.session.get(Product, pid) is not None


def test_material_delete_blocked_by_supplier_link(app, client):
    """Nullable FK: deleting used to silently null supplier_materials.material_id."""
    login(client)
    with app.app_context():
        m = Material(material_code='M1', material_name='SS304', density=8000)
        s = Supplier(supplier_code='S1', company_name='SteelCo')
        db.session.add_all([m, s])
        db.session.commit()
        db.session.add(SupplierMaterial(supplier_id=s.id, material_id=m.id))
        db.session.commit()
        mid = m.id

    resp = client.post(f'/materials/{mid}/delete', follow_redirects=True)
    assert 'Cannot delete' in resp.get_data(as_text=True)
    with app.app_context():
        assert db.session.get(Material, mid) is not None
        # The link must be intact, not silently nulled.
        link = SupplierMaterial.query.first()
        assert link.material_id == mid


def test_material_delete_succeeds_when_unreferenced(app, client):
    login(client)
    with app.app_context():
        m = Material(material_code='M2', material_name='Brass', density=8500)
        db.session.add(m)
        db.session.commit()
        mid = m.id

    client.post(f'/materials/{mid}/delete', follow_redirects=True)
    with app.app_context():
        assert db.session.get(Material, mid) is None
