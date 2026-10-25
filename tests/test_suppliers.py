"""Supplier / Master Data module tests.

Exercise the real Flask app, real models and real request handlers against
an in-memory SQLite database (the app's existing 'testing' config).
No mocks: every assertion runs the production code path.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Material, Supplier, SupplierType, SupplierMaterial, Vendor, Customer
)
from app.utils import (compute_supplier_summary, generate_supplier_code,
                       generate_supplier_type_code)


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
    admin = User(
        username='admin', email='admin@test.com', full_name='Admin',
        role='admin', password_hash=bcrypt.generate_password_hash('admin123').decode()
    )
    sales = User(
        username='sales', email='sales@test.com', full_name='Sales',
        role='sales', password_hash=bcrypt.generate_password_hash('sales123').decode()
    )
    db.session.add_all([admin, sales])
    db.session.flush()

    raw = SupplierType(type_code='SUPT-001', type_name='Raw Material', sort_order=10)
    cons = SupplierType(type_code='SUPT-002', type_name='Consumables', sort_order=20)
    db.session.add_all([raw, cons])
    db.session.flush()

    m1 = Material(material_code='MAT-0001', material_name='SS304', density=8000,
                  grade='AISI 304', default_rate=200, rate_unit='₹/kg')
    m2 = Material(material_code='MAT-0002', material_name='EN8', density=7850,
                  grade='EN8', default_rate=85, rate_unit='₹/kg')
    db.session.add_all([m1, m2])
    db.session.flush()

    s1 = Supplier(supplier_code='SUP-0001', company_name='Bharat Steel',
                  supplier_type_id=raw.id, contact_person='Rakesh',
                  email='bharat@test.com', city='Mumbai', state='Maharashtra',
                  gst_vat='27AABCB1234C1Z5', lead_time=7, payment_terms='Net 30',
                  is_active=True)
    s2 = Supplier(supplier_code='SUP-0002', company_name='ToolMart',
                  supplier_type_id=cons.id, contact_person='Anil',
                  email='toolmart@test.com', city='Pune', state='Maharashtra',
                  lead_time=4, is_active=False)
    db.session.add_all([s1, s2])
    db.session.flush()

    db.session.add_all([
        SupplierMaterial(supplier_id=s1.id, material_id=m1.id, grade='AISI 304',
                         rate=210.0, rate_unit='₹/kg', lead_time=7, is_preferred=True),
        SupplierMaterial(supplier_id=s1.id, material_id=m2.id, grade='EN8',
                         rate=85.0, rate_unit='₹/kg', lead_time=7, is_preferred=False),
    ])
    db.session.commit()


def login(client, username='admin', password='admin123'):
    return client.post('/auth/login', data={'username': username, 'password': password},
                       follow_redirects=True)


# ============================================================
# Model / helper unit tests
# ============================================================

def test_supplier_summary(app):
    with app.app_context():
        s1 = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        summary = compute_supplier_summary(s1)
        assert summary['supplied_materials'] == 2
        assert summary['preferred_materials'] == 1
        assert summary['material_grades'] == 2


def test_supplier_code_generators(app):
    with app.app_context():
        assert generate_supplier_code() == 'SUP-0003'
        assert generate_supplier_type_code() == 'SUPT-003'


def test_supplier_full_address(app):
    with app.app_context():
        s1 = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        s1.address_line1 = 'Plot 5'
        s1.address_line2 = None
        s1.city = 'Mumbai'
        s1.state = 'Maharashtra'
        s1.postal_code = '400001'
        s1.country = 'India'
        assert s1.full_address == 'Plot 5, Mumbai, Maharashtra, 400001, India'


# ============================================================
# Auth / access tests
# ============================================================

def test_supplier_index_requires_login(client):
    resp = client.get('/suppliers/')
    assert resp.status_code == 302
    assert '/auth/login' in resp.headers['Location']


def test_supplier_types_require_admin(client):
    login(client, 'sales', 'sales123')
    resp = client.get('/suppliers/types', follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert 'permission' in body.lower()


def test_supplier_index_lists(client):
    login(client)
    body = client.get('/suppliers/').get_data(as_text=True)
    assert 'Bharat Steel' in body
    assert 'ToolMart' in body


def test_supplier_index_search_and_filters(client):
    login(client)
    body = client.get('/suppliers/?search=Bharat').get_data(as_text=True)
    assert 'Bharat Steel' in body and 'ToolMart' not in body

    body = client.get('/suppliers/?status=inactive').get_data(as_text=True)
    assert 'ToolMart' in body and 'Bharat Steel' not in body


def test_supplier_index_type_filter_and_sort(client):
    login(client)
    with client.application.app_context():
        raw_id = SupplierType.query.filter_by(type_code='SUPT-001').first().id

    body = client.get(f'/suppliers/?type={raw_id}').get_data(as_text=True)
    assert 'Bharat Steel' in body and 'ToolMart' not in body

    # sort by name descending should not error
    resp = client.get('/suppliers/?sort=name&dir=desc')
    assert resp.status_code == 200


# ============================================================
# CRUD tests
# ============================================================

def test_create_supplier(client):
    login(client)
    resp = client.post('/suppliers/new', data={
        'supplier_code': 'SUP-0099',
        'company_name': 'New Metals Co',
        'supplier_type_id': '',
        'contact_person': 'New Contact',
        'email': 'new@metals.com',
        'city': 'Nashik',
        'country': 'India',
        'default_currency': 'INR',
        'payment_terms': 'Net 15',
        'lead_time': '5',
        'is_active': 'y',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        created = Supplier.query.filter_by(supplier_code='SUP-0099').first()
        assert created is not None
        assert created.company_name == 'New Metals Co'
        assert created.lead_time == 5
        assert created.is_active


def test_edit_supplier_tracks_revision(client):
    login(client)
    with client.application.app_context():
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    resp = client.post(f'/suppliers/{s1_id}/edit', data={
        'supplier_code': 'SUP-0001',
        'company_name': 'Bharat Steel Traders',
        'supplier_type_id': '',
        'contact_person': 'Rakesh Sharma',
        'email': 'bharat@test.com',
        'city': 'Mumbai',
        'country': 'India',
        'default_currency': 'INR',
        'payment_terms': 'Net 45',
        'lead_time': '9',
        'is_active': 'y',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        updated = Supplier.query.get(s1_id)
        assert updated.company_name == 'Bharat Steel Traders'
        assert updated.payment_terms == 'Net 45'
        assert updated.lead_time == 9

        from app.models import RevisionHistory
        revs = RevisionHistory.query.filter_by(entity_type='supplier',
                                               entity_id=s1_id).all()
        fields = {r.field_name for r in revs}
        assert 'company_name' in fields
        assert 'payment_terms' in fields


def test_deactivate_and_activate_supplier(client):
    login(client)
    with client.application.app_context():
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    client.post(f'/suppliers/{s1_id}/deactivate', follow_redirects=True)
    with client.application.app_context():
        assert Supplier.query.get(s1_id).is_active is False

    client.post(f'/suppliers/{s1_id}/activate', follow_redirects=True)
    with client.application.app_context():
        assert Supplier.query.get(s1_id).is_active is True


def test_delete_supplier_blocked_when_materials_linked(client):
    login(client)
    with client.application.app_context():
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    resp = client.post(f'/suppliers/{s1_id}/delete', follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert 'Cannot delete' in body

    with client.application.app_context():
        assert Supplier.query.get(s1_id) is not None


def test_delete_supplier_without_links_succeeds(client):
    login(client)
    with client.application.app_context():
        s2_id = Supplier.query.filter_by(supplier_code='SUP-0002').first().id

    client.post(f'/suppliers/{s2_id}/delete', follow_redirects=True)
    with client.application.app_context():
        assert Supplier.query.get(s2_id) is None


def test_delete_supplier_blocked_when_referenced_by_material(client):
    login(client)
    with client.application.app_context():
        s2 = Supplier.query.filter_by(supplier_code='SUP-0002').first()
        m1 = Material.query.filter_by(material_code='MAT-0001').first()
        m1.supplier_id = s2.id
        db.session.commit()
        s2_id = s2.id

    resp = client.post(f'/suppliers/{s2_id}/delete', follow_redirects=True)
    assert 'Cannot delete' in resp.get_data(as_text=True)
    with client.application.app_context():
        assert Supplier.query.get(s2_id) is not None


# ============================================================
# Supplied materials
# ============================================================

def test_add_and_remove_supplied_material(client):
    login(client)
    with client.application.app_context():
        s2_id = Supplier.query.filter_by(supplier_code='SUP-0002').first().id
        m1_id = Material.query.filter_by(material_code='MAT-0001').first().id

    client.post(f'/suppliers/{s2_id}/material', data={
        'material_id': str(m1_id),
        'grade': 'AISI 304',
        'rate': '215',
        'rate_unit': '₹/kg',
        'lead_time': '6',
        'is_preferred': 'y',
    }, follow_redirects=True)

    with client.application.app_context():
        link = SupplierMaterial.query.filter_by(supplier_id=s2_id,
                                                material_id=m1_id).first()
        assert link is not None
        assert link.is_preferred
        link_id = link.id

    client.post(f'/suppliers/{s2_id}/material/{link_id}/delete', follow_redirects=True)
    with client.application.app_context():
        assert SupplierMaterial.query.get(link_id) is None


def test_duplicate_supplied_material_rejected(client):
    login(client)
    with client.application.app_context():
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        m1_id = Material.query.filter_by(material_code='MAT-0001').first().id

    resp = client.post(f'/suppliers/{s1_id}/material', data={
        'material_id': str(m1_id), 'rate_unit': '₹/kg',
    }, follow_redirects=True)
    assert 'already linked' in resp.get_data(as_text=True)

    with client.application.app_context():
        assert SupplierMaterial.query.filter_by(supplier_id=s1_id,
                                                material_id=m1_id).count() == 1


# ============================================================
# Supplier types
# ============================================================

def test_create_and_edit_supplier_type(client):
    login(client)
    client.post('/suppliers/types/new', data={
        'type_code': 'SUPT-050', 'type_name': 'Packaging',
        'description': 'Packaging materials', 'sort_order': '30', 'is_active': 'y',
    }, follow_redirects=True)

    with client.application.app_context():
        st = SupplierType.query.filter_by(type_code='SUPT-050').first()
        assert st is not None and st.type_name == 'Packaging'
        st_id = st.id

    client.post(f'/suppliers/types/{st_id}/edit', data={
        'type_code': 'SUPT-050', 'type_name': 'Packaging & Labels',
        'description': 'Packaging materials', 'sort_order': '35', 'is_active': 'y',
    }, follow_redirects=True)

    with client.application.app_context():
        assert SupplierType.query.get(st_id).type_name == 'Packaging & Labels'


def test_delete_supplier_type_blocked_when_in_use(client):
    login(client)
    with client.application.app_context():
        raw_id = SupplierType.query.filter_by(type_code='SUPT-001').first().id

    resp = client.post(f'/suppliers/types/{raw_id}/delete', follow_redirects=True)
    assert 'Cannot delete' in resp.get_data(as_text=True)
    with client.application.app_context():
        assert SupplierType.query.get(raw_id) is not None


def test_delete_unused_supplier_type_succeeds(client):
    login(client)
    with client.application.app_context():
        st = SupplierType(type_code='SUPT-099', type_name='Temporary')
        db.session.add(st)
        db.session.commit()
        st_id = st.id

    client.post(f'/suppliers/types/{st_id}/delete', follow_redirects=True)
    with client.application.app_context():
        assert SupplierType.query.get(st_id) is None


def test_inactive_type_kept_in_edit_form(client):
    """A supplier linked to a now-inactive type must still be editable."""
    login(client)
    with client.application.app_context():
        s1 = Supplier.query.filter_by(supplier_code='SUP-0001').first()
        s1.supplier_type.is_active = False
        db.session.commit()
        s1_id = s1.id

    resp = client.get(f'/suppliers/{s1_id}/edit')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Raw Material' in body  # linked type still offered


# ============================================================
# Export + API
# ============================================================

def test_supplier_export(client):
    login(client)
    resp = client.get('/suppliers/export')
    assert resp.status_code == 200
    assert 'spreadsheetml' in resp.headers['Content-Type']


def test_supplier_api_list_and_detail(client):
    login(client)
    resp = client.get('/api/suppliers?summary=1')
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1  # only the active supplier by default
    assert data[0]['code'] == 'SUP-0001'
    assert data[0]['summary']['supplied_materials'] == 2

    with client.application.app_context():
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    resp = client.get(f'/api/suppliers/{s1_id}')
    detail = resp.get_json()
    assert detail['name'] == 'Bharat Steel'
    assert len(detail['supplied_materials']) == 2
    assert detail['type'] == 'Raw Material'


def test_supplier_types_api(client):
    login(client)
    data = client.get('/api/supplier-types').get_json()
    codes = {t['code'] for t in data}
    assert 'SUPT-001' in codes
    raw = next(t for t in data if t['code'] == 'SUPT-001')
    assert raw['supplier_count'] == 1


def test_supplier_api_search(client):
    login(client)
    data = client.get('/suppliers/api/search?q=Bharat').get_json()
    assert len(data) == 1
    assert data[0]['code'] == 'SUP-0001'

    # too short -> empty
    assert client.get('/suppliers/api/search?q=B').get_json() == []


# ============================================================
# Navigation / no-duplication checks
# ============================================================

def test_master_data_nav_present_once(client):
    login(client)
    body = client.get('/dashboard').get_data(as_text=True)
    assert 'Master Data' in body
    # each master-data entity appears exactly once in the sidebar
    assert body.count('>Suppliers<') == 1
    assert body.count('>Supplier Types<') == 1
    assert body.count('>Vendors<') == 1
    assert body.count('>Customers<') == 1


def test_existing_customer_and_vendor_pages_still_work(client):
    login(client)
    assert client.get('/customers/').status_code == 200
    assert client.get('/vendors/').status_code == 200
    assert client.get('/materials/').status_code == 200


# ============================================================
# Material <-> Supplier master link (additive upgrade)
# ============================================================

def test_material_form_offers_master_suppliers(client):
    login(client)
    body = client.get('/materials/new').get_data(as_text=True)
    assert 'Supplier (master)' in body
    assert 'Bharat Steel' in body


def test_material_edit_links_master_supplier(client):
    login(client)
    with client.application.app_context():
        m1_id = Material.query.filter_by(material_code='MAT-0001').first().id
        s1_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id

    resp = client.post(f'/materials/{m1_id}/edit', data={
        'material_code': 'MAT-0001',
        'material_name': 'SS304',
        'density': '8000',
        'grade': 'AISI 304',
        'hsn_code': '7228',
        'default_supplier': '',
        'supplier_id': str(s1_id),
        'default_rate': '200',
        'rate_unit': '₹/kg',
        'is_active': 'y',
    }, follow_redirects=True)
    assert resp.status_code == 200

    with client.application.app_context():
        assert Material.query.get(m1_id).supplier_id == s1_id


def test_material_edit_can_clear_master_supplier(client):
    login(client)
    with client.application.app_context():
        m1 = Material.query.filter_by(material_code='MAT-0001').first()
        m1.supplier_id = Supplier.query.filter_by(supplier_code='SUP-0001').first().id
        db.session.commit()
        m1_id = m1.id

    client.post(f'/materials/{m1_id}/edit', data={
        'material_code': 'MAT-0001', 'material_name': 'SS304', 'density': '8000',
        'grade': 'AISI 304', 'supplier_id': '0', 'default_rate': '200',
        'rate_unit': '₹/kg', 'is_active': 'y',
    }, follow_redirects=True)

    with client.application.app_context():
        assert Material.query.get(m1_id).supplier_id is None


def test_material_edit_keeps_inactive_supplier_selected(client):
    """An inactive linked supplier must remain in the dropdown while editing."""
    login(client)
    with client.application.app_context():
        s2 = Supplier.query.filter_by(supplier_code='SUP-0002').first()
        s2.is_active = False
        m1 = Material.query.filter_by(material_code='MAT-0001').first()
        m1.supplier_id = s2.id
        db.session.commit()
        m1_id = m1.id

    body = client.get(f'/materials/{m1_id}/edit').get_data(as_text=True)
    assert 'ToolMart' in body
    assert 'selected' in body
