"""Regression tests for form CSRF coverage and General Settings saving.

Two defects are pinned here:
  * the General Settings page handled POST in its view but the route was
    GET-only, so the form returned 405 and nothing ever saved;
  * three POST forms (quotation add-item, quotation item edit, admin user
    delete) rendered no CSRF token, so they returned 400 in dev/prod where
    CSRF is enabled.

The CSRF test renders every GET page with CSRF enabled and asserts each POST
form carries a token, which also catches forms built from includes.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (User, Customer, Product, Material, Currency,
                        SystemSetting)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = True   # match dev/prod behaviour
    with app.app_context():
        db.create_all()
        db.session.add(User(
            username='admin', email='admin@test.com', full_name='Admin',
            role='admin',
            password_hash=bcrypt.generate_password_hash('admin123').decode()))
        db.session.add(Customer(customer_code='C1', company_name='Acme'))
        db.session.add(Product(internal_code='P1', product_name='Bolt'))
        db.session.add(Material(material_code='M1', material_name='Steel',
                                density=7.85))
        db.session.add(Currency(currency_code='INR', currency_name='Rupee',
                                symbol='Rs', exchange_rate=1.0, is_base=True,
                                is_active=True))
        db.session.add(Currency(currency_code='USD', currency_name='Dollar',
                                symbol='$', exchange_rate=83.0, is_active=True))
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def _csrf(client):
    # Cached at login time; the session token stays valid for the session.
    return client._csrf_token


@pytest.fixture()
def client(app):
    client = app.test_client()
    html = client.get('/auth/login').get_data(as_text=True)
    client._csrf_token = re.search(
        r'name="csrf_token"[^>]*value="([^"]+)"', html).group(1)
    client.post('/auth/login', data={'username': 'admin',
                                     'password': 'admin123',
                                     'csrf_token': client._csrf_token},
                follow_redirects=True)
    return client


def test_general_settings_route_accepts_post(client, app):
    with app.app_context():
        methods = next(r.methods for r in app.url_map.iter_rules()
                       if r.endpoint == 'setting.general')
    assert 'POST' in methods


def test_general_settings_saves_submitted_values(client, app):
    with app.app_context():
        db.session.add(SystemSetting(setting_key='DEFAULT_CURRENCY',
                                     setting_value='INR'))
        db.session.commit()

    resp = client.post('/settings/general', data={
        'csrf_token': _csrf(client),
        'company_name': 'Nirmal Precision Pvt Ltd',
        'company_gst': '27ABCDE1234F1Z5',
        'default_currency': 'USD',
    })
    assert resp.status_code == 302

    with app.app_context():
        saved = {s.setting_key: s.setting_value
                 for s in SystemSetting.query.all()}
    assert saved['COMPANY_NAME'] == 'Nirmal Precision Pvt Ltd'
    assert saved['DEFAULT_CURRENCY'] == 'USD'
    # A key the form did not submit must not be created as a null setting.
    assert 'COMPANY_PHONE' not in saved


def test_general_settings_page_lists_currencies(client):
    html = client.get('/settings/general').get_data(as_text=True)
    assert 'name="default_currency"' in html
    assert 'USD' in html and 'INR' in html


def test_saved_company_name_shows_in_footer(client, app):
    """A name saved on the settings page must reach the rendered page.

    Previously the footer and the document header read COMPANY_NAME straight
    from app.config (the environment), so saving it had no visible effect.
    """
    with app.app_context():
        db.session.add(SystemSetting(setting_key='COMPANY_NAME',
                                     setting_value='Nirmal Precision Pvt Ltd'))
        db.session.commit()

    html = client.get('/dashboard').get_data(as_text=True)
    assert 'Nirmal Precision Pvt Ltd' in html


def test_costing_defaults_come_from_settings(client, app):
    with app.app_context():
        db.session.add(SystemSetting(setting_key='DEFAULT_OVERHEAD_PERCENT',
                                     setting_value='33'))
        db.session.commit()

    resp = client.post('/quotations/1/cost-item', data={
        'csrf_token': _csrf(client),
        'material_cost': 100, 'machining_cost': 0,
        'vendor_cost': 0, 'tooling_cost': 0})
    assert resp.status_code == 200
    # 33 from SystemSetting, not the config default of 15.
    assert resp.get_json()['overhead_percent'] == 33.0


def test_costing_defaults_fall_back_to_config(client, app):
    """An unset/invalid setting must not break the calculator."""
    with app.app_context():
        db.session.add(SystemSetting(setting_key='DEFAULT_OVERHEAD_PERCENT',
                                     setting_value='not-a-number'))
        db.session.commit()

    resp = client.post('/quotations/1/cost-item', data={
        'csrf_token': _csrf(client),
        'material_cost': 100, 'machining_cost': 0,
        'vendor_cost': 0, 'tooling_cost': 0})
    assert resp.status_code == 200
    assert resp.get_json()['overhead_percent'] == 15.0   # config default



def test_every_rendered_post_form_has_csrf_token(client, app):
    with app.app_context():
        params = {
            'id': [1, Customer.query.first().id, Product.query.first().id,
                   Material.query.first().id],
            'customer_id': [Customer.query.first().id],
            'product_id': [Product.query.first().id],
            'material_id': [Material.query.first().id],
        }

    missing, checked = [], 0
    for rule in app.url_map.iter_rules():
        if 'GET' not in rule.methods or rule.endpoint == 'static':
            continue
        if rule.endpoint.startswith('auth.logout'):
            continue
        url = str(rule)
        for arg in rule.arguments:
            if arg in params:
                url = url.replace(f'<int:{arg}>', str(params[arg][0]))
                url = url.replace(f'<{arg}>', str(params[arg][0]))
            else:
                url = None
                break
        if not url or '<' in url:
            continue
        resp = client.get(url)
        if resp.status_code >= 400 or 'text/html' not in resp.headers.get(
                'Content-Type', ''):
            continue
        checked += 1
        for form in re.finditer(r'<form\b[^>]*>(.*?)</form>',
                                resp.get_data(as_text=True), re.S | re.I):
            tag = form.group(0)
            if 'method="post"' not in tag.lower():
                continue
            if 'csrf_token' in tag:
                continue
            missing.append(url)

    assert checked >= 25, f'only checked {checked} pages'
    assert not missing, f'POST forms without CSRF token on: {sorted(set(missing))}'
