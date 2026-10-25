"""Security regression tests for login redirects and user deletion.

Two defects are pinned here:
  * the post-login ``next`` parameter was passed straight to ``redirect``,
    making ``/auth/login?next=https://evil.example.com`` an open redirect;
  * deleting a user who had authored records left dangling creator ids (and
    would raise an IntegrityError once the DB enforces the foreign keys).
"""
import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import (User, Customer, Product, Order, Notification,
                        AlertAcknowledgement)


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        admin = User(username='admin', email='admin@test.com', full_name='Admin',
                     role='admin',
                     password_hash=bcrypt.generate_password_hash('admin123').decode())
        db.session.add(admin)
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client):
    return client.post('/auth/login', data={'username': 'admin',
                                            'password': 'admin123'},
                       follow_redirects=True)


@pytest.mark.parametrize('evil', [
    'https://evil.example.com/phish',
    'http://evil.example.com',
    '//evil.example.com',
    '/\\evil.example.com',
    'javascript:alert(1)',
])
def test_login_next_rejects_external_targets(client, evil):
    resp = client.post(f'/auth/login?next={evil}',
                       data={'username': 'admin', 'password': 'admin123'},
                       follow_redirects=False)
    location = resp.headers.get('Location', '')
    assert resp.status_code == 302
    assert location == '/dashboard'          # fell back, did not honour `next`
    assert 'evil.example.com' not in location


def test_login_next_allows_local_path(client):
    resp = client.post('/auth/login?next=/orders/?view=current',
                       data={'username': 'admin', 'password': 'admin123'},
                       follow_redirects=False)
    assert resp.headers.get('Location') == '/orders/?view=current'


def test_delete_user_with_authored_records_is_blocked(app, client):
    login(client)
    with app.app_context():
        author = User(username='author', email='author@test.com',
                      full_name='Author', role='sales',
                      password_hash=bcrypt.generate_password_hash('auth1234').decode())
        db.session.add(author)
        db.session.commit()
        c = Customer(customer_code='C1', company_name='Acme')
        p = Product(internal_code='P1', product_name='Bolt')
        db.session.add_all([c, p])
        db.session.commit()
        db.session.add(Order(
            order_number='SO-1', customer_id=c.id, product_id=p.id,
            part_description='x', quantity_ordered=1, order_date=date.today(),
            due_date=date.today() + timedelta(days=1), status='draft',
            priority='normal', created_by=author.id))
        db.session.commit()
        aid = author.id

    resp = client.post(f'/auth/users/{aid}/delete', follow_redirects=True)
    assert 'Cannot delete' in resp.get_data(as_text=True)
    with app.app_context():
        assert db.session.get(User, aid) is not None
        assert Order.query.first().created_by == aid


def test_delete_unused_user_cleans_owned_rows(app, client):
    login(client)
    with app.app_context():
        u = User(username='unused', email='unused@test.com', full_name='Unused',
                 role='read_only',
                 password_hash=bcrypt.generate_password_hash('unused12').decode())
        db.session.add(u)
        db.session.commit()
        db.session.add(Notification(user_id=u.id, title='hi'))
        db.session.add(AlertAcknowledgement(alert_key='k1', user_id=u.id))
        db.session.commit()
        uid = u.id

    client.post(f'/auth/users/{uid}/delete', follow_redirects=True)
    with app.app_context():
        assert db.session.get(User, uid) is None
        assert Notification.query.filter_by(user_id=uid).count() == 0
        assert AlertAcknowledgement.query.filter_by(user_id=uid).count() == 0
