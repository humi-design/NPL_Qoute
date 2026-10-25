"""Error-page tests.

Before this, the app registered no error handlers, so a 404 or 500 rendered
Flask's bare default page and a server error was never logged. These tests pin
the branded pages and prove the 500 page never leaks a traceback.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')

from app.extensions import create_app, db, bcrypt
from app.models import User


@pytest.fixture()
def app():
    app = create_app('testing')
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = False

    @app.route('/__boom__')
    def _boom():
        raise RuntimeError('synthetic failure for the error-handler test')

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
    return client.post('/auth/login', data={'username': 'admin',
                                            'password': 'admin123'},
                       follow_redirects=True)


def test_404_renders_branded_page(client):
    login(client)
    resp = client.get('/no-such-page-here')
    body = resp.get_data(as_text=True)
    assert resp.status_code == 404
    assert 'Page not found' in body
    assert '<title>404 Not Found</title>' not in body  # not Flask's default
    assert 'Back to Dashboard' in body


def test_405_renders_branded_page(client):
    login(client)
    resp = client.put('/auth/login')
    assert resp.status_code == 405
    assert 'Method not allowed' in resp.get_data(as_text=True)


def test_500_renders_branded_page_and_hides_traceback(client):
    login(client)
    resp = client.get('/__boom__')
    body = resp.get_data(as_text=True)
    assert resp.status_code == 500
    assert 'Something went wrong' in body
    # The exception detail must stay server-side.
    assert 'RuntimeError' not in body
    assert 'synthetic failure' not in body
    assert 'Traceback' not in body
