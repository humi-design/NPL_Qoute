"""Guards for the shipped Apache configuration.

The 404 outage happened because `.htaccess` carried a mod_wsgi rewrite and no
Passenger block. cPanel runs Python apps through Passenger, so that file made
Apache serve the application directory as static files: every URL 404'd and the
source tree became downloadable.

These tests pin the contract so the file cannot silently regress to that state.
"""
import io
import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTACCESS = os.path.join(REPO_ROOT, '.htaccess')


def _read():
    with open(HTACCESS, encoding='utf-8') as fh:
        return fh.read()


def test_htaccess_has_passenger_block():
    """Passenger must be configured, or the app is never started."""
    text = _read()
    for directive in ('PassengerAppRoot', 'PassengerBaseURI',
                      'PassengerAppType', 'PassengerStartupFile',
                      'PassengerPython'):
        assert directive in text, f'{directive} missing from .htaccess'


def test_htaccess_points_at_the_passenger_startup_file():
    """Passenger loads the named file; application.wsgi is not the right one."""
    assert 'PassengerStartupFile passenger_wsgi.py' in _read()


def test_htaccess_has_no_mod_wsgi_rewrite():
    """A rewrite to application.wsgi is the exact cause of the 404 outage.

    mod_wsgi is not installed on cPanel, so the rule does nothing there while
    still shadowing Passenger's routing.
    """
    text = _read()
    live = '\n'.join(line for line in text.splitlines()
                     if not line.lstrip().startswith('#'))
    assert 'application.wsgi' not in live, \
        'the mod_wsgi rewrite must not be active in .htaccess'


def test_htaccess_denies_source_and_secret_files():
    """The outage also exposed source files; keep the deny rules."""
    text = _read()
    blocks = re.findall(r'<FilesMatch\s+"([^"]+)"', text)
    assert blocks, 'no <FilesMatch> deny rules in .htaccess'
    joined = ' '.join(blocks)
    assert re.search(r'\bpy\b|\.py', joined), \
        'Python source must be denied'
    assert 'env' in joined, 'environment files must be denied'
    assert 'Require all denied' in text


def test_htaccess_paths_are_not_placeholders():
    """Placeholders silently point Passenger at a directory that does not exist."""
    for value in re.findall(r'^Passenger(?:AppRoot|Python)\s+"([^"]+)"',
                            _read(), re.MULTILINE):
        assert 'USER' not in value and 'APP' not in value, \
            f'placeholder left in .htaccess: {value}'
        assert value.startswith('/home/'), value


def test_passenger_wsgi_exports_application():
    """Passenger looks for a WSGI callable named `application`."""
    path = os.path.join(REPO_ROOT, 'passenger_wsgi.py')
    result = subprocess.run(
        ['grep', '-qE', r'^from run import app as application|'
         r'^application\s*=', path],
        capture_output=True)
    assert result.returncode == 0, \
        'passenger_wsgi.py must export `application`'


def test_app_handles_the_passenger_base_uri_prefix(tmp_path, monkeypatch):
    """`PassengerBaseURI /RFQ` sets SCRIPT_NAME; links must keep the prefix.

    Passenger strips `/RFQ` from the path and passes it as SCRIPT_NAME. Flask
    must put it back into generated URLs, or every link and form action points
    one level too high and the site is unusable.
    """
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{tmp_path}/prefix.db')
    monkeypatch.setenv('FLASK_ENV', 'production')
    monkeypatch.setenv('AUTO_UPGRADE_DB', 'true')

    from app.extensions import create_app
    app = create_app('production')

    def get(script_name, path_info):
        captured = {}

        def start_response(status, headers, exc_info=None):
            captured['status'] = status
            return lambda chunk: None

        environ = {
            'REQUEST_METHOD': 'GET',
            'SCRIPT_NAME': script_name,
            'PATH_INFO': path_info,
            'QUERY_STRING': '',
            'SERVER_NAME': 'example.com',
            'SERVER_PORT': '443',
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': 'https',
            'wsgi.input': io.BytesIO(b''),
            'wsgi.errors': io.StringIO(),
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
            'HTTP_HOST': 'example.com',
        }
        body = b''.join(app.wsgi_app(environ, start_response))
        return captured.get('status'), body

    status, body = get('/RFQ', '/auth/login')
    assert status and status.startswith('200'), status
    assert b'/RFQ/auth/login' in body, \
        'generated URLs must include the Passenger base URI'


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__, '-v']))