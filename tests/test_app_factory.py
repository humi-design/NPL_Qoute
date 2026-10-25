"""Application-factory robustness tests.

The app factory runs on every WSGI worker boot on shared hosting (cPanel
Passenger / mod_wsgi). If anything in it raises, the WSGI application object is
never created and *every* URL on the site fails to load. These tests pin the
behaviours that keep that from happening.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SECRET_KEY', 'test-secret')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')


def _create_app():
    from app.extensions import create_app
    return create_app('production')


def test_create_app_survives_unwritable_upload_root(monkeypatch):
    """A read-only document root must not stop the app from loading.

    Regression: create_app() created the upload directories unguarded, so a
    permission error on a new folder (added by a later release, and therefore
    absent on an existing server) crashed app creation and took the whole site
    down with it.
    """
    real_makedirs = os.makedirs

    def deny_uploads(path, *args, **kwargs):
        if 'uploads' in str(path):
            raise PermissionError(13, 'Permission denied', str(path))
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(os, 'makedirs', deny_uploads)
    app = _create_app()
    assert app is not None
    assert len(list(app.url_map.iter_rules())) > 100


def test_upload_dirs_are_anchored_to_project_root(monkeypatch, tmp_path):
    """Upload directories must not depend on the process working directory.

    Passenger and mod_wsgi do not guarantee the CWD is the application root, so
    relative paths could create the folders in the wrong place (or fail).
    """
    import app.extensions as extensions

    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(extensions.__file__)))
    created = []
    real_makedirs = os.makedirs

    def record(path, *args, **kwargs):
        created.append(str(path))
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(os, 'makedirs', record)
    monkeypatch.chdir(tmp_path)
    _create_app()

    upload_paths = [p for p in created if 'uploads' in p]
    assert upload_paths, 'no upload directories were created'
    for path in upload_paths:
        assert os.path.isabs(path), f'upload dir not absolute: {path}'
        assert path.startswith(project_root + os.sep), \
            f'upload dir not under project root: {path}'
