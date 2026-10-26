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


def test_boot_upgrades_an_outdated_database(monkeypatch, tmp_path):
    """Booting on an older database must create the missing tables itself.

    Regression: the upgrade was a manual step, so deploying newer code onto an
    older database left the new tables absent and every page that touched them
    failed with "no such table" (HTTP 500) until someone ran upgrade_db.py.
    """
    from app.config import ProductionConfig

    monkeypatch.setattr(ProductionConfig, 'SQLALCHEMY_DATABASE_URI',
                        f'sqlite:///{tmp_path / "old.db"}')
    monkeypatch.setattr(ProductionConfig, 'AUTO_UPGRADE_DB', False)

    from app.extensions import create_app, db
    from app.schema_upgrade import upgrade_database

    # Build the database with only a subset of the tables, as an old release
    # would have left it.
    old_tables = {'users', 'customers'}
    app = create_app('production')
    with app.app_context():
        db.metadata.create_all(
            bind=db.engine,
            tables=[db.metadata.tables[n] for n in old_tables])
        existing = set(db.inspect(db.engine).get_table_names())
    assert existing == old_tables

    # Now boot the current code against it: the missing tables must appear.
    monkeypatch.setattr(ProductionConfig, 'AUTO_UPGRADE_DB', True)
    fresh = create_app('production')
    with fresh.app_context():
        after = set(db.inspect(db.engine).get_table_names())
    assert 'orders' in after, 'orders was not created on boot'
    assert after > old_tables

    # And a second pass is a no-op.
    report = upgrade_database(fresh)
    assert not report['tables_created']
    assert not report['columns_added']


def test_boot_does_not_touch_schema_when_disabled(monkeypatch, tmp_path):
    """AUTO_UPGRADE_DB=false must leave the schema alone."""
    from app.config import ProductionConfig

    monkeypatch.setattr(ProductionConfig, 'SQLALCHEMY_DATABASE_URI',
                        f'sqlite:///{tmp_path / "untouched.db"}')
    monkeypatch.setattr(ProductionConfig, 'AUTO_UPGRADE_DB', False)

    from app.extensions import create_app, db
    app = create_app('production')
    with app.app_context():
        tables = set(db.inspect(db.engine).get_table_names())
    assert tables == set(), f'schema was modified despite opt-out: {tables}'
