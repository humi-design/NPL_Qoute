"""Bring an existing database up to date with the current models.

Deploying newer code onto a database created by an older release fails at
runtime with "no such table" / "unknown column" errors, because the application
never creates tables at startup. This script closes that gap.

It is additive and idempotent:

* missing tables are created;
* missing columns are added to existing tables.

Nothing is ever dropped, renamed or rewritten, and existing rows are never
modified beyond backfilling the constant model defaults into columns this run
creates. Running it twice is a no-op.

The same upgrader runs automatically when the application starts, so this
script is normally only needed to upgrade a database without booting the app,
or to see a report of what changed.

Usage (on cPanel, from the application root, with the virtualenv active):

    python upgrade_db.py

Or use the wrapper, which activates the virtualenv for you:

    ./run_upgrade.sh

It reads the same ``DATABASE_URL`` / ``DB_*`` environment variables (or ``.env``)
as the application.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

# Anchor .env to the project root. load_dotenv() defaults to the current working
# directory, which under Passenger or a cron job is not necessarily the
# application root, and the script would then silently fall back to a different
# database.
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Must be set before app.config is imported: the config class reads this at
# import time, and create_app() would otherwise upgrade the schema first and
# leave this script with nothing to report.
os.environ['AUTO_UPGRADE_DB'] = 'false'

from app.extensions import create_app  # noqa: E402
from app.schema_upgrade import upgrade_database  # noqa: E402


def main():
    app = create_app(os.environ.get('FLASK_ENV', 'production'))
    app.config['AUTO_UPGRADE_DB'] = False

    url = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    safe_url = url
    if '@' in url and '//' in url:
        scheme, rest = url.split('//', 1)
        safe_url = f'{scheme}//***@{rest.split("@", 1)[1]}'
    print(f'Target database: {safe_url}')

    report = upgrade_database(app)

    if report['tables_created']:
        print(f'\nCreated {len(report["tables_created"])} missing table(s):')
        for name in report['tables_created']:
            print(f'  + {name}')
    else:
        print('\nAll tables already present.')

    if report['columns_added']:
        print(f'\nAdded {len(report["columns_added"])} missing column(s):')
        for name in report['columns_added']:
            print(f'  + {name}')
    else:
        print('All columns already present.')

    if report['values_backfilled']:
        print(f'\nBackfilled default values in '
              f'{len(report["values_backfilled"])} column(s):')
        for name in report['values_backfilled']:
            print(f'  ~ {name}')

    print('\nDatabase is up to date. No existing data was modified.')


if __name__ == '__main__':
    main()
