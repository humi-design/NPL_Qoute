from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_cors import CORS
from app.config import config

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
bcrypt = Bcrypt()
migrate = Migrate()
cors = CORS()


def create_app(config_name='default'):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)
    
    # Register blueprints BEFORE initializing CSRF
    from app.main import main_bp
    from app.auth import auth_bp
    from app.customer import customer_bp
    from app.rfq import rfq_bp
    from app.product import product_bp
    from app.material import material_bp
    from app.machine import machine_bp
    from app.vendor import vendor_bp
    from app.supplier import supplier_bp
    from app.procurement import procurement_bp
    from app.orders import orders_bp
    from app.inventory import inventory_bp
    from app.documents import documents_bp
    from app.alerts import alerts_bp
    from app.notifications import notifications_bp
    from app.costing import costing_bp
    from app.process import process_bp
    from app.quotation import quotation_bp
    from app.report import report_bp
    from app.setting import setting_bp
    from app.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(customer_bp, url_prefix='/customers')
    app.register_blueprint(rfq_bp, url_prefix='/rfqs')
    app.register_blueprint(product_bp, url_prefix='/products')
    app.register_blueprint(material_bp, url_prefix='/materials')
    app.register_blueprint(machine_bp, url_prefix='/machines')
    app.register_blueprint(vendor_bp, url_prefix='/vendors')
    app.register_blueprint(supplier_bp, url_prefix='/suppliers')
    app.register_blueprint(procurement_bp, url_prefix='/procurement')
    app.register_blueprint(orders_bp, url_prefix='/orders')
    app.register_blueprint(inventory_bp, url_prefix='/inventory')
    app.register_blueprint(documents_bp, url_prefix='/documents')
    app.register_blueprint(alerts_bp, url_prefix='/alerts')
    app.register_blueprint(notifications_bp, url_prefix='/notifications')
    app.register_blueprint(costing_bp, url_prefix='/costing')
    app.register_blueprint(process_bp, url_prefix='/processes')
    app.register_blueprint(quotation_bp, url_prefix='/quotations')
    app.register_blueprint(report_bp, url_prefix='/reports')
    app.register_blueprint(setting_bp, url_prefix='/settings')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize CSRF protection AFTER blueprints are registered
    csrf.init_app(app)
    
    # Exempt API blueprint from CSRF protection
    csrf.exempt(api_bp)

    # Central write guard.
    #
    # The newer modules (orders, procurement, inventory, documents, costing)
    # call their own _require_write() helper. The older master-data modules
    # (customer, vendor, material, product, process, machine, quotation, rfq,
    # setting) never did, which let a 'read_only' account create, edit and
    # delete records. This guard closes that gap for every mutating request in
    # one place, so a future route cannot silently reopen it.
    #
    # Endpoints that are personal state or pure computation (own profile, own
    # alert acknowledgement, own notifications, the calculators) stay allowed.
    _WRITE_EXEMPT_ENDPOINTS = {
        'auth.login', 'auth.logout', 'auth.profile',
        'alerts.acknowledge', 'alerts.unacknowledge',
        'notifications.mark_read', 'notifications.mark_all_read',
        'notifications.delete',
        'api.calculate_costing', 'api.calculate_material',
        'main.calculate_material',
        'quotation.cost_item',
    }

    @app.before_request
    def _enforce_write_permission():
        from flask import request, flash, redirect, url_for
        from flask_login import current_user
        if request.method in ('GET', 'HEAD', 'OPTIONS'):
            return None
        if request.endpoint is None:
            return None
        if request.endpoint in _WRITE_EXEMPT_ENDPOINTS:
            return None
        if request.endpoint == 'static':
            return None
        # Anonymous requests are handled by @login_required / the login view.
        if not current_user.is_authenticated:
            return None
        if current_user.has_permission('write'):
            return None
        flash('You do not have permission to make changes.', 'danger')
        return redirect(request.referrer or url_for('main.dashboard'))
    
    # Friendly error pages.
    #
    # Without these, Flask serves its own unstyled pages and a 500 is only
    # visible in the console. The page is intentionally standalone (it does not
    # extend base.html) because the database may be what failed.
    _ERROR_PAGES = {
        403: ('Access denied',
              'You do not have permission to view this page.'),
        404: ('Page not found',
              'The page you are looking for does not exist or has been moved.'),
        405: ('Method not allowed',
              'That action is not allowed on this page.'),
        500: ('Something went wrong',
              'An unexpected error occurred. The problem has been logged and '
              'will be looked into.'),
    }

    def _render_error(code):
        from flask import render_template
        title, message = _ERROR_PAGES[code]
        return render_template('errors/error.html', code=code,
                               title=title, message=message), code

    @app.errorhandler(403)
    def _handle_403(e):
        return _render_error(403)

    @app.errorhandler(404)
    def _handle_404(e):
        return _render_error(404)

    @app.errorhandler(405)
    def _handle_405(e):
        return _render_error(405)

    @app.errorhandler(500)
    def _handle_500(e):
        # Log the traceback; never leak it to the browser.
        app.logger.exception('Unhandled server error')
        return _render_error(500)

    # Login manager settings
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return User.query.get(int(user_id))
    
    # Context processor for global variables
    @app.context_processor
    def inject_user():
        from flask_login import current_user
        from sqlalchemy import desc
        notifications = []
        alert_count = 0
        if current_user.is_authenticated:
            from app.models import Notification
            notifications = Notification.query.filter_by(
                user_id=current_user.id, is_read=False
            ).order_by(desc(Notification.created_at)).limit(5).all()
            try:
                from app.utils import count_unacknowledged_alerts
                alert_count = count_unacknowledged_alerts(current_user)
            except Exception:
                alert_count = 0

        def notification_link(notification):
            """Resolve a notification's referenced record to a URL (or None)."""
            try:
                from app.notifications.routes import resolve_notification_link
                return resolve_notification_link(notification)
            except Exception:
                return None

        unread_notifications = 0
        if current_user.is_authenticated:
            from app.models import Notification as _N
            unread_notifications = _N.query.filter_by(
                user_id=current_user.id, is_read=False).count()

        from app.utils import get_company_settings
        try:
            company = get_company_settings()
        except Exception:
            company = {}

        return dict(notifications=notifications, alert_count=alert_count,
                    unread_notifications=unread_notifications,
                    notification_link=notification_link,
                    company=company)
    
    # Create upload directories.
    #
    # These are anchored to the project root (the parent of this package) rather
    # than the process working directory, because Passenger/mod_wsgi do not
    # guarantee the CWD is the application root. Creation is best-effort: a
    # read-only or quota-limited document root must not stop the whole WSGI
    # application from loading, which would take every URL on the site down.
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_folders = [
        'uploads/customers',
        'uploads/products',
        'uploads/rfqs',
        'uploads/quotations',
        'uploads/processes',
        'uploads/vendors',
        'uploads/suppliers',
        'uploads/machines',
        'uploads/orders',
        'uploads/lots',
        'uploads/inspections',
        'uploads/dispatches',
        'uploads/documents'
    ]
    for folder in upload_folders:
        try:
            os.makedirs(os.path.join(project_root, folder), exist_ok=True)
        except OSError as exc:
            app.logger.warning('Could not create upload directory %s: %s',
                               folder, exc)

    return app
