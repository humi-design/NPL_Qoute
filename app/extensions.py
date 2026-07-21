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
    
    # Initialize CSRF (exempt API endpoints)
    csrf.init_app(app)
    
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
        if current_user.is_authenticated:
            from app.models import Notification
            notifications = Notification.query.filter_by(
                user_id=current_user.id, is_read=False
            ).order_by(desc(Notification.created_at)).limit(5).all()
        return dict(notifications=notifications)
    
    # Register blueprints
    from app.main import main_bp
    from app.auth import auth_bp
    from app.customer import customer_bp
    from app.rfq import rfq_bp
    from app.product import product_bp
    from app.material import material_bp
    from app.machine import machine_bp
    from app.vendor import vendor_bp
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
    app.register_blueprint(process_bp, url_prefix='/processes')
    app.register_blueprint(quotation_bp, url_prefix='/quotations')
    app.register_blueprint(report_bp, url_prefix='/reports')
    app.register_blueprint(setting_bp, url_prefix='/settings')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Create upload directories
    import os
    upload_folders = [
        'uploads/customers',
        'uploads/products',
        'uploads/rfqs',
        'uploads/quotations',
        'uploads/processes',
        'uploads/vendors',
        'uploads/machines'
    ]
    for folder in upload_folders:
        os.makedirs(folder, exist_ok=True)
    
    return app
