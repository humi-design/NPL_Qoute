import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f"mysql+pymysql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}@" \
        f"{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT')}/{os.environ.get('DB_NAME')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # File Upload
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH') or 16 * 1024 * 1024)  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'step', 'stp', 'iges', 'igs', 'dxf', 'dwg', 'zip'}
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = False  # Set True if using HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CSRF Configuration
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None  # No time limit on CSRF tokens
    
    # Company Settings
    COMPANY_NAME = os.environ.get('COMPANY_NAME', 'NPL Fasteners')
    COMPANY_ADDRESS = os.environ.get('COMPANY_ADDRESS', '')
    COMPANY_PHONE = os.environ.get('COMPANY_PHONE', '')
    COMPANY_EMAIL = os.environ.get('COMPANY_EMAIL', '')
    COMPANY_GST = os.environ.get('COMPANY_GST', '')
    COMPANY_LOGO = os.environ.get('COMPANY_LOGO', 'default_logo.png')
    
    # Currency Settings
    DEFAULT_CURRENCY = os.environ.get('DEFAULT_CURRENCY', 'INR')
    USD_RATE = float(os.environ.get('USD_RATE', 83.0))
    EUR_RATE = float(os.environ.get('EUR_RATE', 90.0))
    GBP_RATE = float(os.environ.get('GBP_RATE', 100.0))
    AED_RATE = float(os.environ.get('AED_RATE', 22.5))
    
    # Costing Defaults
    DEFAULT_OVERHEAD_PERCENT = float(os.environ.get('DEFAULT_OVERHEAD_PERCENT', 15))
    DEFAULT_PROFIT_PERCENT = float(os.environ.get('DEFAULT_PROFIT_PERCENT', 20))
    DEFAULT_SCRAP_PERCENT = float(os.environ.get('DEFAULT_SCRAP_PERCENT', 2))


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    WTF_CSRF_ENABLED = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': ProductionConfig
}
