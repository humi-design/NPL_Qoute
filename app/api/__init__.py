from flask import Blueprint

api_bp = Blueprint('api', __name__)

# Import routes first, then apply CSRF exemption at extension level
from app.api import routes
