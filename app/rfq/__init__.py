from flask import Blueprint

rfq_bp = Blueprint('rfq', __name__)

from app.rfq import routes
