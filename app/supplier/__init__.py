from flask import Blueprint

supplier_bp = Blueprint('supplier', __name__)

from app.supplier import routes
