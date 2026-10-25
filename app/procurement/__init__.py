from flask import Blueprint

procurement_bp = Blueprint('procurement', __name__)

from app.procurement import routes
