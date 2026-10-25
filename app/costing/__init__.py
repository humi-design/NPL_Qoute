from flask import Blueprint

costing_bp = Blueprint('costing', __name__)

from app.costing import routes
