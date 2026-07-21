from flask import Blueprint

material_bp = Blueprint('material', __name__)

from app.material import routes
