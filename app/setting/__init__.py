from flask import Blueprint

setting_bp = Blueprint('setting', __name__)

from app.setting import routes
