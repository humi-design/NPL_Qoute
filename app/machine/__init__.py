from flask import Blueprint

machine_bp = Blueprint('machine', __name__)

from app.machine import routes
