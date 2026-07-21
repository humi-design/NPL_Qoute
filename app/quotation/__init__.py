from flask import Blueprint

quotation_bp = Blueprint('quotation', __name__)

from app.quotation import routes
