#!/usr/bin/env python3
"""
WSGI configuration for cPanel/Apache deployment
"""

import os
import sys

# Add the application directory to the Python path
virtual_env = os.path.expanduser("~/virtualenv/npl_erp/3.11/lib/python3.11/site-packages")
if os.path.exists(virtual_env):
    sys.path.insert(0, virtual_env)

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Set the Flask environment
os.environ['FLASK_ENV'] = 'production'
os.environ['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# Import and create the Flask application
from run import app as application

# Application is exposed as 'application' for WSGI
