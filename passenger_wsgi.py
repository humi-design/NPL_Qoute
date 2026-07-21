#!/usr/bin/env python3
"""
Passenger WSGI configuration for cPanel/Apache with mod_passenger

This file is used when cPanel/Passenger is configured to serve the application.
Place this file in the application root directory.
"""

import os
import sys

# Ensure the application directory is in the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

# Load environment variables
from dotenv import load_dotenv
dotenv_path = os.path.join(app_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Set production environment
os.environ['FLASK_ENV'] = 'production'

# Import and expose the Flask application
from run import app as application

# For mod_passenger compatibility
# application is the default object Passenger looks for
