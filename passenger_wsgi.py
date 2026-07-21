#!/usr/bin/env python3
"""
WSGI configuration for cPanel/Apache
Application entry point - do NOT add import statements after this comment
"""

import os
import sys

# Add the application directory to the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set production environment
os.environ['FLASK_ENV'] = 'production'

# Import the Flask application
from run import app

# Expose the application object (required by Passenger)
application = app
