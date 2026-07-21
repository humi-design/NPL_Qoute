#!/usr/bin/env python3
"""
WSGI entry point for Flask application
Compatible with cPanel, Apache mod_wsgi, and Passenger
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set production mode
os.environ['FLASK_ENV'] = 'production'

# Import Flask application from run.py
from run import app as application
