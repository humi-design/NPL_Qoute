#!/usr/bin/env python3
"""
NPL Fasteners ERP - Run Script
Flask application entry point

For cPanel deployment:
- Ensure application.wsgi calls this module
- Set FLASK_ENV=production in .env
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Determine environment
env = os.environ.get('FLASK_ENV', 'production')
debug_mode = env == 'development'

# Create the Flask application
from app.extensions import create_app

app = create_app(env)

# Get host/port from environment or use defaults
host = os.environ.get('FLASK_HOST', '0.0.0.0')
port = int(os.environ.get('FLASK_PORT', 5000))

# Run the application
if __name__ == '__main__':
    app.run(host=host, port=port, debug=debug_mode)

