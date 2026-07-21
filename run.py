#!/usr/bin/env python3
"""
NPL Fasteners ERP - Run Script
Flask application entry point
"""

import os
import sys

# Add current directory to Python path (for WSGI)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

