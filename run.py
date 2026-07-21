#!/usr/bin/env python3
"""
NPL Fasteners ERP - Run Script
Flask application entry point
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine environment
env = os.environ.get('FLASK_ENV', 'production')

# Create and run app
from app.extensions import create_app

app = create_app(env)

if __name__ == '__main__':
    # Development mode
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = env == 'development'
    
    app.run(host=host, port=port, debug=debug)
