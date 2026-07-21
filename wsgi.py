#!/usr/bin/env python3
"""
Minimal WSGI entry point for cPanel
"""
import os
os.environ['FLASK_ENV'] = 'production'
from run import app as application
