#!/bin/bash
# =============================================================================
# cPanel Deployment Script for NPL Fasteners ERP
# =============================================================================
# This script deploys the Flask application to cPanel shared hosting
# Run this script after uploading files via FTP/cPanel File Manager
#
# Usage: bash deploy_cpanel.sh
# =============================================================================

set -e

echo "=========================================="
echo "NPL ERP - cPanel Deployment Script"
echo "=========================================="

# Get the domain name from cPanel or set manually
DOMAIN="${DOMAIN:-npl.yourdomain.com}"
APP_DIR="$HOME/npl_erp"
VENV_DIR="$HOME/virtualenv/npl_erp/3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on cPanel
if [ ! -d "$HOME/public_html" ]; then
    log_error "This doesn't appear to be a cPanel environment."
    log_error "Please ensure you're running this script on your cPanel server."
    exit 1
fi

log_info "Starting deployment for domain: $DOMAIN"

# Step 1: Create virtual environment
log_info "Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    /opt/cpanel/ea-python311/root/usr/bin/python3.11 -m venv "$VENV_DIR"
    log_info "Virtual environment created at: $VENV_DIR"
else
    log_warn "Virtual environment already exists."
fi

# Step 2: Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 3: Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
log_info "Installing Python dependencies..."
pip install -r requirements.txt

# Step 5: Create .env file if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    log_info "Creating .env file from template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    
    # Update with actual values
    sed -i "s/DB_HOST=localhost/DB_HOST=localhost/" "$APP_DIR/.env"
    
    log_warn "Please edit $APP_DIR/.env with your actual database credentials!"
fi

# Step 6: Initialize database
log_info "Initializing database..."
cd "$APP_DIR"
export FLASK_APP=run.py
export FLASK_ENV=production

# Create database if needed (requires MySQL credentials)
log_warn "Please create the MySQL database manually via cPanel:"
log_warn "1. Go to MySQL Databases in cPanel"
log_warn "2. Create a database named 'erp_fastener'"
log_warn "3. Create a user and grant privileges"
log_warn "4. Update .env with the credentials"
log_warn "5. Run: python seed_data.py"

# Step 7: Set proper permissions
log_info "Setting permissions..."
chmod -R 755 "$APP_DIR"
chmod -R 775 "$APP_DIR/uploads"
chmod -R 775 "$APP_DIR/migrations"

# Step 8: Test the application
log_info "Testing application..."
timeout 10 python run.py &
sleep 5
pkill -f "python run.py" || true

log_info "=========================================="
log_info "Deployment preparation complete!"
log_info "=========================================="
log_info ""
log_info "Next steps:"
log_info "1. Create MySQL database via cPanel"
log_info "2. Update .env with database credentials"
log_info "3. Run: source ~/virtualenv/npl_erp/3.11/bin/activate"
log_info "4. Run: python seed_data.py"
log_info "5. Restart Apache: systemctl restart httpd"
log_info ""
log_info "Or use Setup Python App in cPanel:"
log_info "1. cPanel > Setup Python App"
log_info "2. Create application with this directory"
log_info "3. Set entry point to: application.wsgi"
log_info "4. Install requirements"
log_info "=========================================="
