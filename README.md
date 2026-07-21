# NPL Fasteners ERP

## Production-Ready Flask ERP System for Fastener Quotation & Manufacturing Costing

**✓ cPanel Ready** - Fully compatible with shared hosting

---

## Features

### Core Modules
- **Dashboard**: Real-time KPIs, charts, quick actions
- **Customers**: Full CRM with addresses, contacts, GST
- **RFQs**: Request for Quotation management with file uploads
- **Products**: Reusable Product Library with revision tracking
- **Materials**: Material master with density, rates, HS codes
- **Raw Material Calculator**: Dynamic blank calculations
- **Machines**: Machine master with hourly rates
- **Vendors**: Vendor management with rate history
- **Processes**: Dynamic manufacturing operations
- **Process Templates**: Reusable manufacturing routes
- **Quotations**: Full costing with PDF export
- **Reports**: Sales analysis, margin analysis
- **Settings**: System configuration, currencies

### Key Capabilities
- **Dynamic Costing Engine**: No hardcoded operations
- **Role-Based Access Control**: Admin, Sales, Production
- **REST APIs**: Mobile app ready
- **File Uploads**: Drawings, images, CAD files
- **PDF Generation**: Professional quotation PDFs
- **Excel Export**: Data export to Excel

---

## cPanel Installation (Recommended)

### Method 1: Setup Python App

1. Upload files to `public_html/npl_erp/`
2. In cPanel → **Setup Python App**
3. Create application:
   - Python version: `3.11`
   - Application root: `/home/user/npl_erp`
   - Startup file: `application.wsgi`
4. Install dependencies in the virtual environment
5. Create MySQL database via cPanel
6. Update `.env` with database credentials
7. Run `python seed_data.py`
8. Restart the application

See [CPANEL_DEPLOYMENT.md](./CPANEL_DEPLOYMENT.md) for detailed instructions.

### Method 2: Traditional Deployment

1. Upload files to your domain
2. Create virtual environment:
   ```bash
   python3.11 -m venv ~/virtualenv/npl_erp/3.11
   source ~/virtualenv/npl_erp/3.11/bin/activate
   pip install -r requirements.txt
   ```
3. Configure database and `.env`
4. Run `python seed_data.py`
5. Configure Apache with `.htaccess` and `application.wsgi`

---

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with MySQL settings
python seed_data.py
python run.py
```

---

## Default Login

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| sales | sales123 | Sales |
| production | prod123 | Production |

**⚠️ Change these passwords immediately!**

---

## Technology Stack

- Flask 3.0, SQLAlchemy, Flask-Login, WTForms
- MySQL (cPanel/Namecheap compatible)
- Bootstrap 5, jQuery, DataTables, Chart.js
- ReportLab for PDF, Pandas/openpyxl for Excel
- No Docker required

---

## Files for cPanel Deployment

- `application.wsgi` - WSGI entry point
- `passenger_wsgi.py` - Passenger compatibility
- `.htaccess` - Apache configuration
- `deploy_cpanel.sh` - Deployment script
- `CPANEL_DEPLOYMENT.md` - Detailed guide
