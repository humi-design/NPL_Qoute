# NPL Fasteners ERP

## Production-Ready Flask ERP System for Fastener Quotation & Manufacturing Costing

---

## Features

### Core Modules
- **Dashboard**: Real-time KPIs, charts, quick actions
- **Customers**: Full CRM with addresses, contacts, payment terms
- **RFQs**: Request for Quotation management with file uploads
- **Products**: Reusable Product Library with drawing uploads, revision tracking
- **Materials**: Material master with density, rates, HS codes
- **Raw Material Calculator**: Dynamic blank calculations with unlimited allowances
- **Machines**: Machine master with hourly rates
- **Vendors**: Vendor management with rate history
- **Processes**: Dynamic manufacturing operations (internal & vendor)
- **Process Templates**: Reusable manufacturing routes
- **Quotations**: Full costing with quantity breaks, PDF export
- **Reports**: Sales analysis, margin analysis, exports
- **Settings**: System configuration, currencies, allowances

### Key Capabilities
- **Dynamic Costing Engine**: No hardcoded operations
- **Role-Based Access Control**: Admin, Sales, Production, Accounts, Read Only
- **REST APIs**: Mobile app ready
- **File Uploads**: Drawings, images, CAD files
- **PDF Generation**: Professional quotation PDFs
- **Excel Export**: Data export to Excel
- **Revision History**: Track all changes
- **Global Search**: Search across all entities

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your MySQL database settings
```

### 3. Initialize Database
```bash
python seed_data.py
```

### 4. Run the Application
```bash
python run.py
```

### Default Login
- Admin: admin / admin123
- Sales: sales / sales123
- Production: production / prod123

---

## For Production Deployment with Apache + mod_wsgi

See README.md in the repository for full deployment instructions.

---

## Technology Stack

- Flask 3.0, SQLAlchemy, Flask-Login, WTForms
- MySQL (Namecheap cPanel compatible)
- Bootstrap 5, jQuery, DataTables, Chart.js
- ReportLab for PDF generation
- Pandas, openpyxl for Excel export
