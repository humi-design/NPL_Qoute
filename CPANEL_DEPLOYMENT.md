# cPanel Deployment Guide for NPL Fasteners ERP

This guide provides step-by-step instructions for deploying the NPL Fasteners ERP system on cPanel shared hosting.

---

## Prerequisites

Before starting, ensure you have:
- cPanel access with Python App support
- SSH access (optional but recommended)
- MySQL database created via cPanel
- Domain pointed to your hosting

---

## Method 1: Using cPanel Python App (Recommended)

### Step 1: Upload Files

1. Log into cPanel
2. Open **File Manager**
3. Navigate to `public_html` or your domain's document root
4. Create a new folder called `npl_erp`
5. Upload all project files to this folder

### Step 2: Create Python Application

1. In cPanel, go to **Setup Python App**
2. Click **Create Application**
3. Configure:
   - **Python version**: `3.11`
   - **Application root**: `/home/username/npl_erp`
   - **Application URL**: Your domain or subdomain
   - **Application startup file**: `application.wsgi`
   - **Passenger log file**: `logs/error.log`
4. Click **Create**

### Step 3: Install Dependencies

1. In the Python App setup, find **Enter to the virtual environment** command
2. SSH into your server or use the terminal in cPanel
3. Run:
   ```bash
   cd ~/npl_erp
   source ~/virtualenv/npl_erp/3.11/bin/activate
   pip install -r requirements.txt
   ```

### Step 4: Configure Database

1. Go to **MySQL Databases** in cPanel
2. Create a database named: `erp_fastener`
3. Create a user with full privileges
4. Note down the credentials

### Step 5: Update Environment

1. In File Manager, edit `npl_erp/.env`
2. Update these values:
   ```env
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=your_mysql_user
   DB_PASSWORD=your_mysql_password
   DB_NAME=erp_fastener
   
   SECRET_KEY=your-very-secure-secret-key-min-32-chars
   FLASK_ENV=production
   ```

### Step 6: Initialize Database

1. SSH into your server or use cPanel Terminal
2. Run:
   ```bash
   cd ~/npl_erp
   source ~/virtualenv/npl_erp/3.11/bin/activate
   python seed_data.py
   ```

### Step 7: Restart Application

1. Go to **Setup Python App** in cPanel
2. Click **Restart** button for your application

---

## Method 2: Traditional Apache + mod_wsgi

### Step 1: Upload Files

Upload all files to your domain's document root via FTP or File Manager.

### Step 2: Create Virtual Environment

SSH into your server:
```bash
cd ~
python3.11 -m venv virtualenv/npl_erp/3.11
source ~/virtualenv/npl_erp/3.11/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Apache

Create/edit `.htaccess` in your application root:

```apache
RewriteEngine On
RewriteBase /
RewriteRule ^(.*)$ application.wsgi/$1 [QSA,L]

<FilesMatch "\.(env|pyc|sqlite)$">
    Order allow,deny
    Deny from all
</FilesMatch>
```

### Step 4: Update Configuration

Follow Steps 4-7 from Method 1.

---

## Directory Structure on cPanel

```
/home/username/
├── npl_erp/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── forms.py
│   │   ├── utils.py
│   │   ├── extensions.py
│   │   ├── api/
│   │   ├── auth/
│   │   ├── customer/
│   │   ├── main/
│   │   ├── material/
│   │   ├── machine/
│   │   ├── process/
│   │   ├── product/
│   │   ├── quotation/
│   │   ├── report/
│   │   ├── rfq/
│   │   ├── setting/
│   │   ├── vendor/
│   │   ├── static/
│   │   └── templates/
│   ├── migrations/
│   ├── uploads/
│   ├── .env
│   ├── .env.example
│   ├── .htaccess
│   ├── application.wsgi
│   ├── requirements.txt
│   ├── run.py
│   └── seed_data.py
├── logs/
│   └── error.log
└── virtualenv/
    └── npl_erp/
        └── 3.11/
            ├── bin/
            ├── include/
            └── lib/
                └── python3.11/
                    └── site-packages/
```

---

## Troubleshooting

### Issue: Application won't start

1. Check error logs: `~/logs/error.log`
2. Verify `.env` file exists and has correct values
3. Ensure virtual environment is activated
4. Check MySQL connection

### Issue: Database connection error

1. Verify MySQL credentials in `.env`
2. Ensure database exists in cPanel
3. Check user has proper privileges
4. Try connecting via phpMyAdmin

### Issue: Static files not loading

1. Verify `.htaccess` allows static files
2. Check Apache Alias configuration
3. Ensure `/static/` folder exists and has files

### Issue: Import errors

1. Reinstall dependencies:
   ```bash
   source ~/virtualenv/npl_erp/3.11/bin/activate
   pip install --force-reinstall -r requirements.txt
   ```

### Issue: Permission denied

1. Set proper permissions:
   ```bash
   chmod 755 ~/npl_erp
   chmod 775 ~/npl_erp/uploads
   chmod 775 ~/npl_erp/migrations
   ```

---

## Security Checklist

- [ ] Change default admin password after first login
- [ ] Update `SECRET_KEY` in `.env`
- [ ] Use strong MySQL password
- [ ] Enable HTTPS (SSL certificate)
- [ ] Review file permissions
- [ ] Backup database regularly

---

## Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Sales | sales | sales123 |
| Production | production | prod123 |

**⚠️ IMPORTANT**: Change these passwords immediately after first login!

---

## Automatic Backup Script

Create `backup.sh` in your home directory:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/backups"
mkdir -p $BACKUP_DIR

# Database backup
mysqldump erp_fastener > "$BACKUP_DIR/db_$DATE.sql"

# Files backup
tar -czf "$BACKUP_DIR/files_$DATE.tar.gz" npl_erp

# Keep only last 7 backups
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Run daily via cPanel Cron Jobs:
```bash
0 2 * * * /home/username/backup.sh
```

---

## Support

For issues:
1. Check error logs first
2. Verify all configuration files
3. Ensure dependencies are installed
4. Contact your hosting provider if issues persist
