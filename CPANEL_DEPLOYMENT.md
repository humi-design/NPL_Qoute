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
   - **Application startup file**: `passenger_wsgi.py`
   - **Passenger log file**: `logs/error.log`
4. Click **Create**

> **The startup file must be `passenger_wsgi.py`, not `application.wsgi`.**
> cPanel runs Python apps through Passenger, and Passenger loads the file you
> name here. `passenger_wsgi.py` exports the WSGI callable as `application`,
> which is the name Passenger expects; `application.wsgi` does the same but
> carries a `.wsgi` extension that Passenger does not look for. Setting the
> wrong one makes cPanel serve the directory as static files — every URL
> returns 404, and your source code becomes downloadable. See
> [Troubleshooting](#issue-every-page-returns-404) below.

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

### Step 6: Initialize / Upgrade Database

**The application now upgrades the schema by itself on startup.** Deploying
newer code onto an older database creates any missing tables and columns
automatically, so there is no required manual step and the site cannot be left
broken by a forgotten command. Set `AUTO_UPGRADE_DB=false` to opt out.

The commands below remain useful for a fresh install, or to upgrade a database
without booting the app and see a report of what changed.

**Fresh database** — load the schema, then the additive migration:

```bash
cd ~/npl_erp
source ~/virtualenv/npl_erp/3.11/bin/activate
python seed_data.py          # creates tables + demo data
```

**Existing database (upgrading an install that is already live)** — run the
additive upgrader to see what changed. New code needs tables the old database
does not have, and without this step pages fail with `no such table` (HTTP 500):

```bash
cd ~/npl_erp
source ~/virtualenv/npl_erp/3.11/bin/activate
python upgrade_db.py
```

This is safe to run repeatedly and never deletes or rewrites existing rows.

### Step 6b: Namecheap (and any host without a Python terminal)

Namecheap shared hosting runs **MySQL 8**, and its Python App terminal is
sometimes unavailable or times out on long scripts.

**Simplest: cPanel's Run Script button.** In **Setup Python App**, find your
application and use its **Run Script** control, and enter:

```
upgrade_db.py
```

> **Select `upgrade_db.py`, not `run_upgrade.sh`.** Run Script executes the file
> you name **with the application's Python interpreter**. `run_upgrade.sh` is a
> bash script, so Python cannot parse it and fails immediately with
> `SyntaxError: invalid syntax` on its first `cd ... || exit 1` line. That is
> expected, not a fault in the script. Only `upgrade_db.py` is a Python file.

`upgrade_db.py` is self-contained: it finds the application root, reads `.env`
from there regardless of the working directory, and prints a report of what it
changed.

**In a real shell** (cPanel *Terminal*, or SSH), either form works:

```bash
~/npl_erp/run_upgrade.sh              # finds the virtualenv itself; any directory
```

```bash
# CORRECT - interpreter, then the script, exactly one path
/home/CPANEL_USER/virtualenv/npl_erp/3.11/bin/python /home/CPANEL_USER/npl_erp/upgrade_db.py
```

Three things go wrong here often enough to be worth naming:

* **Do not select `run_upgrade.sh` in Run Script.** It is bash; Run Script uses
  Python. See above.
* **Do not put `source .../activate` and `python upgrade_db.py` on one line.**
  They are two commands. Pasted together, the shell runs
  `python source .../activate` and Python reports
  `can't open file '.../source': [Errno 2] No such file or directory`.
* **Do not pass the interpreter path twice.** On CloudLinux/cPanel the
  virtualenv's `bin/python` is a shell script that runs the real interpreter,
  not the interpreter itself. Writing
  `.../bin/python .../bin/python upgrade_db.py` makes Python read that shell
  script and fail with `SyntaxError: invalid syntax` on its `[[ $EUID -eq 0 ]]`
  line. `run_upgrade.sh` avoids this one.

If the terminal is unavailable and Run Script is not offered, run the same
command through a cron job (**cPanel → Cron Jobs**, "Once Per Minute", then
delete the job afterwards):

```
cd /home/CPANEL_USER/npl_erp && /home/CPANEL_USER/virtualenv/npl_erp/3.11/bin/python upgrade_db.py >> logs/upgrade.log 2>&1
```

**Fallback: phpMyAdmin.** Open **phpMyAdmin**, select your database, click
**SQL**, and paste the contents of `migrate.sql`, then **Go**.

`migrate.sql` is written to run on both MySQL 8 and MariaDB. It guards every
`ALTER TABLE` and `CREATE INDEX` with an `information_schema` check, so it is
idempotent and safe to run repeatedly. If you paste only part of the file and
MySQL stops with `ERROR 1064`, run it again from the beginning — already-applied
statements are skipped.

**Do not** run `seed_data.py` on a live database; it is for fresh installs and
inserts demo data.

### Step 7: Restart Application

1. Go to **Setup Python App** in cPanel
2. Click **Restart** button for your application

---

## Method 2: Traditional Apache + mod_wsgi (not cPanel)

> **This method does not apply to cPanel or Namecheap.** Those hosts run Python
> through Passenger, not mod_wsgi, and the rewrite below is what breaks the
> application when it reaches an application root that Passenger manages. Keep
> it for a plain Apache server where you installed `mod_wsgi` yourself.

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

On a host where you control Apache and `mod_wsgi` is installed, point the WSGI
daemon at `application.wsgi`:

```apache
WSGIDaemonProcess npl_erp python-home=/home/USER/virtualenv/npl_erp/3.11 \
    python-path=/home/USER/npl_erp
WSGIProcessGroup npl_erp
WSGIScriptAlias / /home/USER/npl_erp/application.wsgi

<FilesMatch "\.(env|pyc|sqlite|db)$">
    Require all denied
</FilesMatch>
```

Do **not** port this into a cPanel application root. There, Passenger owns the
routing and `WSGIScriptAlias` is not read; adding it only hides the app.

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

### Issue: Every page returns 404

This is the most common cPanel failure, and it is **not** a bug in the
application. It means Apache is serving your application directory as plain
files instead of handing requests to Passenger, so the app never runs.

**Confirm it in one command.** If Passenger were running the app, source files
would not be downloadable. If these return `200`, the app is not running:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://YOUR_DOMAIN/RFQ/requirements.txt
```

`200` means the file was served as text — Apache is treating `/RFQ` as a
directory of files. (`404` means Passenger is routing correctly and the URL is
simply wrong.)

**Two causes, in order of likelihood:**

1. **The `.htaccess` in your application root has no Passenger block.** Only one
   `.htaccess` can exist per directory, so uploading the repository's copy
   overwrites the block cPanel wrote. Look for these lines and restore them via
   **Setup Python App → your app → Restart**, which makes cPanel rewrite them:

   ```apache
   # DO NOT REMOVE. CLOUDLINUX PASSENGER CONFIGURATION BEGIN
   PassengerAppRoot "/home/USER/APP"
   PassengerBaseURI "/RFQ"
   PassengerAppType wsgi
   PassengerStartupFile passenger_wsgi.py
   PassengerPython "/home/USER/virtualenv/.../bin/python"
   # DO NOT REMOVE. CLOUDLINUX PASSENGER CONFIGURATION END
   ```

   The `.htaccess` shipped with this project already contains this block, so
   after deploying it you only need to check that the four paths match your
   account.

2. **The startup file is set to `application.wsgi`.** Change it to
   `passenger_wsgi.py` in **Setup Python App**.

**While the app is not running, your source code is public.** Anyone can
download `app/models.py`, `schema.sql` and the rest. Fix this before doing
anything else.

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
