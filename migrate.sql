-- NPL Fasteners ERP - Safe Migration Script
-- Only creates/alters if something is missing
-- Run in the phpMyAdmin SQL tab
--
-- Works on both MySQL 8 and MariaDB. Every ALTER TABLE / CREATE INDEX is
-- wrapped in an information_schema guard, because MySQL does not support the
-- MariaDB-only "ADD COLUMN IF NOT EXISTS" / "CREATE INDEX IF NOT EXISTS"
-- syntax (it fails with ERROR 1064). The guards make each statement a no-op
-- when it has already been applied, so this file is safe to run repeatedly.

-- =============================================
-- USERS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(120),
    role VARCHAR(20) NOT NULL DEFAULT 'read_only',
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login DATETIME
);

-- Add missing columns
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='username')=0,
  'ALTER TABLE `users` ADD COLUMN `username` VARCHAR(64) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='email')=0,
  'ALTER TABLE `users` ADD COLUMN `email` VARCHAR(120) NOT NULL UNIQUE AFTER username',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='password_hash')=0,
  'ALTER TABLE `users` ADD COLUMN `password_hash` VARCHAR(255) NOT NULL AFTER email',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='full_name')=0,
  'ALTER TABLE `users` ADD COLUMN `full_name` VARCHAR(120) AFTER password_hash',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='role')=0,
  'ALTER TABLE `users` ADD COLUMN `role` VARCHAR(20) NOT NULL DEFAULT ''read_only'' AFTER full_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='is_active')=0,
  'ALTER TABLE `users` ADD COLUMN `is_active` BOOLEAN DEFAULT TRUE AFTER role',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='users' AND column_name='last_login')=0,
  'ALTER TABLE `users` ADD COLUMN `last_login` DATETIME AFTER updated_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Add indexes
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.statistics
  WHERE table_schema=DATABASE() AND table_name='users' AND index_name='idx_username')=0,
  'CREATE INDEX `idx_username` ON `users` (username)',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.statistics
  WHERE table_schema=DATABASE() AND table_name='users' AND index_name='idx_email')=0,
  'CREATE INDEX `idx_email` ON `users` (email)',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Add admin user (password: admin123)
INSERT IGNORE INTO users (username, email, password_hash, full_name, role, is_active)
VALUES ('admin', 'admin@npl.com', '$2b$12$d3B1dphBAxJfGuLIcbaa1OMq8wElLjof1pPFoEn9v/eC82VFBNbG6', 'Administrator', 'admin', TRUE);
-- Add sales user (password: sales123)
INSERT IGNORE INTO users (username, email, password_hash, full_name, role, is_active)
VALUES ('sales', 'sales@npl.com', '$2b$12$0vR2yNYn3.wmb2epITX8iebCSKdf3h9BXaXutixaubCHvqahqpjfu', 'Sales User', 'sales', TRUE);

-- =============================================
-- CUSTOMERS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS customers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_code VARCHAR(20) NOT NULL UNIQUE,
    company_name VARCHAR(200) NOT NULL,
    contact_person VARCHAR(120),
    email VARCHAR(120),
    phone VARCHAR(30),
    mobile VARCHAR(30),
    gst_vat VARCHAR(30),
    pan VARCHAR(20),
    country VARCHAR(100) DEFAULT 'India',
    currency VARCHAR(10) DEFAULT 'INR',
    payment_terms VARCHAR(100),
    incoterms VARCHAR(50),
    shipping_terms VARCHAR(200),
    billing_address TEXT,
    shipping_address TEXT,
    delivery_address TEXT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='customer_code')=0,
  'ALTER TABLE `customers` ADD COLUMN `customer_code` VARCHAR(20) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='company_name')=0,
  'ALTER TABLE `customers` ADD COLUMN `company_name` VARCHAR(200) NOT NULL AFTER customer_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='mobile')=0,
  'ALTER TABLE `customers` ADD COLUMN `mobile` VARCHAR(30) AFTER phone',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='pan')=0,
  'ALTER TABLE `customers` ADD COLUMN `pan` VARCHAR(20) AFTER gst_vat',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='payment_terms')=0,
  'ALTER TABLE `customers` ADD COLUMN `payment_terms` VARCHAR(100) AFTER currency',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='incoterms')=0,
  'ALTER TABLE `customers` ADD COLUMN `incoterms` VARCHAR(50) AFTER payment_terms',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='shipping_terms')=0,
  'ALTER TABLE `customers` ADD COLUMN `shipping_terms` VARCHAR(200) AFTER incoterms',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='customers' AND column_name='delivery_address')=0,
  'ALTER TABLE `customers` ADD COLUMN `delivery_address` TEXT AFTER shipping_address',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- RFQS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS rfqs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rfq_number VARCHAR(30) NOT NULL UNIQUE,
    customer_id INT NOT NULL,
    subject VARCHAR(300),
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'normal',
    due_date DATE,
    conversion_date DATETIME,
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='rfqs' AND column_name='rfq_number')=0,
  'ALTER TABLE `rfqs` ADD COLUMN `rfq_number` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='rfqs' AND column_name='conversion_date')=0,
  'ALTER TABLE `rfqs` ADD COLUMN `conversion_date` DATETIME AFTER due_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='rfqs' AND column_name='created_by')=0,
  'ALTER TABLE `rfqs` ADD COLUMN `created_by` INT AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- RFQ ITEMS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS rfq_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rfq_id INT NOT NULL,
    product_id INT,
    part_description VARCHAR(300),
    drawing_number VARCHAR(100),
    material VARCHAR(100),
    quantity INT,
    target_price DECIMAL(12,4),
    required_date DATE,
    notes TEXT,
    sequence INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='rfq_items' AND column_name='sequence')=0,
  'ALTER TABLE `rfq_items` ADD COLUMN `sequence` INT DEFAULT 0 AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- PRODUCTS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    internal_code VARCHAR(30) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    customer_part_number VARCHAR(100),
    drawing_number VARCHAR(100),
    revision VARCHAR(10) DEFAULT 'A',
    material VARCHAR(100),
    weight FLOAT,
    standard VARCHAR(50),
    standard_number VARCHAR(30),
    thread VARCHAR(50),
    dimensions TEXT,
    description TEXT,
    manufacturing_template_id INT,
    status VARCHAR(20) DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_internal_code (internal_code),
    INDEX idx_product_name (product_name)
);

-- Rename product_code to internal_code if exists
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='product_code')=1
  AND (SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='internal_code')=0,
  'ALTER TABLE `products` CHANGE COLUMN `product_code` `internal_code` VARCHAR(30) NOT NULL UNIQUE',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='internal_code')=0,
  'ALTER TABLE `products` ADD COLUMN `internal_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='weight')=0,
  'ALTER TABLE `products` ADD COLUMN `weight` FLOAT AFTER material',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='standard')=0,
  'ALTER TABLE `products` ADD COLUMN `standard` VARCHAR(50) AFTER weight',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='standard_number')=0,
  'ALTER TABLE `products` ADD COLUMN `standard_number` VARCHAR(30) AFTER standard',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='products' AND column_name='dimensions')=0,
  'ALTER TABLE `products` ADD COLUMN `dimensions` TEXT AFTER thread',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- MATERIALS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS materials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    material_code VARCHAR(30) NOT NULL UNIQUE,
    material_name VARCHAR(100) NOT NULL,
    density FLOAT NOT NULL,
    grade VARCHAR(50),
    hsn_code VARCHAR(20),
    default_supplier VARCHAR(100),
    default_rate FLOAT,
    rate_unit VARCHAR(20) DEFAULT 'per_kg',
    remarks TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='material_code')=0,
  'ALTER TABLE `materials` ADD COLUMN `material_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='material_name')=0,
  'ALTER TABLE `materials` ADD COLUMN `material_name` VARCHAR(100) NOT NULL AFTER material_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='density')=0,
  'ALTER TABLE `materials` ADD COLUMN `density` FLOAT NOT NULL AFTER material_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='grade')=0,
  'ALTER TABLE `materials` ADD COLUMN `grade` VARCHAR(50) AFTER density',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='default_supplier')=0,
  'ALTER TABLE `materials` ADD COLUMN `default_supplier` VARCHAR(100) AFTER hsn_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='rate_unit')=0,
  'ALTER TABLE `materials` ADD COLUMN `rate_unit` VARCHAR(20) DEFAULT ''per_kg'' AFTER default_rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Insert sample materials
INSERT IGNORE INTO materials (material_code, material_name, density, hsn_code, is_active) VALUES
('SS303', 'Stainless Steel 303', 8.0, '7219', TRUE),
('SS304', 'Stainless Steel 304', 8.0, '7219', TRUE),
('SS316', 'Stainless Steel 316', 8.0, '7219', TRUE),
('SS410', 'Stainless Steel 410', 7.8, '7219', TRUE),
('EN8', 'EN8 Carbon Steel', 7.85, '7214', TRUE),
('EN1A', 'EN1A Free Cutting Steel', 7.85, '7214', TRUE),
('BRASS', 'Brass', 8.5, '7403', TRUE),
('COPPER', 'Copper', 8.96, '7403', TRUE),
('ALU6061', 'Aluminium 6061', 2.7, '7604', TRUE),
('TITANIUM', 'Titanium Grade 5', 4.43, '8108', TRUE);

-- =============================================
-- MATERIAL RATES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS material_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    material_id INT NOT NULL,
    rate FLOAT NOT NULL,
    unit VARCHAR(20) DEFAULT 'per_kg',
    supplier VARCHAR(100),
    effective_date DATE NOT NULL,
    is_current BOOLEAN DEFAULT TRUE,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='material_rates' AND column_name='is_current')=0,
  'ALTER TABLE `material_rates` ADD COLUMN `is_current` BOOLEAN DEFAULT TRUE AFTER effective_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='material_rates' AND column_name='created_by')=0,
  'ALTER TABLE `material_rates` ADD COLUMN `created_by` INT AFTER is_current',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- MACHINES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS machines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    machine_code VARCHAR(30) NOT NULL UNIQUE,
    machine_name VARCHAR(100) NOT NULL,
    machine_type VARCHAR(50),
    hourly_rate FLOAT NOT NULL,
    operator_name VARCHAR(100),
    power_consumption FLOAT,
    efficiency FLOAT DEFAULT 100.0,
    department VARCHAR(50),
    default_setup_time FLOAT DEFAULT 0,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='machine_code')=0,
  'ALTER TABLE `machines` ADD COLUMN `machine_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='machine_type')=0,
  'ALTER TABLE `machines` ADD COLUMN `machine_type` VARCHAR(50) AFTER machine_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='operator_name')=0,
  'ALTER TABLE `machines` ADD COLUMN `operator_name` VARCHAR(100) AFTER hourly_rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='power_consumption')=0,
  'ALTER TABLE `machines` ADD COLUMN `power_consumption` FLOAT AFTER operator_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='efficiency')=0,
  'ALTER TABLE `machines` ADD COLUMN `efficiency` FLOAT DEFAULT 100.0 AFTER power_consumption',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='machines' AND column_name='default_setup_time')=0,
  'ALTER TABLE `machines` ADD COLUMN `default_setup_time` FLOAT DEFAULT 0 AFTER department',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Insert sample machines
INSERT IGNORE INTO machines (machine_code, machine_name, hourly_rate, department, is_active) VALUES
('TRAUB', 'Traub TNL32', 800, 'Machining', TRUE),
('CNC1', 'CNC Lathe 1', 600, 'Machining', TRUE),
('CNC2', 'CNC Lathe 2', 600, 'Machining', TRUE),
('MILL', 'Milling Machine', 500, 'Machining', TRUE),
('DRILL', 'Drilling Machine', 300, 'Machining', TRUE),
('GRIND', 'Grinding Machine', 450, 'Finishing', TRUE);

-- =============================================
-- VENDORS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS vendors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_code VARCHAR(30) NOT NULL UNIQUE,
    vendor_name VARCHAR(200) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(120),
    phone VARCHAR(30),
    mobile VARCHAR(30),
    gst VARCHAR(30),
    address TEXT,
    country VARCHAR(100) DEFAULT 'India',
    default_currency VARCHAR(10) DEFAULT 'INR',
    payment_terms VARCHAR(100),
    lead_time INT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendors' AND column_name='vendor_code')=0,
  'ALTER TABLE `vendors` ADD COLUMN `vendor_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendors' AND column_name='mobile')=0,
  'ALTER TABLE `vendors` ADD COLUMN `mobile` VARCHAR(30) AFTER phone',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendors' AND column_name='country')=0,
  'ALTER TABLE `vendors` ADD COLUMN `country` VARCHAR(100) DEFAULT ''India'' AFTER address',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendors' AND column_name='payment_terms')=0,
  'ALTER TABLE `vendors` ADD COLUMN `payment_terms` VARCHAR(100) AFTER default_currency',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendors' AND column_name='lead_time')=0,
  'ALTER TABLE `vendors` ADD COLUMN `lead_time` INT AFTER payment_terms',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- VENDOR RATES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS vendor_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    rate FLOAT NOT NULL,
    unit VARCHAR(20) DEFAULT 'per_piece',
    min_quantity INT DEFAULT 1,
    lead_time_days INT,
    effective_date DATE NOT NULL,
    is_current BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_rates' AND column_name='service_name')=0,
  'ALTER TABLE `vendor_rates` ADD COLUMN `service_name` VARCHAR(100) NOT NULL AFTER vendor_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_rates' AND column_name='min_quantity')=0,
  'ALTER TABLE `vendor_rates` ADD COLUMN `min_quantity` INT DEFAULT 1 AFTER unit',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_rates' AND column_name='lead_time_days')=0,
  'ALTER TABLE `vendor_rates` ADD COLUMN `lead_time_days` INT AFTER min_quantity',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_rates' AND column_name='is_current')=0,
  'ALTER TABLE `vendor_rates` ADD COLUMN `is_current` BOOLEAN DEFAULT TRUE AFTER effective_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_rates' AND column_name='created_by')=0,
  'ALTER TABLE `vendor_rates` ADD COLUMN `created_by` INT AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- PROCESSES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS processes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    process_code VARCHAR(30) NOT NULL UNIQUE,
    process_name VARCHAR(100) NOT NULL,
    process_type VARCHAR(20) DEFAULT 'internal',
    description TEXT,
    machine_id INT,
    vendor_id INT,
    department VARCHAR(50),
    setup_time FLOAT DEFAULT 0,
    cycle_time FLOAT DEFAULT 0,
    cost_type VARCHAR(30) DEFAULT 'cycle_time',
    cost_value FLOAT DEFAULT 0,
    cost_formula VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    is_custom BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='process_code')=0,
  'ALTER TABLE `processes` ADD COLUMN `process_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='process_type')=0,
  'ALTER TABLE `processes` ADD COLUMN `process_type` VARCHAR(20) DEFAULT ''internal'' AFTER process_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='department')=0,
  'ALTER TABLE `processes` ADD COLUMN `department` VARCHAR(50) AFTER vendor_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='setup_time')=0,
  'ALTER TABLE `processes` ADD COLUMN `setup_time` FLOAT DEFAULT 0 AFTER department',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='cycle_time')=0,
  'ALTER TABLE `processes` ADD COLUMN `cycle_time` FLOAT DEFAULT 0 AFTER setup_time',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='cost_type')=0,
  'ALTER TABLE `processes` ADD COLUMN `cost_type` VARCHAR(30) DEFAULT ''cycle_time'' AFTER cycle_time',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='cost_value')=0,
  'ALTER TABLE `processes` ADD COLUMN `cost_value` FLOAT DEFAULT 0 AFTER cost_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='cost_formula')=0,
  'ALTER TABLE `processes` ADD COLUMN `cost_formula` VARCHAR(200) AFTER cost_value',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='processes' AND column_name='is_custom')=0,
  'ALTER TABLE `processes` ADD COLUMN `is_custom` BOOLEAN DEFAULT FALSE AFTER is_active',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Insert sample processes
INSERT IGNORE INTO processes (process_code, process_name, process_type, department, cost_type, is_active) VALUES
('RAW', 'Raw Material', 'internal', 'Material', 'per_kg', TRUE),
('TRAUB', 'Traub Operation', 'internal', 'Machining', 'cycle_time', TRUE),
('CNC', 'CNC Machining', 'internal', 'Machining', 'cycle_time', TRUE),
('MILL', 'Milling', 'internal', 'Machining', 'cycle_time', TRUE),
('DRILL', 'Drilling', 'internal', 'Machining', 'cycle_time', TRUE),
('TAP', 'Tapping', 'internal', 'Machining', 'cycle_time', TRUE),
('GRIND', 'Grinding', 'internal', 'Finishing', 'cycle_time', TRUE),
('HT', 'Heat Treatment', 'vendor', 'Heat Treatment', 'per_batch', TRUE),
('PLATE', 'Plating', 'vendor', 'Surface', 'per_piece', TRUE),
('INSPECT', 'Inspection', 'internal', 'Quality', 'per_piece', TRUE),
('PACK', 'Packing', 'internal', 'Packing', 'per_piece', TRUE);

-- =============================================
-- PROCESS TEMPLATES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS process_templates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    template_code VARCHAR(30) NOT NULL UNIQUE,
    template_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='process_templates' AND column_name='template_code')=0,
  'ALTER TABLE `process_templates` ADD COLUMN `template_code` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- PROCESS TEMPLATE STEPS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS process_template_steps (
    id INT AUTO_INCREMENT PRIMARY KEY,
    template_id INT NOT NULL,
    process_id INT,
    sequence INT NOT NULL DEFAULT 1,
    notes TEXT
);

-- Repair legacy installs created from an older definition that had
-- step_number but no sequence. The model requires sequence (NOT NULL).
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='process_template_steps' AND column_name='sequence')=0,
  'ALTER TABLE `process_template_steps` ADD COLUMN `sequence` INT NOT NULL DEFAULT 1',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- ALLOWANCES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS allowances (
    id INT AUTO_INCREMENT PRIMARY KEY,
    allowance_name VARCHAR(50) NOT NULL,
    allowance_type VARCHAR(20) DEFAULT 'machining',
    default_value DECIMAL(10,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='allowances' AND column_name='allowance_type')=0,
  'ALTER TABLE `allowances` ADD COLUMN `allowance_type` VARCHAR(20) DEFAULT ''machining'' AFTER allowance_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='allowances' AND column_name='default_value')=0,
  'ALTER TABLE `allowances` ADD COLUMN `default_value` DECIMAL(10,2) AFTER allowance_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='allowances' AND column_name='allowance_code')=0,
  'ALTER TABLE `allowances` ADD COLUMN `allowance_code` VARCHAR(30) AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='allowances' AND column_name='value')=0,
  'ALTER TABLE `allowances` ADD COLUMN `value` FLOAT DEFAULT 0 AFTER allowance_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Insert default allowances
INSERT IGNORE INTO allowances (allowance_name, allowance_type, default_value, is_active) VALUES
('Parting', 'cutting', 3.0, TRUE),
('Facing', 'machining', 2.0, TRUE),
('Machining Allowance', 'machining', 1.0, TRUE),
('Grinding Allowance', 'grinding', 0.5, TRUE),
('Chamfer Allowance', 'finishing', 0.5, TRUE);

-- =============================================
-- QUOTATIONS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS quotations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    quotation_number VARCHAR(30) NOT NULL UNIQUE,
    quotation_ref VARCHAR(30),
    quotation_version VARCHAR(10) DEFAULT 'V1',
    customer_id INT NOT NULL,
    quotation_date DATE NOT NULL,
    valid_until DATE,
    status VARCHAR(20) DEFAULT 'draft',
    priority VARCHAR(20) DEFAULT 'normal',
    currency VARCHAR(10) DEFAULT 'INR',
    exchange_rate DECIMAL(12,4) DEFAULT 1,
    subtotal DECIMAL(14,4) DEFAULT 0,
    tax_percent DECIMAL(5,2) DEFAULT 18,
    tax_amount DECIMAL(14,4) DEFAULT 0,
    total_amount DECIMAL(14,4) DEFAULT 0,
    margin_percent DECIMAL(6,2) DEFAULT 20,
    billing_address TEXT,
    shipping_address TEXT,
    payment_terms TEXT,
    delivery_terms TEXT,
    warranty VARCHAR(100),
    internal_notes TEXT,
    terms_conditions TEXT,
    created_by INT,
    sent_date DATETIME,
    won_date DATETIME,
    lost_date DATETIME,
    lost_reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='quotation_ref')=0,
  'ALTER TABLE `quotations` ADD COLUMN `quotation_ref` VARCHAR(30) AFTER quotation_number',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='quotation_version')=0,
  'ALTER TABLE `quotations` ADD COLUMN `quotation_version` VARCHAR(10) DEFAULT ''V1'' AFTER quotation_ref',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='exchange_rate')=0,
  'ALTER TABLE `quotations` ADD COLUMN `exchange_rate` DECIMAL(12,4) DEFAULT 1 AFTER currency',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='subtotal')=0,
  'ALTER TABLE `quotations` ADD COLUMN `subtotal` DECIMAL(14,4) DEFAULT 0 AFTER exchange_rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='tax_amount')=0,
  'ALTER TABLE `quotations` ADD COLUMN `tax_amount` DECIMAL(14,4) DEFAULT 0 AFTER subtotal',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='margin_percent')=0,
  'ALTER TABLE `quotations` ADD COLUMN `margin_percent` DECIMAL(6,2) DEFAULT 20 AFTER total_amount',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='delivery_terms')=0,
  'ALTER TABLE `quotations` ADD COLUMN `delivery_terms` TEXT AFTER payment_terms',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='internal_notes')=0,
  'ALTER TABLE `quotations` ADD COLUMN `internal_notes` TEXT AFTER warranty',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='created_by')=0,
  'ALTER TABLE `quotations` ADD COLUMN `created_by` INT AFTER terms_conditions',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotations' AND column_name='lost_reason')=0,
  'ALTER TABLE `quotations` ADD COLUMN `lost_reason` TEXT AFTER lost_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- QUOTATION ITEMS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS quotation_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    quotation_id INT NOT NULL,
    product_id INT,
    item_number INT,
    part_description VARCHAR(300),
    drawing_number VARCHAR(100),
    material VARCHAR(100),
    quantity_1 INT,
    quantity_2 INT,
    quantity_3 INT,
    quantity_4 INT,
    quantity_5 INT,
    unit_price_1 DECIMAL(12,4),
    unit_price_2 DECIMAL(12,4),
    unit_price_3 DECIMAL(12,4),
    unit_price_4 DECIMAL(12,4),
    unit_price_5 DECIMAL(12,4),
    tooling_cost DECIMAL(12,4) DEFAULT 0,
    tooling_amortization_qty INT DEFAULT 0,
    material_cost DECIMAL(12,4) DEFAULT 0,
    machining_cost DECIMAL(12,4) DEFAULT 0,
    vendor_cost DECIMAL(12,4) DEFAULT 0,
    other_cost DECIMAL(12,4) DEFAULT 0,
    overhead_amount DECIMAL(12,4) DEFAULT 0,
    profit_amount DECIMAL(12,4) DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0,
    margin_percent DECIMAL(6,2) DEFAULT 20,
    sequence INT DEFAULT 0,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='item_number')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `item_number` INT AFTER product_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='tooling_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `tooling_cost` DECIMAL(12,4) DEFAULT 0 AFTER quantity_5',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='tooling_amortization_qty')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `tooling_amortization_qty` INT DEFAULT 0 AFTER tooling_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='material_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `material_cost` DECIMAL(12,4) DEFAULT 0 AFTER tooling_amortization_qty',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='machining_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `machining_cost` DECIMAL(12,4) DEFAULT 0 AFTER material_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='vendor_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `vendor_cost` DECIMAL(12,4) DEFAULT 0 AFTER machining_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='other_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `other_cost` DECIMAL(12,4) DEFAULT 0 AFTER vendor_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='overhead_amount')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `overhead_amount` DECIMAL(12,4) DEFAULT 0 AFTER other_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='profit_amount')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `profit_amount` DECIMAL(12,4) DEFAULT 0 AFTER overhead_amount',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='total_cost')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `total_cost` DECIMAL(12,4) DEFAULT 0 AFTER profit_amount',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='margin_percent')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `margin_percent` DECIMAL(6,2) DEFAULT 20 AFTER total_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='sequence')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `sequence` INT DEFAULT 0 AFTER margin_percent',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_items' AND column_name='notes')=0,
  'ALTER TABLE `quotation_items` ADD COLUMN `notes` TEXT AFTER sequence',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- QUOTATION OPERATIONS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS quotation_operations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    quotation_id INT NOT NULL,
    quotation_item_id INT,
    process_id INT,
    operation_name VARCHAR(100),
    operation_type VARCHAR(20),
    sequence INT DEFAULT 0,
    machine_id INT,
    vendor_id INT,
    setup_time DECIMAL(10,2),
    cycle_time DECIMAL(10,2),
    cost_type VARCHAR(20),
    cost_value DECIMAL(12,4),
    calculated_cost FLOAT DEFAULT 0,
    quantity_for_calculation INT DEFAULT 1,
    cost_amount DECIMAL(12,4) DEFAULT 0,
    notes TEXT
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='quotation_item_id')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `quotation_item_id` INT AFTER quotation_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='process_id')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `process_id` INT AFTER quotation_item_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='operation_type')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `operation_type` VARCHAR(20) AFTER operation_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='calculated_cost')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `calculated_cost` FLOAT DEFAULT 0 AFTER cost_value',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='quantity_for_calculation')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `quantity_for_calculation` INT DEFAULT 1 AFTER calculated_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='quotation_operations' AND column_name='notes')=0,
  'ALTER TABLE `quotation_operations` ADD COLUMN `notes` TEXT',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- ATTACHMENTS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS attachments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entity_type VARCHAR(30) NOT NULL,
    entity_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    original_name VARCHAR(255),
    file_type VARCHAR(50),
    file_size INT,
    file_path VARCHAR(500) NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    uploaded_by INT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='entity_type')=0,
  'ALTER TABLE `attachments` ADD COLUMN `entity_type` VARCHAR(30) NOT NULL AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='original_name')=0,
  'ALTER TABLE `attachments` ADD COLUMN `original_name` VARCHAR(255) AFTER file_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='file_type')=0,
  'ALTER TABLE `attachments` ADD COLUMN `file_type` VARCHAR(50) AFTER original_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='file_path')=0,
  'ALTER TABLE `attachments` ADD COLUMN `file_path` VARCHAR(500) NOT NULL AFTER file_size',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='is_primary')=0,
  'ALTER TABLE `attachments` ADD COLUMN `is_primary` BOOLEAN DEFAULT FALSE AFTER file_path',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='uploaded_by')=0,
  'ALTER TABLE `attachments` ADD COLUMN `uploaded_by` INT AFTER is_primary',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='uploaded_at')=0,
  'ALTER TABLE `attachments` ADD COLUMN `uploaded_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER uploaded_by',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- REVISION HISTORY TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS revision_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entity_type VARCHAR(30) NOT NULL,
    entity_id INT NOT NULL,
    field_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    changed_by INT,
    changed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='revision_history' AND column_name='field_name')=0,
  'ALTER TABLE `revision_history` ADD COLUMN `field_name` VARCHAR(100) AFTER entity_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='revision_history' AND column_name='changed_by')=0,
  'ALTER TABLE `revision_history` ADD COLUMN `changed_by` INT AFTER new_value',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- CURRENCIES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS currencies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    currency_code VARCHAR(10) NOT NULL UNIQUE,
    currency_name VARCHAR(100) NOT NULL,
    exchange_rate DECIMAL(12,4) DEFAULT 1,
    symbol VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='currencies' AND column_name='symbol')=0,
  'ALTER TABLE `currencies` ADD COLUMN `symbol` VARCHAR(10) AFTER exchange_rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='currencies' AND column_name='is_base')=0,
  'ALTER TABLE `currencies` ADD COLUMN `is_base` BOOLEAN DEFAULT FALSE AFTER symbol',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Insert sample currencies
INSERT IGNORE INTO currencies (currency_code, currency_name, exchange_rate, symbol, is_active) VALUES
('INR', 'Indian Rupee', 1.0, '₹', TRUE),
('USD', 'US Dollar', 83.0, '$', TRUE),
('EUR', 'Euro', 90.0, '€', TRUE),
('GBP', 'British Pound', 105.0, '£', TRUE),
('AED', 'UAE Dirham', 22.0, 'د.إ', TRUE);

-- =============================================
-- SYSTEM SETTINGS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value TEXT,
    description VARCHAR(200),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =============================================
-- PRODUCT COST HISTORY TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS product_cost_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT,
    material_id INT,
    quotation_id INT,
    raw_material_cost DECIMAL(12,4) DEFAULT 0,
    machining_cost DECIMAL(12,4) DEFAULT 0,
    vendor_cost DECIMAL(12,4) DEFAULT 0,
    tooling_cost DECIMAL(12,4) DEFAULT 0,
    overhead_cost DECIMAL(12,4) DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0,
    selling_price DECIMAL(12,4) DEFAULT 0,
    margin_percent DECIMAL(8,2) DEFAULT 0,
    quantity INT DEFAULT 1,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    recorded_by INT
);

-- Add missing columns if they don't exist
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='material_id')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `material_id` INT AFTER product_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='quotation_id')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `quotation_id` INT AFTER material_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='vendor_cost')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `vendor_cost` DECIMAL(12,4) DEFAULT 0 AFTER machining_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='tooling_cost')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `tooling_cost` DECIMAL(12,4) DEFAULT 0 AFTER vendor_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='overhead_cost')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `overhead_cost` DECIMAL(12,4) DEFAULT 0 AFTER tooling_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='selling_price')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `selling_price` DECIMAL(12,4) DEFAULT 0 AFTER total_cost',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='margin_percent')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `margin_percent` DECIMAL(8,2) DEFAULT 0 AFTER selling_price',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='quantity')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `quantity` INT DEFAULT 1 AFTER margin_percent',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='product_cost_history' AND column_name='recorded_by')=0,
  'ALTER TABLE `product_cost_history` ADD COLUMN `recorded_by` INT AFTER recorded_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- RAW MATERIAL CALCULATIONS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS raw_material_calculations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT,
    shape_type VARCHAR(30) DEFAULT 'round_bar',
    finished_length DECIMAL(10,2),
    diameter DECIMAL(10,2),
    width DECIMAL(10,2),
    thickness DECIMAL(10,2),
    height DECIMAL(10,2),
    material_id INT,
    material_density DECIMAL(10,4),
    parting DECIMAL(10,2) DEFAULT 3,
    facing DECIMAL(10,2) DEFAULT 2,
    machining_allowance DECIMAL(10,2) DEFAULT 1,
    grinding_allowance DECIMAL(10,2) DEFAULT 0,
    chamfer_allowance DECIMAL(10,2) DEFAULT 0.5,
    blank_length DECIMAL(10,2),
    volume DECIMAL(12,4),
    weight DECIMAL(12,4),
    pieces_per_bar INT,
    utilization_percent DECIMAL(6,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- NOTIFICATIONS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    notification_type VARCHAR(50),
    reference_type VARCHAR(30),
    reference_id INT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='notifications' AND column_name='notification_type')=0,
  'ALTER TABLE `notifications` ADD COLUMN `notification_type` VARCHAR(50) AFTER message',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='notifications' AND column_name='reference_type')=0,
  'ALTER TABLE `notifications` ADD COLUMN `reference_type` VARCHAR(30) AFTER notification_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='notifications' AND column_name='reference_id')=0,
  'ALTER TABLE `notifications` ADD COLUMN `reference_id` INT AFTER reference_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='notifications' AND column_name='is_read')=0,
  'ALTER TABLE `notifications` ADD COLUMN `is_read` BOOLEAN DEFAULT FALSE AFTER reference_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- VENDOR WORK ORDERS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS vendor_work_orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    work_order_number VARCHAR(30) NOT NULL UNIQUE,
    vendor_id INT NOT NULL,
    process_id INT,
    product_id INT,
    quotation_id INT,
    title VARCHAR(200),
    part_description VARCHAR(300),
    drawing_number VARCHAR(100),
    service_name VARCHAR(100),
    notes TEXT,
    quantity_ordered INT DEFAULT 0,
    quantity_received INT DEFAULT 0,
    quantity_rejected INT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'pcs',
    rate DECIMAL(12,4) DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    order_date DATE,
    due_date DATE,
    completed_date DATE,
    status VARCHAR(20) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'normal',
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='work_order_number')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `work_order_number` VARCHAR(30) NOT NULL UNIQUE AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='vendor_id')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `vendor_id` INT NOT NULL AFTER work_order_number',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='process_id')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `process_id` INT AFTER vendor_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='product_id')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `product_id` INT AFTER process_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='quotation_id')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `quotation_id` INT AFTER product_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='title')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `title` VARCHAR(200) AFTER quotation_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='part_description')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `part_description` VARCHAR(300) AFTER title',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='drawing_number')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `drawing_number` VARCHAR(100) AFTER part_description',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='service_name')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `service_name` VARCHAR(100) AFTER drawing_number',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='notes')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `notes` TEXT AFTER service_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='quantity_ordered')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `quantity_ordered` INT DEFAULT 0 AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='quantity_received')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `quantity_received` INT DEFAULT 0 AFTER quantity_ordered',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='quantity_rejected')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `quantity_rejected` INT DEFAULT 0 AFTER quantity_received',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='unit')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `unit` VARCHAR(20) DEFAULT ''pcs'' AFTER quantity_rejected',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='rate')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `rate` DECIMAL(12,4) DEFAULT 0 AFTER unit',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='currency')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `currency` VARCHAR(10) DEFAULT ''INR'' AFTER rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='order_date')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `order_date` DATE AFTER currency',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='due_date')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `due_date` DATE AFTER order_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='completed_date')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `completed_date` DATE AFTER due_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='status')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `status` VARCHAR(20) DEFAULT ''pending'' AFTER completed_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='priority')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `priority` VARCHAR(20) DEFAULT ''normal'' AFTER status',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='created_by')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `created_by` INT AFTER priority',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='created_at')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER created_by',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_work_orders' AND column_name='updated_at')=0,
  'ALTER TABLE `vendor_work_orders` ADD COLUMN `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Indexes: see the note at the end of this file (MySQL has no
-- CREATE INDEX IF NOT EXISTS). The equivalent indexes are defined in schema.sql.

-- =============================================
-- VENDOR QUALITY ISSUES TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS vendor_quality_issues (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT NOT NULL,
    work_order_id INT,
    issue_date DATE NOT NULL,
    issue_type VARCHAR(30) DEFAULT 'rejection',
    severity VARCHAR(20) DEFAULT 'minor',
    description TEXT,
    quantity_affected INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open',
    resolution TEXT,
    resolved_date DATE,
    reported_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='vendor_id')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `vendor_id` INT NOT NULL AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='work_order_id')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `work_order_id` INT AFTER vendor_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='issue_date')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `issue_date` DATE NOT NULL AFTER work_order_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='issue_type')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `issue_type` VARCHAR(30) DEFAULT ''rejection'' AFTER issue_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='severity')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `severity` VARCHAR(20) DEFAULT ''minor'' AFTER issue_type',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='description')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `description` TEXT AFTER severity',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='quantity_affected')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `quantity_affected` INT DEFAULT 0 AFTER description',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='status')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `status` VARCHAR(20) DEFAULT ''open'' AFTER quantity_affected',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='resolution')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `resolution` TEXT AFTER status',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='resolved_date')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `resolved_date` DATE AFTER resolution',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='reported_by')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `reported_by` INT AFTER resolved_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='created_at')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER reported_by',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='vendor_quality_issues' AND column_name='updated_at')=0,
  'ALTER TABLE `vendor_quality_issues` ADD COLUMN `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- SUPPLIER / MASTER DATA TABLES (additive)
-- =============================================
CREATE TABLE IF NOT EXISTS supplier_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type_code VARCHAR(30) NOT NULL UNIQUE,
    type_name VARCHAR(100) NOT NULL,
    description TEXT,
    sort_order INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='type_code')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `type_code` VARCHAR(30) NOT NULL AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='type_name')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `type_name` VARCHAR(100) NOT NULL AFTER type_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='description')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `description` TEXT AFTER type_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='sort_order')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `sort_order` INT DEFAULT 0 AFTER description',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='is_active')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `is_active` BOOLEAN DEFAULT TRUE AFTER sort_order',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='created_at')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_active',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_types' AND column_name='updated_at')=0,
  'ALTER TABLE `supplier_types` ADD COLUMN `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

CREATE TABLE IF NOT EXISTS suppliers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_code VARCHAR(30) NOT NULL UNIQUE,
    company_name VARCHAR(200) NOT NULL,
    supplier_type_id INT,
    contact_person VARCHAR(120),
    email VARCHAR(120),
    phone VARCHAR(30),
    mobile VARCHAR(30),
    website VARCHAR(200),
    address_line1 VARCHAR(200),
    address_line2 VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100) DEFAULT 'India',
    gst_vat VARCHAR(30),
    pan VARCHAR(30),
    cin VARCHAR(30),
    tax_id VARCHAR(50),
    default_currency VARCHAR(10) DEFAULT 'INR',
    payment_terms VARCHAR(100),
    lead_time INT,
    document_references TEXT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='supplier_code')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `supplier_code` VARCHAR(30) NOT NULL AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='company_name')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `company_name` VARCHAR(200) NOT NULL AFTER supplier_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='supplier_type_id')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `supplier_type_id` INT AFTER company_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='contact_person')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `contact_person` VARCHAR(120) AFTER supplier_type_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='email')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `email` VARCHAR(120) AFTER contact_person',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='phone')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `phone` VARCHAR(30) AFTER email',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='mobile')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `mobile` VARCHAR(30) AFTER phone',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='website')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `website` VARCHAR(200) AFTER mobile',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='address_line1')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `address_line1` VARCHAR(200) AFTER website',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='address_line2')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `address_line2` VARCHAR(200) AFTER address_line1',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='city')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `city` VARCHAR(100) AFTER address_line2',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='state')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `state` VARCHAR(100) AFTER city',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='postal_code')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `postal_code` VARCHAR(20) AFTER state',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='country')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `country` VARCHAR(100) DEFAULT ''India'' AFTER postal_code',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='gst_vat')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `gst_vat` VARCHAR(30) AFTER country',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='pan')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `pan` VARCHAR(30) AFTER gst_vat',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='cin')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `cin` VARCHAR(30) AFTER pan',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='tax_id')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `tax_id` VARCHAR(50) AFTER cin',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='default_currency')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `default_currency` VARCHAR(10) DEFAULT ''INR'' AFTER tax_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='payment_terms')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `payment_terms` VARCHAR(100) AFTER default_currency',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='lead_time')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `lead_time` INT AFTER payment_terms',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='document_references')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `document_references` TEXT AFTER lead_time',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='notes')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `notes` TEXT AFTER document_references',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='is_active')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `is_active` BOOLEAN DEFAULT TRUE AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='created_at')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_active',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='suppliers' AND column_name='updated_at')=0,
  'ALTER TABLE `suppliers` ADD COLUMN `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

CREATE TABLE IF NOT EXISTS supplier_materials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_id INT NOT NULL,
    material_id INT,
    grade VARCHAR(50),
    rate FLOAT,
    rate_unit VARCHAR(20) DEFAULT '₹/kg',
    lead_time INT,
    notes TEXT,
    is_preferred BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='supplier_id')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `supplier_id` INT NOT NULL AFTER id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='material_id')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `material_id` INT AFTER supplier_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='grade')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `grade` VARCHAR(50) AFTER material_id',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='rate')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `rate` FLOAT AFTER grade',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='rate_unit')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `rate_unit` VARCHAR(20) DEFAULT ''₹/kg'' AFTER rate',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='lead_time')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `lead_time` INT AFTER rate_unit',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='notes')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `notes` TEXT AFTER lead_time',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='is_preferred')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `is_preferred` BOOLEAN DEFAULT FALSE AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='supplier_materials' AND column_name='created_at')=0,
  'ALTER TABLE `supplier_materials` ADD COLUMN `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_preferred',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- Link materials to the supplier master (optional, additive; existing text kept)
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='materials' AND column_name='supplier_id')=0,
  'ALTER TABLE `materials` ADD COLUMN `supplier_id` INT AFTER default_supplier',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;


-- =============================================
-- PROCUREMENT: PURCHASE ORDERS, RECEIPTS & BATCHES (additive)
-- =============================================
-- These are new tables created with IF NOT EXISTS, so running this migration
-- against an existing database adds them without touching any existing data.
CREATE TABLE IF NOT EXISTS supplier_purchase_orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    po_number VARCHAR(30) NOT NULL UNIQUE,
    supplier_id INT NOT NULL,
    order_date DATE,
    expected_date DATE,
    status VARCHAR(20) DEFAULT 'draft',
    currency VARCHAR(10) DEFAULT 'INR',
    payment_terms VARCHAR(100),
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    INDEX idx_spo_supplier (supplier_id),
    INDEX idx_spo_status (status)
);

CREATE TABLE IF NOT EXISTS supplier_po_lines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    po_id INT NOT NULL,
    material_id INT,
    description VARCHAR(300),
    grade VARCHAR(50),
    quantity_ordered FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'kg',
    rate FLOAT DEFAULT 0,
    rate_unit VARCHAR(20) DEFAULT '₹/kg',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (po_id) REFERENCES supplier_purchase_orders(id),
    FOREIGN KEY (material_id) REFERENCES materials(id),
    INDEX idx_spol_po (po_id)
);

CREATE TABLE IF NOT EXISTS supplier_order_receipts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    receipt_number VARCHAR(30) NOT NULL UNIQUE,
    po_id INT NOT NULL,
    supplier_id INT NOT NULL,
    receipt_date DATE NOT NULL,
    supplier_document VARCHAR(100),
    notes TEXT,
    received_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (po_id) REFERENCES supplier_purchase_orders(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    INDEX idx_sor_po (po_id),
    INDEX idx_sor_supplier (supplier_id)
);

CREATE TABLE IF NOT EXISTS supplier_receipt_lines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    receipt_id INT NOT NULL,
    po_line_id INT,
    material_id INT,
    quantity_received FLOAT DEFAULT 0,
    quantity_rejected FLOAT DEFAULT 0,
    batch_number VARCHAR(60),
    heat_number VARCHAR(60),
    lot_number VARCHAR(60),
    certificate_ref VARCHAR(120),
    inspection_status VARCHAR(20) DEFAULT 'pending',
    inspection_notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (receipt_id) REFERENCES supplier_order_receipts(id),
    FOREIGN KEY (po_line_id) REFERENCES supplier_po_lines(id),
    FOREIGN KEY (material_id) REFERENCES materials(id),
    INDEX idx_srl_receipt (receipt_id)
);

-- =============================================
-- PAYABLES: SUPPLIER INVOICES, PAYMENTS & ALLOCATIONS (additive)
-- =============================================
CREATE TABLE IF NOT EXISTS supplier_invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_number VARCHAR(60) NOT NULL,
    supplier_id INT NOT NULL,
    po_id INT,
    invoice_date DATE NOT NULL,
    due_date DATE,
    subtotal FLOAT DEFAULT 0,
    tax_amount FLOAT DEFAULT 0,
    total_amount FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    status VARCHAR(20) DEFAULT 'posted',
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_supplier_invoice_number (supplier_id, invoice_number),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    FOREIGN KEY (po_id) REFERENCES supplier_purchase_orders(id),
    INDEX idx_sinv_supplier (supplier_id),
    INDEX idx_sinv_status (status)
);

CREATE TABLE IF NOT EXISTS supplier_payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_number VARCHAR(30) NOT NULL UNIQUE,
    supplier_id INT NOT NULL,
    payment_date DATE NOT NULL,
    amount FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    method VARCHAR(20),
    reference VARCHAR(100),
    is_advance BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    INDEX idx_spay_supplier (supplier_id)
);

CREATE TABLE IF NOT EXISTS supplier_payment_allocations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_id INT NOT NULL,
    invoice_id INT NOT NULL,
    amount FLOAT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_supplier_alloc_payment_invoice (payment_id, invoice_id),
    FOREIGN KEY (payment_id) REFERENCES supplier_payments(id),
    FOREIGN KEY (invoice_id) REFERENCES supplier_invoices(id),
    INDEX idx_spa_payment (payment_id),
    INDEX idx_spa_invoice (invoice_id)
);

-- =============================================
-- PAYABLES: VENDOR (JOB-WORK) INVOICES, PAYMENTS & ALLOCATIONS (additive)
-- =============================================
CREATE TABLE IF NOT EXISTS vendor_invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_number VARCHAR(60) NOT NULL,
    vendor_id INT NOT NULL,
    work_order_id INT,
    invoice_date DATE NOT NULL,
    due_date DATE,
    subtotal FLOAT DEFAULT 0,
    tax_amount FLOAT DEFAULT 0,
    total_amount FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    status VARCHAR(20) DEFAULT 'posted',
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_vendor_invoice_number (vendor_id, invoice_number),
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    FOREIGN KEY (work_order_id) REFERENCES vendor_work_orders(id),
    INDEX idx_vinv_vendor (vendor_id),
    INDEX idx_vinv_status (status)
);

CREATE TABLE IF NOT EXISTS vendor_payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_number VARCHAR(30) NOT NULL UNIQUE,
    vendor_id INT NOT NULL,
    payment_date DATE NOT NULL,
    amount FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    method VARCHAR(20),
    reference VARCHAR(100),
    is_advance BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    INDEX idx_vpay_vendor (vendor_id)
);

CREATE TABLE IF NOT EXISTS vendor_payment_allocations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_id INT NOT NULL,
    invoice_id INT NOT NULL,
    amount FLOAT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_vendor_alloc_payment_invoice (payment_id, invoice_id),
    FOREIGN KEY (payment_id) REFERENCES vendor_payments(id),
    FOREIGN KEY (invoice_id) REFERENCES vendor_invoices(id),
    INDEX idx_vpa_payment (payment_id),
    INDEX idx_vpa_invoice (invoice_id)
);



-- =============================================
-- ORDERS & MANUFACTURING TRACKING
-- =============================================
-- A confirmed customer order anchors the production lifecycle:
--   Order -> OrderOperation (order-specific route, sequence ordered)
--            -> OperationTransaction (issue/return/accept/reject/rework/scrap)
--   Order -> MaterialLot -> MaterialMovement (receipt/allocation/issue/return/scrap/shortage)
--   Order -> Inspection (incoming / in_process / final)
--   Order -> PackagingRecord (packed + warehouse receipt -> completion)
-- Quantities are derived from the ledger rows, and UNIQUE keys stop duplicate
-- movements being posted twice.

CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_number VARCHAR(30) NOT NULL UNIQUE,
    customer_id INT NOT NULL,
    quotation_id INT,
    quotation_item_id INT,
    product_id INT,
    part_description VARCHAR(300),
    drawing_number VARCHAR(100),
    revision VARCHAR(10),
    specifications TEXT,
    material_grade VARCHAR(100),
    standard VARCHAR(50),
    quantity_ordered INT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'pcs',
    rate FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    order_date DATE NOT NULL,
    due_date DATE,
    planned_start DATE,
    actual_start DATE,
    actual_completion DATE,
    status VARCHAR(20) DEFAULT 'confirmed',
    priority VARCHAR(20) DEFAULT 'normal',
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (quotation_id) REFERENCES quotations(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_order_number (order_number),
    INDEX idx_order_customer (customer_id),
    INDEX idx_order_status (status),
    INDEX idx_order_drawing (drawing_number)
);

CREATE TABLE IF NOT EXISTS order_operations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    sequence INT NOT NULL DEFAULT 1,
    process_id INT,
    operation_name VARCHAR(100) NOT NULL,
    is_vendor BOOLEAN DEFAULT FALSE,
    vendor_id INT,
    machine_id INT,
    quantity_planned INT DEFAULT 0,
    planned_issue_date DATE,
    due_date DATE,
    rate FLOAT DEFAULT 0,
    actual_start DATE,
    actual_end DATE,
    status VARCHAR(20) DEFAULT 'pending',
    is_final BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (process_id) REFERENCES processes(id),
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    FOREIGN KEY (machine_id) REFERENCES machines(id),
    INDEX idx_op_order (order_id),
    INDEX idx_op_vendor (vendor_id),
    INDEX idx_op_status (status)
);

CREATE TABLE IF NOT EXISTS operation_transactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_operation_id INT NOT NULL,
    transaction_type VARCHAR(20) NOT NULL,
    transaction_date DATE NOT NULL,
    quantity INT DEFAULT 0,
    reference VARCHAR(100),
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_op_txn_type_reference (order_operation_id, transaction_type, reference),
    FOREIGN KEY (order_operation_id) REFERENCES order_operations(id),
    INDEX idx_op_txn_op (order_operation_id),
    INDEX idx_op_txn_type (transaction_type)
);

CREATE TABLE IF NOT EXISTS material_lots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    lot_number VARCHAR(60) NOT NULL,
    material_id INT,
    order_id INT,
    supplier_id INT,
    po_id INT,
    receipt_id INT,
    receipt_line_id INT,
    grade VARCHAR(100),
    specification VARCHAR(200),
    heat_number VARCHAR(60),
    batch_number VARCHAR(60),
    certificate_ref VARCHAR(120),
    quantity_received FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'kg',
    receipt_date DATE,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_lot_receipt_line (receipt_line_id),
    FOREIGN KEY (material_id) REFERENCES materials(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    FOREIGN KEY (po_id) REFERENCES supplier_purchase_orders(id),
    FOREIGN KEY (receipt_id) REFERENCES supplier_order_receipts(id),
    FOREIGN KEY (receipt_line_id) REFERENCES supplier_receipt_lines(id),
    INDEX idx_lot_number (lot_number),
    INDEX idx_lot_material (material_id),
    INDEX idx_lot_order (order_id)
);

CREATE TABLE IF NOT EXISTS material_movements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    lot_id INT NOT NULL,
    movement_type VARCHAR(20) NOT NULL,
    movement_date DATE NOT NULL,
    quantity FLOAT DEFAULT 0,
    order_id INT,
    order_operation_id INT,
    reference VARCHAR(100),
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_movement_type_reference (lot_id, movement_type, reference),
    FOREIGN KEY (lot_id) REFERENCES material_lots(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (order_operation_id) REFERENCES order_operations(id),
    INDEX idx_mv_lot (lot_id),
    INDEX idx_mv_type (movement_type),
    INDEX idx_mv_order (order_id)
);

CREATE TABLE IF NOT EXISTS inspections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    inspection_number VARCHAR(30) NOT NULL UNIQUE,
    order_id INT NOT NULL,
    order_operation_id INT,
    lot_id INT,
    inspection_type VARCHAR(20) DEFAULT 'final',
    inspection_date DATE NOT NULL,
    acceptance_criteria TEXT,
    tolerance VARCHAR(100),
    measurements TEXT,
    sample_size INT DEFAULT 0,
    quantity_inspected INT DEFAULT 0,
    quantity_accepted INT DEFAULT 0,
    quantity_rejected INT DEFAULT 0,
    quantity_rework INT DEFAULT 0,
    result VARCHAR(20) DEFAULT 'pending',
    notes TEXT,
    inspected_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (order_operation_id) REFERENCES order_operations(id),
    FOREIGN KEY (lot_id) REFERENCES material_lots(id),
    INDEX idx_insp_number (inspection_number),
    INDEX idx_insp_order (order_id),
    INDEX idx_insp_type (inspection_type),
    INDEX idx_insp_result (result)
);

CREATE TABLE IF NOT EXISTS packaging_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    packaging_date DATE NOT NULL,
    packaging_type VARCHAR(50),
    quantity_packed INT DEFAULT 0,
    quantity_accepted INT DEFAULT 0,
    quantity_rejected INT DEFAULT 0,
    label_ref VARCHAR(100),
    warehouse_receipt_number VARCHAR(50),
    warehouse_receipt_date DATE,
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'packed',
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    INDEX idx_pack_order (order_id),
    INDEX idx_pack_status (status)
);

-- =============================================
-- INVENTORY: stock locations, transfers, adjustments
-- =============================================
CREATE TABLE IF NOT EXISTS stock_locations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(30) NOT NULL UNIQUE,
    name VARCHAR(120) NOT NULL,
    kind VARCHAR(30) DEFAULT 'raw_material',
    vendor_id INT,
    address VARCHAR(300),
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    INDEX idx_loc_code (code),
    INDEX idx_loc_kind (kind)
);

CREATE TABLE IF NOT EXISTS stock_transfers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transfer_number VARCHAR(30) NOT NULL UNIQUE,
    from_location_id INT,
    to_location_id INT,
    transfer_date DATE NOT NULL,
    quantity FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'pcs',
    material_lot_id INT,
    order_id INT,
    order_operation_id INT,
    status VARCHAR(20) DEFAULT 'received',
    reference VARCHAR(100),
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_location_id) REFERENCES stock_locations(id),
    FOREIGN KEY (to_location_id) REFERENCES stock_locations(id),
    FOREIGN KEY (material_lot_id) REFERENCES material_lots(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (order_operation_id) REFERENCES order_operations(id),
    INDEX idx_trf_number (transfer_number),
    INDEX idx_trf_date (transfer_date),
    INDEX idx_trf_status (status)
);

CREATE TABLE IF NOT EXISTS stock_adjustments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    adjustment_number VARCHAR(30) NOT NULL UNIQUE,
    adjustment_date DATE NOT NULL,
    material_lot_id INT,
    location_id INT,
    order_id INT,
    adjustment_type VARCHAR(20) DEFAULT 'correction',
    quantity FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'kg',
    reverses_movement_id INT,
    reverses_transfer_id INT,
    reason VARCHAR(300),
    authorized_by INT,
    reference VARCHAR(100),
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_lot_id) REFERENCES material_lots(id),
    FOREIGN KEY (location_id) REFERENCES stock_locations(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (reverses_movement_id) REFERENCES material_movements(id),
    FOREIGN KEY (reverses_transfer_id) REFERENCES stock_transfers(id),
    INDEX idx_adj_number (adjustment_number),
    INDEX idx_adj_date (adjustment_date),
    INDEX idx_adj_type (adjustment_type)
);

-- =============================================
-- COSTING: actual cost entries per order
-- =============================================
CREATE TABLE IF NOT EXISTS order_cost_entries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    category VARCHAR(30) NOT NULL,
    amount FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    amount_base FLOAT DEFAULT 0,
    entry_date DATE,
    description VARCHAR(300),
    source_type VARCHAR(30) DEFAULT 'manual',
    source_id INT,
    order_operation_id INT,
    vendor_id INT,
    supplier_id INT,
    material_lot_id INT,
    is_estimate BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (order_operation_id) REFERENCES order_operations(id),
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    FOREIGN KEY (material_lot_id) REFERENCES material_lots(id),
    UNIQUE KEY uq_cost_source_category (source_type, source_id, category),
    INDEX idx_cost_order (order_id),
    INDEX idx_cost_category (category)
);

-- =============================================
-- PROCUREMENT: supplier RFQs, quotes and comparison
-- =============================================
CREATE TABLE IF NOT EXISTS supplier_rfqs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rfq_number VARCHAR(30) NOT NULL UNIQUE,
    title VARCHAR(200),
    material_id INT,
    order_id INT,
    quantity FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'kg',
    grade VARCHAR(50),
    rfq_date DATE NOT NULL,
    due_date DATE,
    required_date DATE,
    status VARCHAR(20) DEFAULT 'draft',
    delivery_terms VARCHAR(200),
    payment_terms VARCHAR(100),
    currency VARCHAR(10) DEFAULT 'INR',
    notes TEXT,
    awarded_quote_id INT,
    awarded_at DATETIME,
    awarded_by INT,
    po_id INT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES materials(id),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (po_id) REFERENCES supplier_purchase_orders(id),
    INDEX idx_srfq_number (rfq_number),
    INDEX idx_srfq_date (rfq_date),
    INDEX idx_srfq_status (status)
);

CREATE TABLE IF NOT EXISTS supplier_quotes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_rfq_id INT NOT NULL,
    supplier_id INT NOT NULL,
    quote_reference VARCHAR(60),
    quote_date DATE NOT NULL,
    valid_until DATE,
    quantity FLOAT DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'kg',
    unit_price FLOAT DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'INR',
    tax_percent FLOAT DEFAULT 0,
    lead_time_days INT,
    payment_terms VARCHAR(100),
    delivery_terms VARCHAR(200),
    quality_rating FLOAT,
    status VARCHAR(20) DEFAULT 'received',
    notes TEXT,
    comparison_score FLOAT,
    comparison_rank INT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_rfq_id) REFERENCES supplier_rfqs(id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    INDEX idx_sq_rfq (supplier_rfq_id),
    INDEX idx_sq_supplier (supplier_id),
    INDEX idx_sq_status (status)
);

-- =============================================
-- ADD INDEXES IF NOT EXISTS
-- =============================================
-- Note: MySQL doesn't support ADD INDEX IF NOT EXISTS, so we skip this
-- You can manually add indexes if needed

SELECT 'Migration completed successfully!' AS status;


-- =============================================
-- DOCUMENTS & TRACEABILITY: extra attachment columns
-- =============================================
-- The attachments table already stored files against an entity; these columns
-- add document control (category, revision, current/superseded) and richer
-- metadata. Superseded revisions are kept so history is preserved.
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='title')=0,
  'ALTER TABLE `attachments` ADD COLUMN `title` VARCHAR(200) AFTER original_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='file_type')=0,
  'ALTER TABLE `attachments` ADD COLUMN `file_type` VARCHAR(50) AFTER file_name',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='is_primary')=0,
  'ALTER TABLE `attachments` ADD COLUMN `is_primary` BOOLEAN DEFAULT FALSE AFTER file_size',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='category')=0,
  'ALTER TABLE `attachments` ADD COLUMN `category` VARCHAR(40) DEFAULT ''other'' AFTER file_size',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='revision')=0,
  'ALTER TABLE `attachments` ADD COLUMN `revision` VARCHAR(20) AFTER category',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='issue_date')=0,
  'ALTER TABLE `attachments` ADD COLUMN `issue_date` DATE AFTER revision',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='notes')=0,
  'ALTER TABLE `attachments` ADD COLUMN `notes` TEXT AFTER issue_date',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='is_current')=0,
  'ALTER TABLE `attachments` ADD COLUMN `is_current` BOOLEAN DEFAULT TRUE AFTER notes',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='supersedes_id')=0,
  'ALTER TABLE `attachments` ADD COLUMN `supersedes_id` INT AFTER is_current',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;
SET @npl_sql := IF((SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema=DATABASE() AND table_name='attachments' AND column_name='uploaded_at')=0,
  'ALTER TABLE `attachments` ADD COLUMN `uploaded_at` DATETIME DEFAULT CURRENT_TIMESTAMP AFTER uploaded_by',
  'DO 0');
PREPARE npl_stmt FROM @npl_sql; EXECUTE npl_stmt; DEALLOCATE PREPARE npl_stmt;

-- =============================================
-- DISPATCH: shipments of finished goods to customers
-- =============================================
CREATE TABLE IF NOT EXISTS dispatches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dispatch_number VARCHAR(30) NOT NULL UNIQUE,
    order_id INT NOT NULL,
    customer_id INT,
    dispatch_date DATE NOT NULL,
    mode VARCHAR(20) DEFAULT 'surface',
    transporter VARCHAR(150),
    vehicle_number VARCHAR(50),
    shipment_reference VARCHAR(100),
    eway_bill_number VARCHAR(50),
    invoice_number VARCHAR(60),
    invoice_date DATE,
    packing_list_ref VARCHAR(60),
    destination VARCHAR(200),
    status VARCHAR(20) DEFAULT 'prepared',
    delivery_status VARCHAR(20) DEFAULT 'pending',
    delivered_date DATE,
    pod_reference VARCHAR(100),
    receiver_name VARCHAR(120),
    remarks TEXT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (created_by) REFERENCES users(id),
    INDEX idx_dispatch_order (order_id),
    INDEX idx_dispatch_customer (customer_id),
    INDEX idx_dispatch_date (dispatch_date),
    INDEX idx_dispatch_status (status)
);

CREATE TABLE IF NOT EXISTS dispatch_lines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dispatch_id INT NOT NULL,
    packaging_record_id INT,
    lot_id INT,
    product_id INT,
    quantity INT DEFAULT 0,
    quantity_rejected INT DEFAULT 0,
    label_ref VARCHAR(100),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dispatch_id) REFERENCES dispatches(id),
    FOREIGN KEY (packaging_record_id) REFERENCES packaging_records(id),
    FOREIGN KEY (lot_id) REFERENCES material_lots(id),
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_dl_dispatch (dispatch_id),
    INDEX idx_dl_packaging (packaging_record_id)
);

-- =============================================
-- ALERTS: per-user acknowledgement of derived alerts
-- =============================================
CREATE TABLE IF NOT EXISTS alert_acknowledgements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    alert_key VARCHAR(200) NOT NULL,
    user_id INT NOT NULL,
    acknowledged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    note VARCHAR(300),
    UNIQUE KEY uq_alert_ack_user (alert_key, user_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_alert_key (alert_key),
    INDEX idx_alert_user (user_id)
);
