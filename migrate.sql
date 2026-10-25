-- NPL Fasteners ERP - Safe Migration Script
-- Only creates/alters if something is missing
-- Run in phpMyAdmin SQL tab

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
ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(64) NOT NULL UNIQUE AFTER id;
ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(120) NOT NULL UNIQUE AFTER username;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255) NOT NULL AFTER email;
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(120) AFTER password_hash;
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'read_only' AFTER full_name;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE AFTER role;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login DATETIME AFTER updated_at;

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_email ON users(email);

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

ALTER TABLE customers ADD COLUMN IF NOT EXISTS customer_code VARCHAR(20) NOT NULL UNIQUE AFTER id;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS company_name VARCHAR(200) NOT NULL AFTER customer_code;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS mobile VARCHAR(30) AFTER phone;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS pan VARCHAR(20) AFTER gst_vat;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS payment_terms VARCHAR(100) AFTER currency;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS incoterms VARCHAR(50) AFTER payment_terms;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS shipping_terms VARCHAR(200) AFTER incoterms;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS delivery_address TEXT AFTER shipping_address;

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

ALTER TABLE rfqs ADD COLUMN IF NOT EXISTS rfq_number VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE rfqs ADD COLUMN IF NOT EXISTS conversion_date DATETIME AFTER due_date;
ALTER TABLE rfqs ADD COLUMN IF NOT EXISTS created_by INT AFTER notes;

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

ALTER TABLE rfq_items ADD COLUMN IF NOT EXISTS sequence INT DEFAULT 0 AFTER notes;

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
ALTER TABLE products CHANGE COLUMN IF EXISTS product_code internal_code VARCHAR(30) NOT NULL UNIQUE;
ALTER TABLE products ADD COLUMN IF NOT EXISTS internal_code VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE products ADD COLUMN IF NOT EXISTS weight FLOAT AFTER material;
ALTER TABLE products ADD COLUMN IF NOT EXISTS standard VARCHAR(50) AFTER weight;
ALTER TABLE products ADD COLUMN IF NOT EXISTS standard_number VARCHAR(30) AFTER standard;
ALTER TABLE products ADD COLUMN IF NOT EXISTS dimensions TEXT AFTER thread;

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

ALTER TABLE materials ADD COLUMN IF NOT EXISTS material_code VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS material_name VARCHAR(100) NOT NULL AFTER material_code;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS density FLOAT NOT NULL AFTER material_name;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS grade VARCHAR(50) AFTER density;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS default_supplier VARCHAR(100) AFTER hsn_code;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS rate_unit VARCHAR(20) DEFAULT 'per_kg' AFTER default_rate;

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

ALTER TABLE material_rates ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT TRUE AFTER effective_date;
ALTER TABLE material_rates ADD COLUMN IF NOT EXISTS created_by INT AFTER is_current;

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

ALTER TABLE machines ADD COLUMN IF NOT EXISTS machine_code VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE machines ADD COLUMN IF NOT EXISTS machine_type VARCHAR(50) AFTER machine_name;
ALTER TABLE machines ADD COLUMN IF NOT EXISTS operator_name VARCHAR(100) AFTER hourly_rate;
ALTER TABLE machines ADD COLUMN IF NOT EXISTS power_consumption FLOAT AFTER operator_name;
ALTER TABLE machines ADD COLUMN IF NOT EXISTS efficiency FLOAT DEFAULT 100.0 AFTER power_consumption;
ALTER TABLE machines ADD COLUMN IF NOT EXISTS default_setup_time FLOAT DEFAULT 0 AFTER department;

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

ALTER TABLE vendors ADD COLUMN IF NOT EXISTS vendor_code VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE vendors ADD COLUMN IF NOT EXISTS mobile VARCHAR(30) AFTER phone;
ALTER TABLE vendors ADD COLUMN IF NOT EXISTS country VARCHAR(100) DEFAULT 'India' AFTER address;
ALTER TABLE vendors ADD COLUMN IF NOT EXISTS payment_terms VARCHAR(100) AFTER default_currency;
ALTER TABLE vendors ADD COLUMN IF NOT EXISTS lead_time INT AFTER payment_terms;

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

ALTER TABLE vendor_rates ADD COLUMN IF NOT EXISTS service_name VARCHAR(100) NOT NULL AFTER vendor_id;
ALTER TABLE vendor_rates ADD COLUMN IF NOT EXISTS min_quantity INT DEFAULT 1 AFTER unit;
ALTER TABLE vendor_rates ADD COLUMN IF NOT EXISTS lead_time_days INT AFTER min_quantity;
ALTER TABLE vendor_rates ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT TRUE AFTER effective_date;
ALTER TABLE vendor_rates ADD COLUMN IF NOT EXISTS created_by INT AFTER notes;

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

ALTER TABLE processes ADD COLUMN IF NOT EXISTS process_code VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS process_type VARCHAR(20) DEFAULT 'internal' AFTER process_name;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS department VARCHAR(50) AFTER vendor_id;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS setup_time FLOAT DEFAULT 0 AFTER department;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS cycle_time FLOAT DEFAULT 0 AFTER setup_time;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS cost_type VARCHAR(30) DEFAULT 'cycle_time' AFTER cycle_time;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS cost_value FLOAT DEFAULT 0 AFTER cost_type;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS cost_formula VARCHAR(200) AFTER cost_value;
ALTER TABLE processes ADD COLUMN IF NOT EXISTS is_custom BOOLEAN DEFAULT FALSE AFTER is_active;

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

ALTER TABLE process_templates ADD COLUMN IF NOT EXISTS template_code VARCHAR(30) NOT NULL UNIQUE AFTER id;

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
ALTER TABLE process_template_steps ADD COLUMN IF NOT EXISTS sequence INT NOT NULL DEFAULT 1;

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

ALTER TABLE allowances ADD COLUMN IF NOT EXISTS allowance_type VARCHAR(20) DEFAULT 'machining' AFTER allowance_name;
ALTER TABLE allowances ADD COLUMN IF NOT EXISTS default_value DECIMAL(10,2) AFTER allowance_type;
ALTER TABLE allowances ADD COLUMN IF NOT EXISTS allowance_code VARCHAR(30) AFTER id;
ALTER TABLE allowances ADD COLUMN IF NOT EXISTS value FLOAT DEFAULT 0 AFTER allowance_type;

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

ALTER TABLE quotations ADD COLUMN IF NOT EXISTS quotation_ref VARCHAR(30) AFTER quotation_number;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS quotation_version VARCHAR(10) DEFAULT 'V1' AFTER quotation_ref;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(12,4) DEFAULT 1 AFTER currency;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS subtotal DECIMAL(14,4) DEFAULT 0 AFTER exchange_rate;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS tax_amount DECIMAL(14,4) DEFAULT 0 AFTER subtotal;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS margin_percent DECIMAL(6,2) DEFAULT 20 AFTER total_amount;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS delivery_terms TEXT AFTER payment_terms;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS internal_notes TEXT AFTER warranty;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS created_by INT AFTER terms_conditions;
ALTER TABLE quotations ADD COLUMN IF NOT EXISTS lost_reason TEXT AFTER lost_date;

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

ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS item_number INT AFTER product_id;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS tooling_cost DECIMAL(12,4) DEFAULT 0 AFTER quantity_5;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS tooling_amortization_qty INT DEFAULT 0 AFTER tooling_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS material_cost DECIMAL(12,4) DEFAULT 0 AFTER tooling_amortization_qty;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS machining_cost DECIMAL(12,4) DEFAULT 0 AFTER material_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS vendor_cost DECIMAL(12,4) DEFAULT 0 AFTER machining_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS other_cost DECIMAL(12,4) DEFAULT 0 AFTER vendor_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS overhead_amount DECIMAL(12,4) DEFAULT 0 AFTER other_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS profit_amount DECIMAL(12,4) DEFAULT 0 AFTER overhead_amount;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS total_cost DECIMAL(12,4) DEFAULT 0 AFTER profit_amount;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS margin_percent DECIMAL(6,2) DEFAULT 20 AFTER total_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS sequence INT DEFAULT 0 AFTER margin_percent;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS notes TEXT AFTER sequence;

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

ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS quotation_item_id INT AFTER quotation_id;
ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS process_id INT AFTER quotation_item_id;
ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS operation_type VARCHAR(20) AFTER operation_name;
ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS calculated_cost FLOAT DEFAULT 0 AFTER cost_value;
ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS quantity_for_calculation INT DEFAULT 1 AFTER calculated_cost;
ALTER TABLE quotation_operations ADD COLUMN IF NOT EXISTS notes TEXT;

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

ALTER TABLE attachments ADD COLUMN IF NOT EXISTS entity_type VARCHAR(30) NOT NULL AFTER id;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS original_name VARCHAR(255) AFTER file_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_type VARCHAR(50) AFTER original_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) NOT NULL AFTER file_size;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT FALSE AFTER file_path;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS uploaded_by INT AFTER is_primary;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER uploaded_by;

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

ALTER TABLE revision_history ADD COLUMN IF NOT EXISTS field_name VARCHAR(100) AFTER entity_id;
ALTER TABLE revision_history ADD COLUMN IF NOT EXISTS changed_by INT AFTER new_value;

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

ALTER TABLE currencies ADD COLUMN IF NOT EXISTS symbol VARCHAR(10) AFTER exchange_rate;
ALTER TABLE currencies ADD COLUMN IF NOT EXISTS is_base BOOLEAN DEFAULT FALSE AFTER symbol;

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
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS material_id INT AFTER product_id;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS quotation_id INT AFTER material_id;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS vendor_cost DECIMAL(12,4) DEFAULT 0 AFTER machining_cost;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS tooling_cost DECIMAL(12,4) DEFAULT 0 AFTER vendor_cost;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS overhead_cost DECIMAL(12,4) DEFAULT 0 AFTER tooling_cost;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS selling_price DECIMAL(12,4) DEFAULT 0 AFTER total_cost;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS margin_percent DECIMAL(8,2) DEFAULT 0 AFTER selling_price;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS quantity INT DEFAULT 1 AFTER margin_percent;
ALTER TABLE product_cost_history ADD COLUMN IF NOT EXISTS recorded_by INT AFTER recorded_at;

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

ALTER TABLE notifications ADD COLUMN IF NOT EXISTS notification_type VARCHAR(50) AFTER message;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS reference_type VARCHAR(30) AFTER notification_type;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS reference_id INT AFTER reference_type;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS is_read BOOLEAN DEFAULT FALSE AFTER reference_id;

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

ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS work_order_number VARCHAR(30) NOT NULL UNIQUE AFTER id;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS vendor_id INT NOT NULL AFTER work_order_number;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS process_id INT AFTER vendor_id;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS product_id INT AFTER process_id;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS quotation_id INT AFTER product_id;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS title VARCHAR(200) AFTER quotation_id;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS part_description VARCHAR(300) AFTER title;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS drawing_number VARCHAR(100) AFTER part_description;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS service_name VARCHAR(100) AFTER drawing_number;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS notes TEXT AFTER service_name;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS quantity_ordered INT DEFAULT 0 AFTER notes;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS quantity_received INT DEFAULT 0 AFTER quantity_ordered;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS quantity_rejected INT DEFAULT 0 AFTER quantity_received;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS unit VARCHAR(20) DEFAULT 'pcs' AFTER quantity_rejected;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS rate DECIMAL(12,4) DEFAULT 0 AFTER unit;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS currency VARCHAR(10) DEFAULT 'INR' AFTER rate;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS order_date DATE AFTER currency;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS due_date DATE AFTER order_date;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS completed_date DATE AFTER due_date;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' AFTER completed_date;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal' AFTER status;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS created_by INT AFTER priority;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER created_by;
ALTER TABLE vendor_work_orders ADD COLUMN IF NOT EXISTS updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at;

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

ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS vendor_id INT NOT NULL AFTER id;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS work_order_id INT AFTER vendor_id;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS issue_date DATE NOT NULL AFTER work_order_id;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS issue_type VARCHAR(30) DEFAULT 'rejection' AFTER issue_date;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS severity VARCHAR(20) DEFAULT 'minor' AFTER issue_type;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS description TEXT AFTER severity;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS quantity_affected INT DEFAULT 0 AFTER description;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'open' AFTER quantity_affected;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS resolution TEXT AFTER status;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS resolved_date DATE AFTER resolution;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS reported_by INT AFTER resolved_date;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER reported_by;
ALTER TABLE vendor_quality_issues ADD COLUMN IF NOT EXISTS updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at;

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

ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS type_code VARCHAR(30) NOT NULL AFTER id;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS type_name VARCHAR(100) NOT NULL AFTER type_code;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS description TEXT AFTER type_name;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS sort_order INT DEFAULT 0 AFTER description;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE AFTER sort_order;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_active;
ALTER TABLE supplier_types ADD COLUMN IF NOT EXISTS updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at;

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

ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS supplier_code VARCHAR(30) NOT NULL AFTER id;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS company_name VARCHAR(200) NOT NULL AFTER supplier_code;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS supplier_type_id INT AFTER company_name;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS contact_person VARCHAR(120) AFTER supplier_type_id;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS email VARCHAR(120) AFTER contact_person;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS phone VARCHAR(30) AFTER email;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS mobile VARCHAR(30) AFTER phone;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS website VARCHAR(200) AFTER mobile;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS address_line1 VARCHAR(200) AFTER website;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS address_line2 VARCHAR(200) AFTER address_line1;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS city VARCHAR(100) AFTER address_line2;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS state VARCHAR(100) AFTER city;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS postal_code VARCHAR(20) AFTER state;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS country VARCHAR(100) DEFAULT 'India' AFTER postal_code;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS gst_vat VARCHAR(30) AFTER country;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS pan VARCHAR(30) AFTER gst_vat;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS cin VARCHAR(30) AFTER pan;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS tax_id VARCHAR(50) AFTER cin;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS default_currency VARCHAR(10) DEFAULT 'INR' AFTER tax_id;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS payment_terms VARCHAR(100) AFTER default_currency;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS lead_time INT AFTER payment_terms;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS document_references TEXT AFTER lead_time;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS notes TEXT AFTER document_references;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE AFTER notes;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_active;
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at;

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

ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS supplier_id INT NOT NULL AFTER id;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS material_id INT AFTER supplier_id;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS grade VARCHAR(50) AFTER material_id;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS rate FLOAT AFTER grade;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS rate_unit VARCHAR(20) DEFAULT '₹/kg' AFTER rate;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS lead_time INT AFTER rate_unit;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS notes TEXT AFTER lead_time;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS is_preferred BOOLEAN DEFAULT FALSE AFTER notes;
ALTER TABLE supplier_materials ADD COLUMN IF NOT EXISTS created_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER is_preferred;

-- Link materials to the supplier master (optional, additive; existing text kept)
ALTER TABLE materials ADD COLUMN IF NOT EXISTS supplier_id INT AFTER default_supplier;


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
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS title VARCHAR(200) AFTER original_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_type VARCHAR(50) AFTER file_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT FALSE AFTER file_size;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS category VARCHAR(40) DEFAULT 'other' AFTER file_size;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS revision VARCHAR(20) AFTER category;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS issue_date DATE AFTER revision;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS notes TEXT AFTER issue_date;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT TRUE AFTER notes;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS supersedes_id INT AFTER is_current;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP AFTER uploaded_by;

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
