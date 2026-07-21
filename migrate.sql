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

-- Add admin user
INSERT IGNORE INTO users (username, email, password_hash, full_name, role, is_active)
VALUES ('admin', 'admin@npl.com', '$2b$12$d3B1dphBAxJfGuLIcbaa1OMq8wElLjof1pPFoEn9v/eC82VFBNbG6', 'Administrator', 'admin', TRUE);
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

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
    step_number INT,
    process_id INT,
    process_name VARCHAR(100),
    machine_id INT,
    vendor_id INT,
    setup_time DECIMAL(10,2),
    cycle_time DECIMAL(10,2),
    cost_type VARCHAR(20),
    cost_value DECIMAL(12,4),
    notes TEXT
);

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
    other_costs DECIMAL(12,4) DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0,
    margin_percent DECIMAL(6,2) DEFAULT 20,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS item_number INT AFTER product_id;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS tooling_cost DECIMAL(12,4) DEFAULT 0 AFTER quantity_5;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS tooling_amortization_qty INT DEFAULT 0 AFTER tooling_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS material_cost DECIMAL(12,4) DEFAULT 0 AFTER tooling_amortization_qty;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS machining_cost DECIMAL(12,4) DEFAULT 0 AFTER material_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS other_costs DECIMAL(12,4) DEFAULT 0 AFTER machining_cost;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS total_cost DECIMAL(12,4) DEFAULT 0 AFTER other_costs;
ALTER TABLE quotation_items ADD COLUMN IF NOT EXISTS margin_percent DECIMAL(6,2) DEFAULT 20 AFTER total_cost;

-- =============================================
-- QUOTATION OPERATIONS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS quotation_operations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    quotation_id INT NOT NULL,
    operation_name VARCHAR(100),
    sequence INT,
    machine_id INT,
    vendor_id INT,
    setup_time DECIMAL(10,2),
    cycle_time DECIMAL(10,2),
    cost_type VARCHAR(20),
    cost_value DECIMAL(12,4),
    cost_amount DECIMAL(12,4) DEFAULT 0
);

-- =============================================
-- ATTACHMENTS TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS attachments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entity_type VARCHAR(30) NOT NULL,
    entity_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    original_name VARCHAR(255),
    file_path VARCHAR(500) NOT NULL,
    file_size INT,
    mime_type VARCHAR(100),
    uploaded_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE attachments ADD COLUMN IF NOT EXISTS entity_type VARCHAR(30) NOT NULL AFTER id;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS original_name VARCHAR(255) AFTER file_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) NOT NULL AFTER original_name;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS file_size INT AFTER file_path;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) AFTER file_size;
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS uploaded_by INT AFTER mime_type;

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
    raw_material_cost DECIMAL(12,4) DEFAULT 0,
    machining_cost DECIMAL(12,4) DEFAULT 0,
    other_costs DECIMAL(12,4) DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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
    entity_type VARCHAR(30),
    entity_id INT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE notifications ADD COLUMN IF NOT EXISTS notification_type VARCHAR(50) AFTER message;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS entity_type VARCHAR(30) AFTER notification_type;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS entity_id INT AFTER entity_type;
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS is_read BOOLEAN DEFAULT FALSE AFTER entity_id;

-- =============================================
-- ADD INDEXES IF NOT EXISTS
-- =============================================
-- Note: MySQL doesn't support ADD INDEX IF NOT EXISTS, so we skip this
-- You can manually add indexes if needed

SELECT 'Migration completed successfully!' AS status;
