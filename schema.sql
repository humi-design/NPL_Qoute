-- NPL Fasteners ERP - Complete Database Schema
-- Run in phpMyAdmin SQL tab

-- Create database
CREATE DATABASE IF NOT EXISTS humijxhw_npl_web CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE humijxhw_npl_web;

-- Users table
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
    last_login DATETIME,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- Insert admin user (password: admin123)
INSERT INTO users (username, email, password_hash, full_name, role, is_active)
VALUES ('admin', 'admin@npl.com', '$2b$12$d3B1dphBAxJfGuLIcbaa1OMq8wElLjof1pPFoEn9v/eC82VFBNbG6', 'Administrator', 'admin', TRUE)
ON DUPLICATE KEY UPDATE password_hash=VALUES(password_hash);

INSERT INTO users (username, email, password_hash, full_name, role, is_active)
VALUES ('sales', 'sales@npl.com', '$2b$12$0vR2yNYn3.wmb2epITX8iebCSKdf3h9BXaXutixaubCHvqahqpjfu', 'Sales User', 'sales', TRUE)
ON DUPLICATE KEY UPDATE password_hash=VALUES(password_hash);

-- Customers table
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
    payment_terms TEXT,
    incoterms VARCHAR(100),
    shipping_terms TEXT,
    billing_address TEXT,
    shipping_address TEXT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_customer_code (customer_code),
    INDEX idx_company_name (company_name)
);

-- Materials table
CREATE TABLE IF NOT EXISTS materials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    material_code VARCHAR(30) NOT NULL UNIQUE,
    material_name VARCHAR(100) NOT NULL,
    density DECIMAL(10,4) DEFAULT 8.0,
    hsn_code VARCHAR(20),
    grade VARCHAR(50),
    default_supplier VARCHAR(100),
    default_rate DECIMAL(12,4),
    rate_unit VARCHAR(20) DEFAULT 'per_kg',
    remarks TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_material_code (material_code)
);

INSERT INTO materials (material_code, material_name, density, hsn_code, is_active) VALUES
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

-- Machines table
CREATE TABLE IF NOT EXISTS machines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    machine_code VARCHAR(30) NOT NULL UNIQUE,
    machine_name VARCHAR(100) NOT NULL,
    hourly_rate DECIMAL(12,4) DEFAULT 0,
    operator VARCHAR(100),
    power_kw DECIMAL(8,2),
    efficiency DECIMAL(5,2) DEFAULT 100,
    department VARCHAR(50),
    default_setup_time INT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_machine_code (machine_code)
);

INSERT INTO machines (machine_code, machine_name, hourly_rate, department, is_active) VALUES
('TRAUB', 'Traub TNL32', 800, 'Machining', TRUE),
('CNC1', 'CNC Lathe 1', 600, 'Machining', TRUE),
('CNC2', 'CNC Lathe 2', 600, 'Machining', TRUE),
('MILL', 'Milling Machine', 500, 'Machining', TRUE),
('DRILL', 'Drilling Machine', 300, 'Machining', TRUE),
('GRIND', 'Grinding Machine', 450, 'Finishing', TRUE);

-- Vendors table
CREATE TABLE IF NOT EXISTS vendors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_code VARCHAR(30) NOT NULL UNIQUE,
    vendor_name VARCHAR(200) NOT NULL,
    contact_person VARCHAR(120),
    email VARCHAR(120),
    phone VARCHAR(30),
    gst VARCHAR(30),
    address TEXT,
    country VARCHAR(100) DEFAULT 'India',
    default_currency VARCHAR(10) DEFAULT 'INR',
    lead_time_days INT,
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_vendor_code (vendor_code)
);

-- Processes table
CREATE TABLE IF NOT EXISTS processes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    process_code VARCHAR(30) NOT NULL UNIQUE,
    process_name VARCHAR(100) NOT NULL,
    process_type VARCHAR(20) DEFAULT 'internal',
    description TEXT,
    machine_id INT,
    vendor_id INT,
    department VARCHAR(50),
    setup_time DECIMAL(10,2),
    cycle_time DECIMAL(10,2),
    cost_type VARCHAR(20) DEFAULT 'cycle_time',
    cost_value DECIMAL(12,4),
    cost_formula VARCHAR(200),
    is_custom BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_process_code (process_code)
);

INSERT INTO processes (process_code, process_name, process_type, department, cost_type, is_active) VALUES
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

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_code VARCHAR(50) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    customer_part_number VARCHAR(100),
    drawing_number VARCHAR(100),
    revision VARCHAR(10),
    material VARCHAR(100),
    weight DECIMAL(12,4),
    standard VARCHAR(50),
    standard_number VARCHAR(30),
    thread VARCHAR(50),
    dimensions TEXT,
    description TEXT,
    manufacturing_template_id INT,
    status VARCHAR(20) DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_product_code (product_code),
    INDEX idx_product_name (product_name),
    INDEX idx_drawing_number (drawing_number),
    INDEX idx_customer_part (customer_part_number)
);

-- Quotations table
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_quotation_number (quotation_number),
    INDEX idx_customer (customer_id),
    INDEX idx_status (status)
);

-- Quotation Items table
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (quotation_id) REFERENCES quotations(id) ON DELETE CASCADE,
    INDEX idx_quotation (quotation_id)
);

-- RFQs table
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_rfq_number (rfq_number),
    INDEX idx_customer (customer_id),
    INDEX idx_status (status)
);

-- RFQ Items table
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rfq_id) REFERENCES rfqs(id) ON DELETE CASCADE
);

-- Material Rates table
CREATE TABLE IF NOT EXISTS material_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    material_id INT NOT NULL,
    rate DECIMAL(12,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'per_kg',
    effective_date DATE,
    supplier VARCHAR(100),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES materials(id),
    INDEX idx_material (material_id)
);

-- Vendor Rates table
CREATE TABLE IF NOT EXISTS vendor_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT NOT NULL,
    process_id INT,
    rate DECIMAL(12,4) NOT NULL,
    unit VARCHAR(20),
    effective_date DATE,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    INDEX idx_vendor (vendor_id)
);

-- Machine Rates table
CREATE TABLE IF NOT EXISTS machine_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    machine_id INT NOT NULL,
    rate DECIMAL(12,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'per_hour',
    effective_date DATE,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(id),
    INDEX idx_machine (machine_id)
);

-- Process Templates table
CREATE TABLE IF NOT EXISTS process_templates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    template_code VARCHAR(30) NOT NULL UNIQUE,
    template_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Process Template Steps table
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
    notes TEXT,
    FOREIGN KEY (template_id) REFERENCES process_templates(id) ON DELETE CASCADE
);

-- Attachments table
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity (entity_type, entity_id)
);

-- Revision History table
CREATE TABLE IF NOT EXISTS revision_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entity_type VARCHAR(30) NOT NULL,
    entity_id INT NOT NULL,
    field_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    changed_by INT,
    changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity (entity_type, entity_id)
);

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    notification_type VARCHAR(50),
    entity_type VARCHAR(30),
    entity_id INT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_user (user_id)
);

-- System Settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value TEXT,
    description VARCHAR(200),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Currencies table
CREATE TABLE IF NOT EXISTS currencies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    currency_code VARCHAR(10) NOT NULL UNIQUE,
    currency_name VARCHAR(100) NOT NULL,
    exchange_rate DECIMAL(12,4) DEFAULT 1,
    symbol VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Allowances table
CREATE TABLE IF NOT EXISTS allowances (
    id INT AUTO_INCREMENT PRIMARY KEY,
    allowance_name VARCHAR(50) NOT NULL,
    allowance_type VARCHAR(20) DEFAULT 'machining',
    default_value DECIMAL(10,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample currencies
INSERT INTO currencies (currency_code, currency_name, exchange_rate, symbol, is_active) VALUES
('INR', 'Indian Rupee', 1.0, '₹', TRUE),
('USD', 'US Dollar', 83.0, '$', TRUE),
('EUR', 'Euro', 90.0, '€', TRUE),
('GBP', 'British Pound', 105.0, '£', TRUE),
('AED', 'UAE Dirham', 22.0, 'د.إ', TRUE);

-- Insert default allowances
INSERT INTO allowances (allowance_name, allowance_type, default_value, is_active) VALUES
('Parting', 'cutting', 3.0, TRUE),
('Facing', 'machining', 2.0, TRUE),
('Machining Allowance', 'machining', 1.0, TRUE),
('Grinding Allowance', 'grinding', 0.5, TRUE),
('Chamfer Allowance', 'finishing', 0.5, TRUE);

-- Product Cost History table
CREATE TABLE IF NOT EXISTS product_cost_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    raw_material_cost DECIMAL(12,4) DEFAULT 0,
    machining_cost DECIMAL(12,4) DEFAULT 0,
    other_costs DECIMAL(12,4) DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    INDEX idx_product (product_id),
    INDEX idx_recorded (recorded_at)
);

-- Quotation Operations table
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
    cost_amount DECIMAL(12,4) DEFAULT 0,
    FOREIGN KEY (quotation_id) REFERENCES quotations(id) ON DELETE CASCADE
);

-- Raw Material Calculations table
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
