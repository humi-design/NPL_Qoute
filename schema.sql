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
    supplier_id INT,
    default_rate DECIMAL(12,4),
    rate_unit VARCHAR(20) DEFAULT 'per_kg',
    remarks TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_material_code (material_code),
    INDEX idx_material_supplier (supplier_id)
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
('TITANIUM', 'Titanium Grade 5', 4.43, '8108', TRUE)
ON DUPLICATE KEY UPDATE material_code=VALUES(material_code);

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
('GRIND', 'Grinding Machine', 450, 'Finishing', TRUE)
ON DUPLICATE KEY UPDATE machine_code=VALUES(machine_code);

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
('PACK', 'Packing', 'internal', 'Packing', 'per_piece', TRUE)
ON DUPLICATE KEY UPDATE process_code=VALUES(process_code);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    internal_code VARCHAR(50) NOT NULL UNIQUE,
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
    INDEX idx_internal_code (internal_code),
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
    process_id INT,
    sequence INT NOT NULL DEFAULT 1,
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
    title VARCHAR(200),
    file_type VARCHAR(50),
    file_path VARCHAR(500) NOT NULL,
    file_size INT,
    is_primary BOOLEAN DEFAULT FALSE,
    mime_type VARCHAR(100),
    -- document control (drawings, certificates, reports)
    category VARCHAR(40) DEFAULT 'other',
    revision VARCHAR(20),
    issue_date DATE,
    notes TEXT,
    is_current BOOLEAN DEFAULT TRUE,
    supersedes_id INT,
    uploaded_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_att_category (category),
    INDEX idx_att_current (is_current)
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
    is_base BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Allowances table
CREATE TABLE IF NOT EXISTS allowances (
    id INT AUTO_INCREMENT PRIMARY KEY,
    allowance_code VARCHAR(30) NOT NULL UNIQUE,
    allowance_name VARCHAR(100) NOT NULL,
    allowance_type VARCHAR(20) DEFAULT 'machining',
    value FLOAT NOT NULL,
    default_value DECIMAL(10,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample currencies
INSERT INTO currencies (currency_code, currency_name, exchange_rate, symbol, is_base, is_active) VALUES
('INR', 'Indian Rupee', 1.0, '₹', TRUE, TRUE),
('USD', 'US Dollar', 83.0, '$', FALSE, TRUE),
('EUR', 'Euro', 90.0, '€', FALSE, TRUE),
('GBP', 'British Pound', 105.0, '£', FALSE, TRUE),
('AED', 'UAE Dirham', 22.0, 'د.إ', FALSE, TRUE)
ON DUPLICATE KEY UPDATE currency_code=VALUES(currency_code);

-- Insert default allowances
INSERT INTO allowances (allowance_code, allowance_name, allowance_type, value, default_value, is_active) VALUES
('ALL-PART', 'Parting', 'cutting', 3.0, 3.0, TRUE),
('ALL-FACE', 'Facing', 'machining', 2.0, 2.0, TRUE),
('ALL-MACH', 'Machining Allowance', 'machining', 1.0, 1.0, TRUE),
('ALL-GRND', 'Grinding Allowance', 'grinding', 0.5, 0.5, TRUE),
('ALL-CHAM', 'Chamfer Allowance', 'finishing', 0.5, 0.5, TRUE)
ON DUPLICATE KEY UPDATE value=VALUES(value), default_value=VALUES(default_value);

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
    notes TEXT,
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    INDEX idx_wo_number (work_order_number),
    INDEX idx_wo_vendor (vendor_id),
    INDEX idx_wo_status (status),
    INDEX idx_wo_due (due_date)
);

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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(id),
    INDEX idx_vqi_vendor (vendor_id),
    INDEX idx_vqi_work_order (work_order_id)
);

-- =============================================
-- SUPPLIER TYPES TABLE (configurable master)
-- =============================================
CREATE TABLE IF NOT EXISTS supplier_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type_code VARCHAR(30) NOT NULL UNIQUE,
    type_name VARCHAR(100) NOT NULL,
    description TEXT,
    sort_order INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_supplier_type_code (type_code)
);

-- =============================================
-- SUPPLIERS TABLE (procurement master)
-- =============================================
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_type_id) REFERENCES supplier_types(id),
    INDEX idx_supplier_code (supplier_code),
    INDEX idx_supplier_name (company_name),
    INDEX idx_supplier_type (supplier_type_id)
);

-- =============================================
-- SUPPLIER MATERIALS TABLE (materials/grades supplied)
-- =============================================
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
    FOREIGN KEY (material_id) REFERENCES materials(id),
    INDEX idx_supplier_material_supplier (supplier_id),
    INDEX idx_supplier_material_material (material_id)
);

-- =============================================
-- PROCUREMENT: PURCHASE ORDERS, RECEIPTS & BATCHES
-- =============================================
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
-- PAYABLES: SUPPLIER INVOICES, PAYMENTS & ALLOCATIONS
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
-- PAYABLES: VENDOR (JOB-WORK) INVOICES, PAYMENTS & ALLOCATIONS
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
