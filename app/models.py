from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication and authorization."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120))
    role = db.Column(db.String(20), nullable=False, default='read_only')  # admin, sales, production, accounts, read_only
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    quotations = db.relationship('Quotation', backref='created_by_user', lazy='dynamic', foreign_keys='Quotation.created_by')
    revision_history = db.relationship('RevisionHistory', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def has_permission(self, permission):
        """Check if user has specific permission."""
        permissions = {
            'admin': ['read', 'write', 'delete', 'admin', 'reports'],
            'sales': ['read', 'write', 'quotations', 'customers', 'rfqs'],
            'production': ['read', 'write', 'processes', 'machines', 'products'],
            'accounts': ['read', 'reports', 'quotations'],
            'read_only': ['read']
        }
        return permission in permissions.get(self.role, [])


class Customer(db.Model):
    """Customer model."""
    __tablename__ = 'customers'
    
    id = db.Column(db.Integer, primary_key=True)
    customer_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    company_name = db.Column(db.String(200), nullable=False)
    contact_person = db.Column(db.String(120))
    email = db.Column(db.String(120), index=True)
    phone = db.Column(db.String(30))
    mobile = db.Column(db.String(30))
    gst_vat = db.Column(db.String(30))
    country = db.Column(db.String(100), default='India')
    currency = db.Column(db.String(10), default='INR')
    payment_terms = db.Column(db.String(100))
    incoterms = db.Column(db.String(50))
    shipping_terms = db.Column(db.String(200))
    
    # Addresses
    billing_address = db.Column(db.Text)
    shipping_address = db.Column(db.Text)
    delivery_address = db.Column(db.Text)
    
    # Notes and status
    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    rfqs = db.relationship('RFQ', backref='customer', lazy='dynamic')
    quotations = db.relationship('Quotation', backref='customer', lazy='dynamic')
    
    def __repr__(self):
        return f'<Customer {self.customer_code} - {self.company_name}>'


class RFQ(db.Model):
    """Request for Quotation model."""
    __tablename__ = 'rfqs'
    
    id = db.Column(db.Integer, primary_key=True)
    rfq_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    subject = db.Column(db.String(300))
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, converted, lost
    priority = db.Column(db.String(20), default='normal')  # low, normal, high, urgent
    due_date = db.Column(db.Date)
    conversion_date = db.Column(db.DateTime)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    items = db.relationship('RFQItem', backref='rfq', lazy='dynamic', cascade='all, delete-orphan')
    attachments = db.relationship('Attachment', backref='rfq', lazy='dynamic', 
                                 primaryjoin="and_(RFQ.id==Attachment.entity_id, Attachment.entity_type=='rfq')",
                                 foreign_keys='Attachment.entity_id', viewonly=True)
    
    def __repr__(self):
        return f'<RFQ {self.rfq_number}>'


class RFQItem(db.Model):
    """RFQ Item model."""
    __tablename__ = 'rfq_items'
    
    id = db.Column(db.Integer, primary_key=True)
    rfq_id = db.Column(db.Integer, db.ForeignKey('rfqs.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    part_description = db.Column(db.String(300))
    drawing_number = db.Column(db.String(100))
    material = db.Column(db.String(100))
    quantity = db.Column(db.Integer)
    target_price = db.Column(db.Float)
    required_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    sequence = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    product = db.relationship('Product', backref='rfq_items')
    
    def __repr__(self):
        return f'<RFQItem {self.id}>'


class Product(db.Model):
    """Product Library model."""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    internal_code = db.Column(db.String(30), unique=True, nullable=False, index=True)
    product_name = db.Column(db.String(200), nullable=False)
    customer_part_number = db.Column(db.String(100), index=True)
    drawing_number = db.Column(db.String(100), index=True)
    revision = db.Column(db.String(10), default='A')
    
    # Technical specs
    material = db.Column(db.String(100))
    weight = db.Column(db.Float)  # grams
    standard = db.Column(db.String(50))  # DIN, ISO, etc.
    standard_number = db.Column(db.String(30))  # 933, 935, etc.
    thread = db.Column(db.String(50))
    dimensions = db.Column(db.Text)  # JSON string for dimensions
    
    # Description
    description = db.Column(db.Text)
    
    # Manufacturing route template
    manufacturing_template_id = db.Column(db.Integer, db.ForeignKey('process_templates.id'))
    
    # Status
    status = db.Column(db.String(20), default='active')  # active, inactive, archived
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    quotations = db.relationship('QuotationItem', backref='product', lazy='dynamic')
    attachments = db.relationship('Attachment', backref='product', lazy='dynamic',
                                 primaryjoin="and_(Product.id==Attachment.entity_id, Attachment.entity_type=='product')",
                                 foreign_keys='Attachment.entity_id', viewonly=True)
    cost_history = db.relationship('ProductCostHistory', backref='product', lazy='dynamic')
    manufacturing_template = db.relationship('ProcessTemplate', backref='products', foreign_keys=[manufacturing_template_id])
    
    def __repr__(self):
        return f'<Product {self.internal_code} - {self.product_name}>'


class Material(db.Model):
    """Material Master model."""
    __tablename__ = 'materials'
    
    id = db.Column(db.Integer, primary_key=True)
    material_code = db.Column(db.String(30), unique=True, nullable=False)
    material_name = db.Column(db.String(100), nullable=False)
    density = db.Column(db.Float, nullable=False)  # kg/m3
    grade = db.Column(db.String(50))
    hsn_code = db.Column(db.String(20))
    default_supplier = db.Column(db.String(100))
    default_rate = db.Column(db.Float)  # per kg
    rate_unit = db.Column(db.String(20), default='₹/kg')  # ₹/kg, ₹/bar, ₹/bundle, ₹/piece
    remarks = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    rates = db.relationship('MaterialRate', backref='material', lazy='dynamic', cascade='all, delete-orphan')
    cost_history = db.relationship('ProductCostHistory', backref='material_used', lazy='dynamic')
    
    def __repr__(self):
        return f'<Material {self.material_code} - {self.material_name}>'


class MaterialRate(db.Model):
    """Material Rate History model."""
    __tablename__ = 'material_rates'
    
    id = db.Column(db.Integer, primary_key=True)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), nullable=False)
    rate = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), default='₹/kg')
    supplier = db.Column(db.String(100))
    effective_date = db.Column(db.Date, nullable=False)
    is_current = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<MaterialRate {self.material_id} - {self.rate}>'


class Machine(db.Model):
    """Machine model."""
    __tablename__ = 'machines'
    
    id = db.Column(db.Integer, primary_key=True)
    machine_code = db.Column(db.String(30), unique=True, nullable=False)
    machine_name = db.Column(db.String(100), nullable=False)
    machine_type = db.Column(db.String(50))  # CNC, VMC, Traub, Press, etc.
    hourly_rate = db.Column(db.Float, nullable=False)  # per hour
    operator_name = db.Column(db.String(100))
    power_consumption = db.Column(db.Float)  # kW
    efficiency = db.Column(db.Float, default=100.0)  # percentage
    department = db.Column(db.String(50))
    default_setup_time = db.Column(db.Float, default=0)  # minutes
    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    processes = db.relationship('Process', backref='machine', lazy='dynamic')
    
    def __repr__(self):
        return f'<Machine {self.machine_code} - {self.machine_name}>'


class Vendor(db.Model):
    """Vendor model."""
    __tablename__ = 'vendors'
    
    id = db.Column(db.Integer, primary_key=True)
    vendor_code = db.Column(db.String(30), unique=True, nullable=False)
    vendor_name = db.Column(db.String(200), nullable=False)
    contact_person = db.Column(db.String(100))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(30))
    mobile = db.Column(db.String(30))
    gst = db.Column(db.String(30))
    address = db.Column(db.Text)
    lead_time = db.Column(db.Integer)  # days
    default_currency = db.Column(db.String(10), default='INR')
    payment_terms = db.Column(db.String(100))
    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    rates = db.relationship('VendorRate', backref='vendor', lazy='dynamic', cascade='all, delete-orphan')
    processes = db.relationship('Process', backref='vendor', lazy='dynamic',
                                primaryjoin="and_(Vendor.id==Process.vendor_id, Process.process_type=='vendor')",
                                foreign_keys='Process.vendor_id', viewonly=True)
    
    def __repr__(self):
        return f'<Vendor {self.vendor_code} - {self.vendor_name}>'


class VendorRate(db.Model):
    """Vendor Rate History model."""
    __tablename__ = 'vendor_rates'
    
    id = db.Column(db.Integer, primary_key=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), nullable=False)
    service_name = db.Column(db.String(100), nullable=False)
    rate = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), default='₹/piece')  # ₹/piece, ₹/kg, ₹/batch
    min_quantity = db.Column(db.Integer, default=1)
    lead_time_days = db.Column(db.Integer)
    effective_date = db.Column(db.Date, nullable=False)
    is_current = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<VendorRate {self.vendor_id} - {self.service_name}>'


class Process(db.Model):
    """Process/Operation model."""
    __tablename__ = 'processes'
    
    id = db.Column(db.Integer, primary_key=True)
    process_code = db.Column(db.String(30), unique=True, nullable=False)
    process_name = db.Column(db.String(100), nullable=False)
    process_type = db.Column(db.String(20), default='internal')  # internal, vendor
    description = db.Column(db.Text)
    
    # Machine assignment
    machine_id = db.Column(db.Integer, db.ForeignKey('machines.id'))
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'))
    
    # Department
    department = db.Column(db.String(50))
    
    # Timing
    setup_time = db.Column(db.Float, default=0)  # minutes
    cycle_time = db.Column(db.Float, default=0)  # seconds per piece
    
    # Cost type and value
    cost_type = db.Column(db.String(30), default='cycle_time')  # cycle_time, per_piece, per_kg, per_batch, percentage, manual, formula
    cost_value = db.Column(db.Float, default=0)
    cost_formula = db.Column(db.String(200))  # For custom formulas
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    is_custom = db.Column(db.Boolean, default=False)  # User-created custom processes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    quotations = db.relationship('QuotationOperation', backref='process', lazy='dynamic')
    
    def __repr__(self):
        return f'<Process {self.process_code} - {self.process_name}>'


class ProcessTemplate(db.Model):
    """Manufacturing Process Template model."""
    __tablename__ = 'process_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    template_code = db.Column(db.String(30), unique=True, nullable=False)
    template_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    steps = db.relationship('ProcessTemplateStep', backref='template', lazy='dynamic', 
                            cascade='all, delete-orphan', order_by='ProcessTemplateStep.sequence')
    
    def __repr__(self):
        return f'<ProcessTemplate {self.template_code} - {self.template_name}>'


class ProcessTemplateStep(db.Model):
    """Process Template Step model."""
    __tablename__ = 'process_template_steps'
    
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.Integer, db.ForeignKey('process_templates.id'), nullable=False)
    process_id = db.Column(db.Integer, db.ForeignKey('processes.id'))
    sequence = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    
    process = db.relationship('Process')
    
    def __repr__(self):
        return f'<ProcessTemplateStep {self.id}>'


class Allowance(db.Model):
    """Material Allowances model."""
    __tablename__ = 'allowances'
    
    id = db.Column(db.Integer, primary_key=True)
    allowance_code = db.Column(db.String(30), unique=True, nullable=False)
    allowance_name = db.Column(db.String(100), nullable=False)
    allowance_type = db.Column(db.String(20), default='machining')  # machining, grinding, facing, custom
    value = db.Column(db.Float, nullable=False)  # mm
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Allowance {self.allowance_code} - {self.allowance_name}>'


class Quotation(db.Model):
    """Quotation model."""
    __tablename__ = 'quotations'
    
    id = db.Column(db.Integer, primary_key=True)
    quotation_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    quotation_version = db.Column(db.String(10), default='V1')
    quotation_ref = db.Column(db.String(30), index=True)  # Links to original quotation for revisions
    
    # Customer
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    
    # Dates
    quotation_date = db.Column(db.Date, nullable=False)
    valid_until = db.Column(db.Date)
    
    # Status
    status = db.Column(db.String(20), default='draft')  # draft, sent, won, lost, revised
    priority = db.Column(db.String(20), default='normal')
    
    # Currency and terms
    currency = db.Column(db.String(10), default='INR')
    exchange_rate = db.Column(db.Float, default=1.0)
    
    # Addresses
    billing_address = db.Column(db.Text)
    shipping_address = db.Column(db.Text)
    
    # Terms
    payment_terms = db.Column(db.String(200))
    delivery_terms = db.Column(db.String(200))
    warranty = db.Column(db.String(100))
    
    # Totals
    subtotal = db.Column(db.Float, default=0)
    tax_percent = db.Column(db.Float, default=18)
    tax_amount = db.Column(db.Float, default=0)
    total_amount = db.Column(db.Float, default=0)
    margin_percent = db.Column(db.Float, default=0)
    
    # Notes
    internal_notes = db.Column(db.Text)
    terms_conditions = db.Column(db.Text)
    
    # Meta
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sent_date = db.Column(db.DateTime)
    won_date = db.Column(db.DateTime)
    lost_date = db.Column(db.DateTime)
    lost_reason = db.Column(db.Text)
    
    # Relationships
    items = db.relationship('QuotationItem', backref='quotation', lazy='dynamic', cascade='all, delete-orphan')
    operations = db.relationship('QuotationOperation', backref='quotation', lazy='dynamic', cascade='all, delete-orphan')
    attachments = db.relationship('Attachment', backref='quotation', lazy='dynamic',
                                  primaryjoin="and_(Quotation.id==Attachment.entity_id, Attachment.entity_type=='quotation')",
                                  foreign_keys='Attachment.entity_id', viewonly=True)
    
    def __repr__(self):
        return f'<Quotation {self.quotation_number}>'
    
    @property
    def is_revision(self):
        return self.quotation_ref is not None


class QuotationItem(db.Model):
    """Quotation Item model."""
    __tablename__ = 'quotation_items'
    
    id = db.Column(db.Integer, primary_key=True)
    quotation_id = db.Column(db.Integer, db.ForeignKey('quotations.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    
    # Product info (copied from product, not linked)
    part_description = db.Column(db.String(300))
    drawing_number = db.Column(db.String(100))
    material = db.Column(db.String(100))
    
    # Quantity breaks
    quantity_1 = db.Column(db.Integer, default=0)
    quantity_2 = db.Column(db.Integer, default=0)
    quantity_3 = db.Column(db.Integer, default=0)
    quantity_4 = db.Column(db.Integer, default=0)
    quantity_5 = db.Column(db.Integer, default=0)
    
    # Pricing per quantity break
    unit_price_1 = db.Column(db.Float, default=0)
    unit_price_2 = db.Column(db.Float, default=0)
    unit_price_3 = db.Column(db.Float, default=0)
    unit_price_4 = db.Column(db.Float, default=0)
    unit_price_5 = db.Column(db.Float, default=0)
    
    # Tooling
    tooling_cost = db.Column(db.Float, default=0)
    tooling_amortization_qty = db.Column(db.Integer, default=5000)
    
    # Cost breakdown
    material_cost = db.Column(db.Float, default=0)
    machining_cost = db.Column(db.Float, default=0)
    vendor_cost = db.Column(db.Float, default=0)
    other_cost = db.Column(db.Float, default=0)
    overhead_amount = db.Column(db.Float, default=0)
    profit_amount = db.Column(db.Float, default=0)
    
    # Final
    total_cost = db.Column(db.Float, default=0)
    margin_percent = db.Column(db.Float, default=0)
    
    # Status
    sequence = db.Column(db.Integer, default=0)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    operation_breakdown = db.relationship('QuotationOperation', backref='quotation_item', lazy='dynamic',
                                           cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<QuotationItem {self.id}>'


class QuotationOperation(db.Model):
    """Quotation Operation breakdown model."""
    __tablename__ = 'quotation_operations'
    
    id = db.Column(db.Integer, primary_key=True)
    quotation_id = db.Column(db.Integer, db.ForeignKey('quotations.id'), nullable=False)
    quotation_item_id = db.Column(db.Integer, db.ForeignKey('quotation_items.id'))
    process_id = db.Column(db.Integer, db.ForeignKey('processes.id'))
    
    # Operation details
    operation_name = db.Column(db.String(100))
    operation_type = db.Column(db.String(20))  # material, machining, vendor, overhead
    sequence = db.Column(db.Integer, default=0)
    
    # Cost calculation
    machine_id = db.Column(db.Integer, db.ForeignKey('machines.id'))
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'))
    
    setup_time = db.Column(db.Float, default=0)
    cycle_time = db.Column(db.Float, default=0)
    
    cost_type = db.Column(db.String(30))
    cost_value = db.Column(db.Float, default=0)
    calculated_cost = db.Column(db.Float, default=0)
    
    quantity_for_calculation = db.Column(db.Integer, default=1)
    
    notes = db.Column(db.Text)
    
    # Relationships
    machine = db.relationship('Machine', backref='quotation_operations')
    vendor = db.relationship('Vendor', backref='quotation_operations')
    
    def __repr__(self):
        return f'<QuotationOperation {self.operation_name}>'


class Attachment(db.Model):
    """File attachment model."""
    __tablename__ = 'attachments'
    
    id = db.Column(db.Integer, primary_key=True)
    entity_type = db.Column(db.String(20), nullable=False)  # customer, rfq, product, quotation, vendor, etc.
    entity_id = db.Column(db.Integer, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    file_path = db.Column(db.String(500), nullable=False)
    is_primary = db.Column(db.Boolean, default=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Attachment {self.original_name}>'


class RevisionHistory(db.Model):
    """Revision History tracking model."""
    __tablename__ = 'revision_history'
    
    id = db.Column(db.Integer, primary_key=True)
    entity_type = db.Column(db.String(30), nullable=False)  # quotation, product, customer, etc.
    entity_id = db.Column(db.Integer, nullable=False)
    field_name = db.Column(db.String(100), nullable=False)
    old_value = db.Column(db.Text)
    new_value = db.Column(db.Text)
    changed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    changed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<RevisionHistory {self.entity_type}:{self.entity_id} - {self.field_name}>'


class Currency(db.Model):
    """Currency model."""
    __tablename__ = 'currencies'
    
    id = db.Column(db.Integer, primary_key=True)
    currency_code = db.Column(db.String(10), unique=True, nullable=False)
    currency_name = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(10))
    exchange_rate = db.Column(db.Float, default=1.0)
    is_base = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Currency {self.currency_code}>'


class SystemSetting(db.Model):
    """System settings model."""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.Text)
    description = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemSetting {self.setting_key}>'


class ProductCostHistory(db.Model):
    """Product Cost History model."""
    __tablename__ = 'product_cost_history'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'))
    quotation_id = db.Column(db.Integer, db.ForeignKey('quotations.id'))
    
    # Cost breakdown
    raw_material_cost = db.Column(db.Float, default=0)
    machining_cost = db.Column(db.Float, default=0)
    vendor_cost = db.Column(db.Float, default=0)
    tooling_cost = db.Column(db.Float, default=0)
    overhead_cost = db.Column(db.Float, default=0)
    total_cost = db.Column(db.Float, default=0)
    selling_price = db.Column(db.Float, default=0)
    margin_percent = db.Column(db.Float, default=0)
    
    # Quantity
    quantity = db.Column(db.Integer)
    
    # Meta
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    def __repr__(self):
        return f'<ProductCostHistory {self.product_id}>'


class Notification(db.Model):
    """Notification model."""
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text)
    notification_type = db.Column(db.String(50))  # quotation_created, material_rate_changed, etc.
    reference_type = db.Column(db.String(30))  # quotation, rfq, product, etc.
    reference_id = db.Column(db.Integer)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='notifications')
    
    def __repr__(self):
        return f'<Notification {self.title}>'
