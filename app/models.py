from flask_login import UserMixin
from datetime import datetime, date
from app.extensions import db


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
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), index=True)
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
    work_orders = db.relationship('VendorWorkOrder', backref='vendor', lazy='dynamic',
                                  cascade='all, delete-orphan')
    quality_issues = db.relationship('VendorQualityIssue', backref='vendor', lazy='dynamic',
                                     cascade='all, delete-orphan')
    
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


class VendorWorkOrder(db.Model):
    """Vendor Work Order / Job model.

    Tracks work assigned to a vendor (outsourced manufacturing operations),
    quantities ordered and received, delivery dates and status. This is the
    record that powers the vendor "Current / Past / All" work views.
    """
    __tablename__ = 'vendor_work_orders'

    id = db.Column(db.Integer, primary_key=True)
    work_order_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), nullable=False, index=True)

    # Optional links into the existing manufacturing/quotation chain
    process_id = db.Column(db.Integer, db.ForeignKey('processes.id'))
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    quotation_id = db.Column(db.Integer, db.ForeignKey('quotations.id'))

    # Work description
    title = db.Column(db.String(200))
    part_description = db.Column(db.String(300))
    drawing_number = db.Column(db.String(100))
    service_name = db.Column(db.String(100))
    notes = db.Column(db.Text)

    # Quantities
    quantity_ordered = db.Column(db.Integer, default=0)
    quantity_received = db.Column(db.Integer, default=0)
    quantity_rejected = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(20), default='pcs')

    # Commercials
    rate = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')

    # Dates
    order_date = db.Column(db.Date)
    due_date = db.Column(db.Date)
    completed_date = db.Column(db.Date)

    # Status: pending, in_progress, completed, cancelled
    status = db.Column(db.String(20), default='pending', index=True)
    priority = db.Column(db.String(20), default='normal')

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    process = db.relationship('Process', backref='vendor_work_orders')
    product = db.relationship('Product', backref='vendor_work_orders')
    quotation = db.relationship('Quotation', backref='vendor_work_orders')
    quality_issues = db.relationship('VendorQualityIssue', backref='work_order', lazy='dynamic')

    OPEN_STATUSES = ('pending', 'in_progress')
    COMPLETE_STATUSES = ('completed', 'cancelled')

    @property
    def pending_quantity(self):
        """Quantity still outstanding (ordered - received, never negative)."""
        return max((self.quantity_ordered or 0) - (self.quantity_received or 0), 0)

    @property
    def is_open(self):
        """True while the work order is still being worked (not completed/cancelled)."""
        return self.status in self.OPEN_STATUSES

    @property
    def is_completed(self):
        return self.status == 'completed'

    @property
    def is_cancelled(self):
        return self.status == 'cancelled'

    @property
    def is_overdue(self):
        """Open work whose due date has already passed."""
        if not self.is_open or not self.due_date:
            return False
        return self.due_date < date.today()

    @property
    def days_overdue(self):
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days

    @property
    def order_value(self):
        return (self.quantity_ordered or 0) * (self.rate or 0)

    @property
    def completion_percent(self):
        ordered = self.quantity_ordered or 0
        if ordered <= 0:
            return 100 if self.is_completed else 0
        return min(round((self.quantity_received or 0) / ordered * 100, 1), 100)

    def __repr__(self):
        return f'<VendorWorkOrder {self.work_order_number}>'


class VendorQualityIssue(db.Model):
    """Vendor Quality Issue / NCR model.

    Records rejection or quality complaints against a vendor, optionally tied
    to a specific work order. Only used where data exists.
    """
    __tablename__ = 'vendor_quality_issues'

    id = db.Column(db.Integer, primary_key=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), nullable=False, index=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('vendor_work_orders.id'))

    issue_date = db.Column(db.Date, nullable=False)
    issue_type = db.Column(db.String(30), default='rejection')  # rejection, rework, complaint, short_supply
    severity = db.Column(db.String(20), default='minor')  # minor, major, critical
    description = db.Column(db.Text)
    quantity_affected = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='open')  # open, in_progress, resolved, closed
    resolution = db.Column(db.Text)
    resolved_date = db.Column(db.Date)

    reported_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    OPEN_STATUSES = ('open', 'in_progress')

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    def __repr__(self):
        return f'<VendorQualityIssue {self.id} - {self.vendor_id}>'


class SupplierType(db.Model):
    """Configurable supplier category (raw material, consumable, service, ...).

    Types are data, not code, so authorised users can add/rename/deactivate
    them without a schema change. Deleting a type is blocked while suppliers
    still reference it.
    """
    __tablename__ = 'supplier_types'

    id = db.Column(db.Integer, primary_key=True)
    type_code = db.Column(db.String(30), unique=True, nullable=False, index=True)
    type_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    sort_order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    suppliers = db.relationship('Supplier', backref='supplier_type', lazy='dynamic')

    def __repr__(self):
        return f'<SupplierType {self.type_code} - {self.type_name}>'


class Supplier(db.Model):
    """Supplier Master model.

    Distinct from Vendor: a supplier is the procurement-side party we buy
    raw materials/consumables from (see SupplierMaterial). Vendors are the
    subcontracting/service side (see VendorWorkOrder). Kept separate to
    preserve the existing vendor relationships and reports.
    """
    __tablename__ = 'suppliers'

    id = db.Column(db.Integer, primary_key=True)
    supplier_code = db.Column(db.String(30), unique=True, nullable=False, index=True)
    company_name = db.Column(db.String(200), nullable=False)
    supplier_type_id = db.Column(db.Integer, db.ForeignKey('supplier_types.id'), index=True)

    # Primary contact
    contact_person = db.Column(db.String(120))
    email = db.Column(db.String(120), index=True)
    phone = db.Column(db.String(30))
    mobile = db.Column(db.String(30))
    website = db.Column(db.String(200))

    # Address
    address_line1 = db.Column(db.String(200))
    address_line2 = db.Column(db.String(200))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    postal_code = db.Column(db.String(20))
    country = db.Column(db.String(100), default='India')

    # Tax / business identifiers
    gst_vat = db.Column(db.String(30))
    pan = db.Column(db.String(30))
    cin = db.Column(db.String(30))
    tax_id = db.Column(db.String(50))

    # Commercial terms
    default_currency = db.Column(db.String(10), default='INR')
    payment_terms = db.Column(db.String(100))
    lead_time = db.Column(db.Integer)  # days

    # Document references (free text; binary files live in attachments table)
    document_references = db.Column(db.Text)

    notes = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    supplied_materials = db.relationship('SupplierMaterial', backref='supplier',
                                         lazy='dynamic', cascade='all, delete-orphan')

    @property
    def full_address(self):
        parts = [self.address_line1, self.address_line2, self.city,
                 self.state, self.postal_code, self.country]
        return ', '.join(p for p in parts if p)

    def __repr__(self):
        return f'<Supplier {self.supplier_code} - {self.company_name}>'


class SupplierMaterial(db.Model):
    """Link table: which materials (and grades) a supplier provides."""
    __tablename__ = 'supplier_materials'

    id = db.Column(db.Integer, primary_key=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), nullable=False, index=True)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), index=True)
    grade = db.Column(db.String(50))
    rate = db.Column(db.Float)
    rate_unit = db.Column(db.String(20), default='₹/kg')
    lead_time = db.Column(db.Integer)
    notes = db.Column(db.Text)
    is_preferred = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    material = db.relationship('Material', backref=db.backref('supplier_links', lazy='dynamic'))

    def __repr__(self):
        return f'<SupplierMaterial {self.supplier_id} - {self.material_id}>'


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
    """File attachment / controlled document.

    ``entity_type`` + ``entity_id`` link the document to whatever it belongs to
    (customer, rfq, product, quotation, order, supplier, vendor, lot, inspection,
    dispatch, ...). Documents also carry a ``category`` so drawings, material
    certificates, inspection reports and supplier documents can be found
    centrally, and a ``revision`` so the revision that applies to a record can be
    identified. ``is_current`` marks the revision in force; superseded revisions
    are kept (never deleted) so the history is preserved.
    """
    __tablename__ = 'attachments'

    id = db.Column(db.Integer, primary_key=True)
    entity_type = db.Column(db.String(20), nullable=False, index=True)
    entity_id = db.Column(db.Integer, nullable=False, index=True)
    file_name = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    file_path = db.Column(db.String(500), nullable=False)
    is_primary = db.Column(db.Boolean, default=False)

    # Document management fields
    title = db.Column(db.String(200))
    category = db.Column(db.String(40), default='other', index=True)
    revision = db.Column(db.String(20))
    issue_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    is_current = db.Column(db.Boolean, default=True, index=True)
    supersedes_id = db.Column(db.Integer, db.ForeignKey('attachments.id'))

    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    DOCUMENT_CATEGORIES = (
        'drawing', 'revision', 'specification', 'material_certificate',
        'test_certificate', 'inspection_report', 'supplier_document',
        'purchase_order', 'packing_list', 'invoice', 'other',
    )

    superseded_by = db.relationship(
        'Attachment', backref=db.backref('supersedes', remote_side=[id]),
        foreign_keys=[supersedes_id], uselist=False)

    @property
    def display_name(self):
        return self.title or self.original_name

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
    
    user = db.relationship('User', backref=db.backref('notifications', lazy='dynamic'))
    
    def __repr__(self):
        return f'<Notification {self.title}>'


# ============================================================
# PROCUREMENT: Supplier Purchase Orders, Receipts and Batches
# ============================================================
# These tables capture the procurement data the supplier profile needs
# (purchase orders, ordered/received/outstanding quantities, delivery history,
# material batches, heat/lot numbers, certificates and inspection results).
# Quantities are never stored twice: a line's received quantity is derived from
# its receipt lines, so ordered/received/outstanding always reconcile.

class SupplierPurchaseOrder(db.Model):
    """A purchase order raised on a supplier."""
    __tablename__ = 'supplier_purchase_orders'

    id = db.Column(db.Integer, primary_key=True)
    po_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), nullable=False, index=True)

    order_date = db.Column(db.Date)
    expected_date = db.Column(db.Date)
    # draft, sent, partially_received, received, cancelled, closed
    status = db.Column(db.String(20), default='draft', index=True)

    currency = db.Column(db.String(10), default='INR')
    payment_terms = db.Column(db.String(100))
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    lines = db.relationship('SupplierPurchaseOrderLine', backref='purchase_order',
                            lazy='dynamic', cascade='all, delete-orphan')
    receipts = db.relationship('SupplierOrderReceipt', backref='purchase_order',
                               lazy='dynamic', cascade='all, delete-orphan')
    supplier = db.relationship('Supplier', backref=db.backref(
        'purchase_orders', lazy='dynamic'))

    OPEN_STATUSES = ('draft', 'sent', 'partially_received')
    CLOSED_STATUSES = ('received', 'cancelled', 'closed')

    @property
    def total_value(self):
        return sum(l.line_value for l in self.lines.all())

    @property
    def ordered_quantity(self):
        return sum((l.quantity_ordered or 0) for l in self.lines.all())

    @property
    def received_quantity(self):
        return sum(l.received_quantity for l in self.lines.all())

    @property
    def rejected_quantity(self):
        return sum(l.rejected_quantity for l in self.lines.all())

    @property
    def accepted_quantity(self):
        return sum(l.accepted_quantity for l in self.lines.all())

    @property
    def pending_quantity(self):
        return sum(l.pending_quantity for l in self.lines.all())

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_cancelled(self):
        return self.status == 'cancelled'

    @property
    def is_overdue(self):
        """Open PO whose expected date has passed with quantity still pending."""
        if not self.is_open or not self.expected_date:
            return False
        return self.expected_date < date.today() and self.pending_quantity > 0

    @property
    def days_overdue(self):
        if not self.is_overdue:
            return 0
        return (date.today() - self.expected_date).days

    @property
    def completion_percent(self):
        ordered = self.ordered_quantity
        if ordered <= 0:
            return 100 if not self.is_open else 0
        return min(round(self.received_quantity / ordered * 100, 1), 100)

    def __repr__(self):
        return f'<SupplierPurchaseOrder {self.po_number}>'


class SupplierPurchaseOrderLine(db.Model):
    """A single material line on a purchase order."""
    __tablename__ = 'supplier_po_lines'

    id = db.Column(db.Integer, primary_key=True)
    po_id = db.Column(db.Integer, db.ForeignKey('supplier_purchase_orders.id'),
                      nullable=False, index=True)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), index=True)

    description = db.Column(db.String(300))
    grade = db.Column(db.String(50))
    quantity_ordered = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='kg')
    rate = db.Column(db.Float, default=0)
    rate_unit = db.Column(db.String(20), default='₹/kg')
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    material = db.relationship('Material', backref='po_lines')
    receipt_lines = db.relationship('SupplierOrderReceiptLine', backref='po_line',
                                    lazy='dynamic')

    @property
    def line_value(self):
        return (self.quantity_ordered or 0) * (self.rate or 0)

    @property
    def received_quantity(self):
        return sum((rl.quantity_received or 0) for rl in self.receipt_lines.all())

    @property
    def rejected_quantity(self):
        return sum((rl.quantity_rejected or 0) for rl in self.receipt_lines.all())

    @property
    def accepted_quantity(self):
        return sum(rl.accepted_quantity for rl in self.receipt_lines.all())

    @property
    def pending_quantity(self):
        return max((self.quantity_ordered or 0) - self.received_quantity, 0)

    def __repr__(self):
        return f'<SupplierPurchaseOrderLine {self.id}>'


class SupplierOrderReceipt(db.Model):
    """Goods receipt note (GRN) recording what physically arrived for a PO."""
    __tablename__ = 'supplier_order_receipts'

    id = db.Column(db.Integer, primary_key=True)
    receipt_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    po_id = db.Column(db.Integer, db.ForeignKey('supplier_purchase_orders.id'),
                      nullable=False, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'),
                            nullable=False, index=True)

    receipt_date = db.Column(db.Date, nullable=False)
    supplier_document = db.Column(db.String(100))  # supplier challan / DC number
    notes = db.Column(db.Text)
    received_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    lines = db.relationship('SupplierOrderReceiptLine', backref='receipt',
                            lazy='dynamic', cascade='all, delete-orphan')
    supplier = db.relationship('Supplier', backref=db.backref(
        'order_receipts', lazy='dynamic'))

    @property
    def received_quantity(self):
        return sum((l.quantity_received or 0) for l in self.lines.all())

    @property
    def rejected_quantity(self):
        return sum((l.quantity_rejected or 0) for l in self.lines.all())

    def __repr__(self):
        return f'<SupplierOrderReceipt {self.receipt_number}>'


class SupplierOrderReceiptLine(db.Model):
    """A receipt line carrying batch/heat/lot numbers and inspection results."""
    __tablename__ = 'supplier_receipt_lines'

    id = db.Column(db.Integer, primary_key=True)
    receipt_id = db.Column(db.Integer, db.ForeignKey('supplier_order_receipts.id'),
                           nullable=False, index=True)
    po_line_id = db.Column(db.Integer, db.ForeignKey('supplier_po_lines.id'), index=True)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), index=True)

    quantity_received = db.Column(db.Float, default=0)
    quantity_rejected = db.Column(db.Float, default=0)

    batch_number = db.Column(db.String(60))
    heat_number = db.Column(db.String(60))
    lot_number = db.Column(db.String(60))
    certificate_ref = db.Column(db.String(120))  # mill / test certificate reference

    # pending, passed, failed, partial
    inspection_status = db.Column(db.String(20), default='pending')
    inspection_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    material = db.relationship('Material', backref='receipt_lines')

    @property
    def accepted_quantity(self):
        return max((self.quantity_received or 0) - (self.quantity_rejected or 0), 0)

    def __repr__(self):
        return f'<SupplierOrderReceiptLine {self.id}>'



# ============================================================
# PAYABLES: Supplier Invoices, Payments and Allocations
# ============================================================
# A payment is recorded once as a header and applied to invoices through
# allocation rows. A UNIQUE(payment_id, invoice_id) constraint means a payment
# can be applied to a given invoice only once, so payments and receipts are
# never double counted. An unallocated payment is an advance/credit.

class SupplierInvoice(db.Model):
    """A supplier invoice (payable)."""
    __tablename__ = 'supplier_invoices'

    id = db.Column(db.Integer, primary_key=True)
    invoice_number = db.Column(db.String(60), nullable=False, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'),
                            nullable=False, index=True)
    po_id = db.Column(db.Integer, db.ForeignKey('supplier_purchase_orders.id'), index=True)

    invoice_date = db.Column(db.Date, nullable=False)
    due_date = db.Column(db.Date)
    subtotal = db.Column(db.Float, default=0)
    tax_amount = db.Column(db.Float, default=0)
    total_amount = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    # draft, posted, cancelled
    status = db.Column(db.String(20), default='posted', index=True)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('supplier_id', 'invoice_number',
                            name='uq_supplier_invoice_number'),
    )

    purchase_order = db.relationship('SupplierPurchaseOrder', backref='invoices')
    allocations = db.relationship('SupplierPaymentAllocation', backref='invoice',
                                  lazy='dynamic', cascade='all, delete-orphan')
    supplier = db.relationship('Supplier', backref=db.backref(
        'invoices', lazy='dynamic'))

    @property
    def paid_amount(self):
        """Sum of allocations; each allocation row is counted exactly once."""
        return sum((a.amount or 0) for a in self.allocations.all())

    @property
    def outstanding_amount(self):
        if self.status == 'cancelled':
            return 0.0
        return max((self.total_amount or 0) - self.paid_amount, 0)

    @property
    def is_paid(self):
        return (self.total_amount or 0) > 0 and self.paid_amount >= (self.total_amount or 0)

    @property
    def days_overdue(self):
        if not self.due_date or self.outstanding_amount <= 0:
            return 0
        return max((date.today() - self.due_date).days, 0)

    @property
    def ageing_bucket(self):
        """Ageing bucket by days past due: not_due, 0-30, 31-60, 61-90, 90+."""
        if self.outstanding_amount <= 0:
            return None
        d = self.days_overdue
        if d == 0:
            return 'not_due'
        if d <= 30:
            return '0-30'
        if d <= 60:
            return '31-60'
        if d <= 90:
            return '61-90'
        return '90+'

    def __repr__(self):
        return f'<SupplierInvoice {self.invoice_number}>'


class SupplierPayment(db.Model):
    """A payment made to a supplier. Unallocated amount = advance/credit."""
    __tablename__ = 'supplier_payments'

    id = db.Column(db.Integer, primary_key=True)
    payment_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'),
                            nullable=False, index=True)
    payment_date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    method = db.Column(db.String(20))  # bank, cash, upi, cheque
    reference = db.Column(db.String(100))
    is_advance = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    allocations = db.relationship('SupplierPaymentAllocation', backref='payment',
                                  lazy='dynamic', cascade='all, delete-orphan')
    supplier = db.relationship('Supplier', backref=db.backref(
        'payments', lazy='dynamic'))

    @property
    def allocated_amount(self):
        return sum((a.amount or 0) for a in self.allocations.all())

    @property
    def unallocated_amount(self):
        return max((self.amount or 0) - self.allocated_amount, 0)

    def __repr__(self):
        return f'<SupplierPayment {self.payment_number}>'


class SupplierPaymentAllocation(db.Model):
    """Applies one payment to one invoice. UNIQUE(payment_id, invoice_id)
    guarantees a payment cannot be counted twice against the same invoice."""
    __tablename__ = 'supplier_payment_allocations'

    id = db.Column(db.Integer, primary_key=True)
    payment_id = db.Column(db.Integer, db.ForeignKey('supplier_payments.id'),
                           nullable=False, index=True)
    invoice_id = db.Column(db.Integer, db.ForeignKey('supplier_invoices.id'),
                           nullable=False, index=True)
    amount = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('payment_id', 'invoice_id',
                            name='uq_supplier_alloc_payment_invoice'),
    )

    def __repr__(self):
        return f'<SupplierPaymentAllocation {self.payment_id}->{self.invoice_id}>'


# ============================================================
# PAYABLES: Vendor (Job-Work) Invoices, Payments and Allocations
# ============================================================
# Same header + allocation model as suppliers, kept in separate tables so the
# existing vendor relationships and reports stay untouched.

class VendorInvoice(db.Model):
    """A job-work invoice raised by a vendor (payable)."""
    __tablename__ = 'vendor_invoices'

    id = db.Column(db.Integer, primary_key=True)
    invoice_number = db.Column(db.String(60), nullable=False, index=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'),
                          nullable=False, index=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey('vendor_work_orders.id'), index=True)

    invoice_date = db.Column(db.Date, nullable=False)
    due_date = db.Column(db.Date)
    subtotal = db.Column(db.Float, default=0)
    tax_amount = db.Column(db.Float, default=0)
    total_amount = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    status = db.Column(db.String(20), default='posted', index=True)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('vendor_id', 'invoice_number',
                            name='uq_vendor_invoice_number'),
    )

    work_order = db.relationship('VendorWorkOrder', backref='invoices')
    allocations = db.relationship('VendorPaymentAllocation', backref='invoice',
                                  lazy='dynamic', cascade='all, delete-orphan')
    vendor = db.relationship('Vendor', backref=db.backref(
        'invoices', lazy='dynamic'))

    @property
    def paid_amount(self):
        return sum((a.amount or 0) for a in self.allocations.all())

    @property
    def outstanding_amount(self):
        if self.status == 'cancelled':
            return 0.0
        return max((self.total_amount or 0) - self.paid_amount, 0)

    @property
    def is_paid(self):
        return (self.total_amount or 0) > 0 and self.paid_amount >= (self.total_amount or 0)

    @property
    def days_overdue(self):
        if not self.due_date or self.outstanding_amount <= 0:
            return 0
        return max((date.today() - self.due_date).days, 0)

    @property
    def ageing_bucket(self):
        if self.outstanding_amount <= 0:
            return None
        d = self.days_overdue
        if d == 0:
            return 'not_due'
        if d <= 30:
            return '0-30'
        if d <= 60:
            return '31-60'
        if d <= 90:
            return '61-90'
        return '90+'

    def __repr__(self):
        return f'<VendorInvoice {self.invoice_number}>'


class VendorPayment(db.Model):
    """A payment made to a vendor. Unallocated amount = advance/credit."""
    __tablename__ = 'vendor_payments'

    id = db.Column(db.Integer, primary_key=True)
    payment_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'),
                          nullable=False, index=True)
    payment_date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    method = db.Column(db.String(20))
    reference = db.Column(db.String(100))
    is_advance = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    allocations = db.relationship('VendorPaymentAllocation', backref='payment',
                                  lazy='dynamic', cascade='all, delete-orphan')
    vendor = db.relationship('Vendor', backref=db.backref(
        'payments', lazy='dynamic'))

    @property
    def allocated_amount(self):
        return sum((a.amount or 0) for a in self.allocations.all())

    @property
    def unallocated_amount(self):
        return max((self.amount or 0) - self.allocated_amount, 0)

    def __repr__(self):
        return f'<VendorPayment {self.payment_number}>'


class VendorPaymentAllocation(db.Model):
    """Applies one vendor payment to one invoice (no double counting)."""
    __tablename__ = 'vendor_payment_allocations'

    id = db.Column(db.Integer, primary_key=True)
    payment_id = db.Column(db.Integer, db.ForeignKey('vendor_payments.id'),
                           nullable=False, index=True)
    invoice_id = db.Column(db.Integer, db.ForeignKey('vendor_invoices.id'),
                           nullable=False, index=True)
    amount = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('payment_id', 'invoice_id',
                            name='uq_vendor_alloc_payment_invoice'),
    )

    def __repr__(self):
        return f'<VendorPaymentAllocation {self.payment_id}->{self.invoice_id}>'



# ============================================================
# ORDERS & MANUFACTURING TRACKING
# ============================================================
# A confirmed sales order (typically from a won quotation) anchors the whole
# production lifecycle:
#
#   Order -> OrderOperation (the part's actual route, ordered by sequence)
#            -> OperationTransaction (issue / return / accept / reject / rework / scrap)
#   Order -> MaterialLot (received stock with heat/lot/certificate)
#            -> MaterialMovement (receipt / allocation / issue / return / scrap / shortage)
#   Order -> Inspection (incoming / in_process / final)
#   Order -> PackagingRecord (packed + warehouse receipt -> completion)
#
# Quantities are always *derived* from the ledger rows (never stored twice), so
# issued / returned / accepted / rejected / rework / scrapped always reconcile.
# A UNIQUE constraint on each ledger stops the same movement being posted twice.

class Order(db.Model):
    """A confirmed customer order (sales order) being manufactured."""
    __tablename__ = 'orders'

    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(30), unique=True, nullable=False, index=True)

    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False, index=True)
    # Optional links back into the existing sales chain
    quotation_id = db.Column(db.Integer, db.ForeignKey('quotations.id'), index=True)
    quotation_item_id = db.Column(db.Integer, db.ForeignKey('quotation_items.id'), index=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), index=True)

    # Part / drawing reference (copied so the order is self-contained)
    part_description = db.Column(db.String(300))
    drawing_number = db.Column(db.String(100), index=True)
    revision = db.Column(db.String(10))
    specifications = db.Column(db.Text)
    material_grade = db.Column(db.String(100))
    standard = db.Column(db.String(50))

    # PLANNED values
    quantity_ordered = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(20), default='pcs')
    rate = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')

    # Dates: planned vs actual are distinct columns.
    order_date = db.Column(db.Date, nullable=False)
    due_date = db.Column(db.Date)
    planned_start = db.Column(db.Date)
    actual_start = db.Column(db.Date)
    actual_completion = db.Column(db.Date)

    # draft, confirmed, in_production, completed, on_hold, cancelled
    status = db.Column(db.String(20), default='confirmed', index=True)
    priority = db.Column(db.String(20), default='normal')
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    customer = db.relationship('Customer', backref=db.backref('orders', lazy='dynamic'))
    quotation = db.relationship('Quotation', backref='orders')
    product = db.relationship('Product', backref=db.backref('orders', lazy='dynamic'))
    operations = db.relationship('OrderOperation', backref='order', lazy='dynamic',
                                 cascade='all, delete-orphan',
                                 order_by='OrderOperation.sequence')
    lots = db.relationship('MaterialLot', backref='order', lazy='dynamic')
    inspections = db.relationship('Inspection', backref='order', lazy='dynamic',
                                  cascade='all, delete-orphan')
    packaging = db.relationship('PackagingRecord', backref='order', lazy='dynamic',
                                cascade='all, delete-orphan')
    cost_entries = db.relationship('OrderCostEntry', backref='order', lazy='dynamic',
                                   cascade='all, delete-orphan')

    OPEN_STATUSES = ('draft', 'confirmed', 'in_production', 'on_hold')
    COMPLETE_STATUSES = ('completed', 'cancelled')

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_completed(self):
        return self.status == 'completed'

    @property
    def is_cancelled(self):
        return self.status == 'cancelled'

    @property
    def is_overdue(self):
        if not self.is_open or not self.due_date:
            return False
        return self.due_date < date.today()

    @property
    def days_overdue(self):
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days

    @property
    def order_value(self):
        return (self.quantity_ordered or 0) * (self.rate or 0)

    # ---- Derived actuals (single source of truth: the ledgers) ----

    @property
    def quantity_produced(self):
        """Good pieces produced = accepted qty of the final operation(s), or the
        best accepted qty across the route when no final step is flagged."""
        ops = self.operations.all()
        if not ops:
            return 0
        finals = [o for o in ops if o.is_final]
        if finals:
            return sum(o.accepted_quantity for o in finals)
        return max((o.accepted_quantity for o in ops), default=0)

    @property
    def quantity_rejected(self):
        return sum(o.rejected_quantity for o in self.operations.all())

    @property
    def quantity_rework(self):
        return sum(o.rework_quantity for o in self.operations.all())

    @property
    def quantity_warehoused(self):
        """Accepted quantity packed and received into finished-goods stock.

        This is a stock position, not a shipment: pieces can sit warehoused for
        a while before any dispatch consumes them.
        """
        return sum(p.quantity_accepted for p in self.packaging.all()
                   if p.status in ('warehoused', 'dispatched'))

    @property
    def quantity_shipped(self):
        """Accepted quantity actually dispatched to the customer (excludes
        cancelled dispatches), so it can be lower than ``quantity_warehoused``
        until the whole stock has shipped."""
        return sum(d.quantity_shipped for d in self.dispatches.all()
                   if d.status != 'cancelled')

    @property
    def quantity_dispatched(self):
        """Deprecated alias for ``quantity_warehoused`` kept for compatibility.

        The name is historical; use ``quantity_warehoused`` (stock) or
        ``quantity_shipped`` (sent to customer) for new code.
        """
        return self.quantity_warehoused

    @property
    def pending_quantity(self):
        return max((self.quantity_ordered or 0) - self.quantity_produced, 0)

    @property
    def completion_percent(self):
        ordered = self.quantity_ordered or 0
        if ordered <= 0:
            return 100 if self.is_completed else 0
        return min(round(self.quantity_produced / ordered * 100, 1), 100)

    @property
    def current_operation(self):
        """The earliest operation in the route that is not yet complete."""
        for op in self.operations.all():
            if not op.is_complete:
                return op
        return None

    def __repr__(self):
        return f'<Order {self.order_number}>'


class OrderOperation(db.Model):
    """One step in a specific order's manufacturing route.

    The route is defined per order (not a fixed template) so different
    fasteners can follow different sequences and mix in-house and vendor
    operations. ``vendor_id`` is set when the step is subcontracted.
    """
    __tablename__ = 'order_operations'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False, index=True)

    sequence = db.Column(db.Integer, nullable=False, default=1)
    process_id = db.Column(db.Integer, db.ForeignKey('processes.id'))
    operation_name = db.Column(db.String(100), nullable=False)
    is_vendor = db.Column(db.Boolean, default=False)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), index=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machines.id'))

    # PLANNED values
    quantity_planned = db.Column(db.Integer, default=0)
    planned_issue_date = db.Column(db.Date)
    due_date = db.Column(db.Date)
    rate = db.Column(db.Float, default=0)

    # ACTUAL (manually confirmed) dates
    actual_start = db.Column(db.Date)
    actual_end = db.Column(db.Date)

    # pending, in_progress, completed, cancelled
    status = db.Column(db.String(20), default='pending', index=True)
    is_final = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    process = db.relationship('Process', backref='order_operations')
    vendor = db.relationship('Vendor', backref=db.backref('order_operations', lazy='dynamic'))
    machine = db.relationship('Machine', backref='order_operations')
    transactions = db.relationship('OperationTransaction', backref='operation',
                                   lazy='dynamic', cascade='all, delete-orphan')
    inspections = db.relationship('Inspection', backref='operation', lazy='dynamic')

    OPEN_STATUSES = ('pending', 'in_progress')

    # ---- Derived quantities from the transaction ledger ----

    def _sum(self, transaction_type):
        return sum(t.quantity for t in self.transactions.filter_by(
            transaction_type=transaction_type).all())

    @property
    def issued_quantity(self):
        """Quantity issued to this operation (vendor or in-house)."""
        return self._sum('issue')

    @property
    def returned_quantity(self):
        return self._sum('return')

    @property
    def processed_quantity(self):
        """Quantity processed = issued that has come back as a 'return'."""
        return self.returned_quantity

    @property
    def accepted_quantity(self):
        return self._sum('accept')

    @property
    def rejected_quantity(self):
        return self._sum('reject')

    @property
    def rework_quantity(self):
        return self._sum('rework')

    @property
    def scrapped_quantity(self):
        return self._sum('scrap')

    @property
    def pending_quantity(self):
        """Issued but not yet returned / rejected / scrapped."""
        outstanding = (self.issued_quantity - self.returned_quantity
                       - self.rejected_quantity - self.scrapped_quantity)
        return max(outstanding, 0)

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_complete(self):
        return self.status == 'completed'

    @property
    def is_overdue(self):
        if not self.is_open or not self.due_date:
            return False
        return self.due_date < date.today()

    @property
    def completion_percent(self):
        planned = self.quantity_planned or 0
        if planned <= 0:
            return 100 if self.is_complete else 0
        return min(round(self.accepted_quantity / planned * 100, 1), 100)

    def __repr__(self):
        return f'<OrderOperation {self.order_id}#{self.sequence} {self.operation_name}>'


class OperationTransaction(db.Model):
    """Ledger of quantity movements for an order operation.

    A UNIQUE(order_operation_id, transaction_type, reference) constraint stops
    the same movement (e.g. the same delivery challan) being posted twice.
    Partial deliveries are simply multiple rows; multiple subcontractors are
    handled by having several vendor operations.
    """
    __tablename__ = 'operation_transactions'

    id = db.Column(db.Integer, primary_key=True)
    order_operation_id = db.Column(db.Integer, db.ForeignKey('order_operations.id'),
                                   nullable=False, index=True)

    # issue, return, accept, reject, rework, scrap
    transaction_type = db.Column(db.String(20), nullable=False, index=True)
    transaction_date = db.Column(db.Date, nullable=False)
    quantity = db.Column(db.Integer, default=0)
    # Challan / DC / reference so a duplicate posting can be detected.
    reference = db.Column(db.String(100))
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('order_operation_id', 'transaction_type', 'reference',
                            name='uq_op_txn_type_reference'),
    )

    def __repr__(self):
        return f'<OperationTransaction {self.transaction_type} {self.quantity}>'


class MaterialLot(db.Model):
    """A received material batch with its heat/lot/certificate details.

    Available / allocated / remaining are derived from MaterialMovement rows.
    """
    __tablename__ = 'material_lots'

    id = db.Column(db.Integer, primary_key=True)
    lot_number = db.Column(db.String(60), nullable=False, index=True)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), index=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), index=True)

    # Procurement references
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), index=True)
    po_id = db.Column(db.Integer, db.ForeignKey('supplier_purchase_orders.id'), index=True)
    receipt_id = db.Column(db.Integer, db.ForeignKey('supplier_order_receipts.id'), index=True)
    receipt_line_id = db.Column(db.Integer, db.ForeignKey('supplier_receipt_lines.id'), index=True)

    grade = db.Column(db.String(100))
    specification = db.Column(db.String(200))
    heat_number = db.Column(db.String(60))
    batch_number = db.Column(db.String(60))
    certificate_ref = db.Column(db.String(120))

    quantity_received = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='kg')
    receipt_date = db.Column(db.Date)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    movements = db.relationship('MaterialMovement', backref='lot', lazy='dynamic',
                                cascade='all, delete-orphan')
    material = db.relationship('Material', backref='lots')
    supplier = db.relationship('Supplier', backref=db.backref('material_lots', lazy='dynamic'))

    __table_args__ = (
        # A given receipt line produces at most one lot -> no duplicate stock.
        db.UniqueConstraint('receipt_line_id', name='uq_lot_receipt_line'),
    )

    # ---- Derived balances from the movement ledger ----

    def _sum(self, movement_type):
        return sum(m.quantity for m in self.movements.filter_by(
            movement_type=movement_type).all())

    @property
    def quantity_allocated(self):
        """Reserved for an order but not yet physically issued."""
        return self._sum('allocation')

    @property
    def quantity_issued(self):
        return self._sum('issue')

    @property
    def quantity_returned(self):
        return self._sum('return')

    @property
    def quantity_scrapped(self):
        return self._sum('scrap')

    @property
    def quantity_shortage(self):
        return self._sum('shortage')

    @property
    def quantity_available(self):
        """Received less everything that has left the lot (issued or scrapped)."""
        return max((self.quantity_received or 0) - self.quantity_issued
                   - self.quantity_scrapped, 0)

    @property
    def quantity_remaining(self):
        """Available and not reserved (free to allocate)."""
        return max(self.quantity_available - self.quantity_allocated, 0)

    def __repr__(self):
        return f'<MaterialLot {self.lot_number}>'


class MaterialMovement(db.Model):
    """Ledger of stock movements for a material lot.

    Types:
      receipt    - goods in (mirrors MaterialLot.quantity_received)
      allocation - reserve for an order (reduces remaining, not available)
      issue      - physically issued to production / a vendor operation
      return     - unused material returned to store
      scrap      - material scrapped
      shortage   - material short (recorded against an order/operation)
      adjustment - correction

    UNIQUE(lot_id, movement_type, reference) prevents duplicate postings.
    """
    __tablename__ = 'material_movements'

    id = db.Column(db.Integer, primary_key=True)
    lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'),
                       nullable=False, index=True)

    movement_type = db.Column(db.String(20), nullable=False, index=True)
    movement_date = db.Column(db.Date, nullable=False)
    quantity = db.Column(db.Float, default=0)

    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), index=True)
    order_operation_id = db.Column(db.Integer, db.ForeignKey('order_operations.id'), index=True)
    reference = db.Column(db.String(100))
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    order = db.relationship('Order', backref=db.backref('material_movements', lazy='dynamic'))
    operation = db.relationship('OrderOperation', backref='material_movements')

    __table_args__ = (
        db.UniqueConstraint('lot_id', 'movement_type', 'reference',
                            name='uq_movement_type_reference'),
    )

    def __repr__(self):
        return f'<MaterialMovement {self.movement_type} {self.quantity}>'


class Inspection(db.Model):
    """An inspection linked to an order, and optionally a lot/operation.

    Covers incoming (goods receipt), in-process and final inspection while
    preserving technical fields: acceptance criteria, tolerance and the raw
    measurement values.
    """
    __tablename__ = 'inspections'

    id = db.Column(db.Integer, primary_key=True)
    inspection_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False, index=True)
    order_operation_id = db.Column(db.Integer, db.ForeignKey('order_operations.id'), index=True)
    lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'), index=True)

    # incoming, in_process, final
    inspection_type = db.Column(db.String(20), default='final', index=True)
    inspection_date = db.Column(db.Date, nullable=False)

    # Technical fields
    acceptance_criteria = db.Column(db.Text)
    tolerance = db.Column(db.String(100))
    measurements = db.Column(db.Text)          # free text / JSON of measured values
    sample_size = db.Column(db.Integer, default=0)

    quantity_inspected = db.Column(db.Integer, default=0)
    quantity_accepted = db.Column(db.Integer, default=0)
    quantity_rejected = db.Column(db.Integer, default=0)
    quantity_rework = db.Column(db.Integer, default=0)

    # pending, passed, failed, partial
    result = db.Column(db.String(20), default='pending', index=True)
    notes = db.Column(db.Text)

    inspected_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    lot = db.relationship('MaterialLot', backref='inspections')

    @property
    def is_passed(self):
        return self.result == 'passed'

    @property
    def is_open(self):
        return self.result == 'pending'

    def __repr__(self):
        return f'<Inspection {self.inspection_number}>'


class PackagingRecord(db.Model):
    """Packaging + warehouse receipt record that completes an order."""
    __tablename__ = 'packaging_records'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False, index=True)

    packaging_date = db.Column(db.Date, nullable=False)
    packaging_type = db.Column(db.String(50))     # box, carton, pallet, bag...
    quantity_packed = db.Column(db.Integer, default=0)
    quantity_accepted = db.Column(db.Integer, default=0)
    quantity_rejected = db.Column(db.Integer, default=0)
    label_ref = db.Column(db.String(100))

    warehouse_receipt_number = db.Column(db.String(50))
    warehouse_receipt_date = db.Column(db.Date)
    location = db.Column(db.String(100))

    # packed, warehoused, dispatched
    status = db.Column(db.String(20), default='packed', index=True)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PackagingRecord order={self.order_id} {self.quantity_accepted}>'



# ============================================================
# INVENTORY: Stock locations, transfers and adjustments
# ============================================================
# The existing ledgers already hold the physical truth:
#   * MaterialLot + MaterialMovement        -> raw material / rejected / scrap
#   * OrderOperation + OperationTransaction -> WIP and material at subcontractors
#   * PackagingRecord                       -> finished goods (warehoused)
# Rather than a second, disconnected stock system these new tables add only
# what was missing: named stock locations, internal transfers between them,
# and authorised adjustments. Every balance stays derived from a ledger.

class StockLocation(db.Model):
    """A named place stock can sit (raw-material store, WIP, vendor, FG store...).

    ``kind`` tells the reports which stage of the flow a location represents so
    the stock position can be grouped without duplicating the ledgers.
    """
    __tablename__ = 'stock_locations'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(30), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)

    # raw_material, wip, subcontractor, finished_goods, rejected, scrap, other
    kind = db.Column(db.String(30), default='raw_material', index=True)
    # Optional link when the location is a subcontractor's site.
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), index=True)

    address = db.Column(db.String(300))
    is_active = db.Column(db.Boolean, default=True, index=True)
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    vendor = db.relationship('Vendor', backref=db.backref('stock_locations', lazy='dynamic'))

    LOCATION_KINDS = ('raw_material', 'wip', 'subcontractor', 'finished_goods',
                      'rejected', 'scrap', 'other')

    @property
    def is_vendor_location(self):
        return self.kind == 'subcontractor'

    def __repr__(self):
        return f'<StockLocation {self.code} {self.kind}>'


class StockTransfer(db.Model):
    """Movement of stock between two locations (e.g. store -> vendor -> store).

    Quantities are not stored on the location; the transfer rows are the ledger
    and the net position per location is derived from them.
    """
    __tablename__ = 'stock_transfers'

    id = db.Column(db.Integer, primary_key=True)
    transfer_number = db.Column(db.String(30), unique=True, nullable=False, index=True)

    from_location_id = db.Column(db.Integer, db.ForeignKey('stock_locations.id'), index=True)
    to_location_id = db.Column(db.Integer, db.ForeignKey('stock_locations.id'), index=True)

    transfer_date = db.Column(db.Date, nullable=False, index=True)
    quantity = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='pcs')

    material_lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'), index=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), index=True)
    order_operation_id = db.Column(db.Integer, db.ForeignKey('order_operations.id'), index=True)

    # pending, in_transit, received, cancelled
    status = db.Column(db.String(20), default='received', index=True)
    reference = db.Column(db.String(100))  # challan / DC number
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    from_location = db.relationship('StockLocation', foreign_keys=[from_location_id],
                                    backref=db.backref('transfers_out', lazy='dynamic'))
    to_location = db.relationship('StockLocation', foreign_keys=[to_location_id],
                                  backref=db.backref('transfers_in', lazy='dynamic'))
    lot = db.relationship('MaterialLot', backref=db.backref('transfers', lazy='dynamic'))
    order = db.relationship('Order', backref=db.backref('stock_transfers', lazy='dynamic'))
    operation = db.relationship('OrderOperation',
                                backref=db.backref('stock_transfers', lazy='dynamic'))

    OPEN_STATUSES = ('pending', 'in_transit')

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_effective(self):
        """Only a received transfer has actually moved the stock."""
        return self.status == 'received'

    def __repr__(self):
        return f'<StockTransfer {self.transfer_number} {self.quantity}>'


class StockAdjustment(db.Model):
    """An authorised correction to stock that is not a normal movement.

    Used for physical-count differences, write-offs and reversals. A reversal
    of an existing movement is recorded as a compensating adjustment (never by
    editing history), so the audit trail stays intact.
    """
    __tablename__ = 'stock_adjustments'

    id = db.Column(db.Integer, primary_key=True)
    adjustment_number = db.Column(db.String(30), unique=True, nullable=False, index=True)
    adjustment_date = db.Column(db.Date, nullable=False, index=True)

    material_lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'), index=True)
    location_id = db.Column(db.Integer, db.ForeignKey('stock_locations.id'), index=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), index=True)

    # increase, decrease, write_off, reversal, correction
    adjustment_type = db.Column(db.String(20), default='correction', index=True)
    quantity = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='kg')

    # What this adjustment corrects (for a reversal) - keeps the audit trail.
    reverses_movement_id = db.Column(db.Integer, db.ForeignKey('material_movements.id'),
                                     index=True)
    reverses_transfer_id = db.Column(db.Integer, db.ForeignKey('stock_transfers.id'),
                                     index=True)

    reason = db.Column(db.String(300))
    # Who authorised an out-of-policy adjustment (negative stock / over-allocation).
    authorized_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    reference = db.Column(db.String(100))
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    lot = db.relationship('MaterialLot', backref=db.backref('adjustments', lazy='dynamic'))
    location = db.relationship('StockLocation', backref=db.backref('adjustments', lazy='dynamic'))
    order = db.relationship('Order', backref=db.backref('stock_adjustments', lazy='dynamic'))

    ADJUSTMENT_TYPES = ('increase', 'decrease', 'write_off', 'reversal', 'correction')

    @property
    def signed_quantity(self):
        """Adjustments that add stock are positive; everything else removes."""
        if self.adjustment_type in ('increase', 'reversal'):
            return self.quantity or 0
        return -(self.quantity or 0)

    def __repr__(self):
        return f'<StockAdjustment {self.adjustment_number} {self.adjustment_type} {self.quantity}>'


# ============================================================
# COSTING: Actual cost entries per order / part
# ============================================================
# The quotation chain already stores the ESTIMATE (QuotationItem +
# QuotationOperation + ProductCostHistory). These rows store the ACTUAL costs
# that have been recorded against an order, one per category and source, so the
# two can be compared. A UNIQUE(source_type, source_id) constraint means the
# same invoice / movement can never be counted twice.

class OrderCostEntry(db.Model):
    """One recorded actual cost for an order, in a named category.

    Categories mirror the shop floor: material, subcontracting/job-work,
    machining, finishing, inspection, packaging, freight and other. ``source_type``
    + ``source_id`` tie the entry back to the document it came from so the same
    document is never counted twice (enforced by a DB unique constraint).
    """
    __tablename__ = 'order_cost_entries'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False, index=True)

    # material, subcontracting, machining, finishing, inspection, packaging,
    # freight, other
    category = db.Column(db.String(30), nullable=False, index=True)
    amount = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    # amount converted to the order currency, so mixed currencies still total.
    amount_base = db.Column(db.Float, default=0)

    entry_date = db.Column(db.Date, index=True)
    description = db.Column(db.String(300))

    # Where the cost came from: manual, material_movement, operation_transaction,
    # vendor_invoice, supplier_invoice, packaging_record, stock_adjustment
    source_type = db.Column(db.String(30), default='manual', index=True)
    source_id = db.Column(db.Integer)

    order_operation_id = db.Column(db.Integer, db.ForeignKey('order_operations.id'), index=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendors.id'), index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'), index=True)
    material_lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'), index=True)

    is_estimate = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    operation = db.relationship('OrderOperation', backref=db.backref('cost_entries', lazy='dynamic'))
    vendor = db.relationship('Vendor', backref=db.backref('order_cost_entries', lazy='dynamic'))
    supplier = db.relationship('Supplier', backref=db.backref('order_cost_entries', lazy='dynamic'))
    lot = db.relationship('MaterialLot', backref=db.backref('cost_entries', lazy='dynamic'))

    COST_CATEGORIES = ('material', 'subcontracting', 'machining', 'finishing',
                       'inspection', 'packaging', 'freight', 'other')

    __table_args__ = (
        # One document contributes at most one cost entry per category -> no
        # double counting of the same invoice/movement.
        db.UniqueConstraint('source_type', 'source_id', 'category',
                            name='uq_cost_source_category'),
    )

    @property
    def effective_amount(self):
        return self.amount_base if self.amount_base else (self.amount or 0)

    def __repr__(self):
        return f'<OrderCostEntry {self.order_id} {self.category} {self.amount}>'


# ============================================================
# PROCUREMENT: Supplier RFQs, quotes and comparison
# ============================================================
# The existing RFQ module is customer-side. These tables add the supplier-side
# request + quotes so comparable prices can be compared and a recommendation
# shown. Nothing is auto-awarded: an authorised user must approve explicitly.

class SupplierRFQ(db.Model):
    """A request for quotation sent to suppliers for material/services."""
    __tablename__ = 'supplier_rfqs'

    id = db.Column(db.Integer, primary_key=True)
    rfq_number = db.Column(db.String(30), unique=True, nullable=False, index=True)

    title = db.Column(db.String(200))
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), index=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), index=True)

    quantity = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='kg')
    grade = db.Column(db.String(50))

    rfq_date = db.Column(db.Date, nullable=False, index=True)
    due_date = db.Column(db.Date)
    required_date = db.Column(db.Date)

    # draft, sent, quoted, compared, awarded, cancelled
    status = db.Column(db.String(20), default='draft', index=True)
    delivery_terms = db.Column(db.String(200))
    payment_terms = db.Column(db.String(100))
    currency = db.Column(db.String(10), default='INR')
    notes = db.Column(db.Text)

    # The winning quote, once an authorised user approves one. Stored as a plain
    # id (no FK) to avoid a circular FK with supplier_quotes.supplier_rfq_id;
    # the award route validates it belongs to this RFQ.
    awarded_quote_id = db.Column(db.Integer, index=True)
    awarded_at = db.Column(db.DateTime)
    awarded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    # PO raised from the award, when the buyer creates one.
    po_id = db.Column(db.Integer, db.ForeignKey('supplier_purchase_orders.id'), index=True)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    material = db.relationship('Material', backref=db.backref('supplier_rfqs', lazy='dynamic'))
    order = db.relationship('Order', backref=db.backref('supplier_rfqs', lazy='dynamic'))
    quotes = db.relationship('SupplierQuote', backref='rfq', lazy='dynamic',
                             cascade='all, delete-orphan',
                             foreign_keys='SupplierQuote.supplier_rfq_id')
    awarded_quote = db.relationship('SupplierQuote', foreign_keys=[awarded_quote_id],
                                    primaryjoin='SupplierRFQ.awarded_quote_id == SupplierQuote.id',
                                    post_update=True, viewonly=True)
    purchase_order = db.relationship('SupplierPurchaseOrder',
                                     backref=db.backref('source_rfq', uselist=False))

    OPEN_STATUSES = ('draft', 'sent', 'quoted', 'compared')
    CLOSED_STATUSES = ('awarded', 'cancelled')

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_awarded(self):
        return self.status == 'awarded' and self.awarded_quote_id is not None

    def __repr__(self):
        return f'<SupplierRFQ {self.rfq_number}>'


class SupplierQuote(db.Model):
    """A supplier's quotation against a SupplierRFQ."""
    __tablename__ = 'supplier_quotes'

    id = db.Column(db.Integer, primary_key=True)
    supplier_rfq_id = db.Column(db.Integer, db.ForeignKey('supplier_rfqs.id'),
                                nullable=False, index=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'),
                            nullable=False, index=True)

    quote_reference = db.Column(db.String(60))
    quote_date = db.Column(db.Date, nullable=False, index=True)
    valid_until = db.Column(db.Date)

    quantity = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20), default='kg')
    unit_price = db.Column(db.Float, default=0)
    currency = db.Column(db.String(10), default='INR')
    tax_percent = db.Column(db.Float, default=0)

    lead_time_days = db.Column(db.Integer)
    payment_terms = db.Column(db.String(100))
    delivery_terms = db.Column(db.String(200))
    quality_rating = db.Column(db.Float)   # optional manual 0-100 score

    # draft, received, shortlisted, rejected, selected
    status = db.Column(db.String(20), default='received', index=True)
    notes = db.Column(db.Text)

    # Saved comparison result so the recommendation is auditable.
    comparison_score = db.Column(db.Float)
    comparison_rank = db.Column(db.Integer)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    supplier = db.relationship('Supplier', backref=db.backref('supplier_quotes', lazy='dynamic'))

    @property
    def extended_price(self):
        return (self.quantity or 0) * (self.unit_price or 0)

    @property
    def landed_price(self):
        """Unit price with tax applied - used for like-for-like comparison."""
        return (self.unit_price or 0) * (1 + (self.tax_percent or 0) / 100)

    def __repr__(self):
        return f'<SupplierQuote {self.supplier_id} {self.unit_price}>'


# ============================================================
# DISPATCH: shipments to customers and their line quantities
# ============================================================
# A dispatch is the outward movement of finished goods. It never invents a
# second stock balance: each dispatch line consumes quantity from a
# PackagingRecord (the finished-goods ledger), so what is available to ship is
# exactly what was packed and accepted into the warehouse. Partial dispatches
# leave the remainder visible without touching the order's completion rule.

class Dispatch(db.Model):
    """A shipment of finished goods to a customer against a sales order."""
    __tablename__ = 'dispatches'

    id = db.Column(db.Integer, primary_key=True)
    dispatch_number = db.Column(db.String(30), unique=True, nullable=False, index=True)

    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False, index=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), index=True)

    dispatch_date = db.Column(db.Date, nullable=False, index=True)
    # surface, air, sea, rail, courier, hand_delivery
    mode = db.Column(db.String(20), default='surface')
    transporter = db.Column(db.String(150))
    vehicle_number = db.Column(db.String(50))
    shipment_reference = db.Column(db.String(100))   # LR / docket / AWB / BL
    eway_bill_number = db.Column(db.String(50))
    invoice_number = db.Column(db.String(60))
    invoice_date = db.Column(db.Date)
    packing_list_ref = db.Column(db.String(60))
    destination = db.Column(db.String(200))

    # prepared, dispatched, in_transit, delivered, returned, cancelled
    status = db.Column(db.String(20), default='prepared', index=True)
    delivery_status = db.Column(db.String(20), default='pending')  # pending, partial, delivered
    delivered_date = db.Column(db.Date)
    pod_reference = db.Column(db.String(100))       # proof of delivery ref
    receiver_name = db.Column(db.String(120))
    remarks = db.Column(db.Text)

    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    order = db.relationship('Order', backref=db.backref('dispatches', lazy='dynamic'))
    customer = db.relationship('Customer', backref=db.backref('dispatches', lazy='dynamic'))
    lines = db.relationship('DispatchLine', backref='dispatch', lazy='dynamic',
                            cascade='all, delete-orphan')

    OPEN_STATUSES = ('prepared', 'dispatched', 'in_transit')

    @property
    def quantity_shipped(self):
        """Total accepted pieces on this dispatch, derived from its lines."""
        return sum((l.quantity or 0) for l in self.lines.all())

    @property
    def is_open(self):
        return self.status in self.OPEN_STATUSES

    @property
    def is_delivered(self):
        return self.status == 'delivered' or self.delivery_status == 'delivered'

    @property
    def is_overdue(self):
        """Dispatched but not yet delivered and past the order's due date."""
        if not self.is_open or self.is_delivered:
            return False
        due = self.order.due_date if self.order else None
        return bool(due and due < date.today())

    def __repr__(self):
        return f'<Dispatch {self.dispatch_number}>'


class DispatchLine(db.Model):
    """One packed record's quantity shipped on a dispatch.

    ``packaging_record_id`` is the link back to warehouse stock so the shipped
    quantity is always traceable to a specific packing/lot record.
    """
    __tablename__ = 'dispatch_lines'

    id = db.Column(db.Integer, primary_key=True)
    dispatch_id = db.Column(db.Integer, db.ForeignKey('dispatches.id'),
                            nullable=False, index=True)
    packaging_record_id = db.Column(db.Integer, db.ForeignKey('packaging_records.id'),
                                    index=True)
    lot_id = db.Column(db.Integer, db.ForeignKey('material_lots.id'), index=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), index=True)

    quantity = db.Column(db.Integer, default=0)
    quantity_rejected = db.Column(db.Integer, default=0)
    label_ref = db.Column(db.String(100))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    packaging_record = db.relationship('PackagingRecord',
                                       backref=db.backref('dispatch_lines', lazy='dynamic'))
    product = db.relationship('Product', backref=db.backref('dispatch_lines', lazy='dynamic'))

    @property
    def accepted_quantity(self):
        return max((self.quantity or 0) - (self.quantity_rejected or 0), 0)

    def __repr__(self):
        return f'<DispatchLine dispatch={self.dispatch_id} {self.quantity}>'


# ============================================================
# ALERTS: acknowledgement state for derived operational alerts
# ============================================================
# Alerts themselves are DERIVED from the ledgers by ``compute_alerts()`` so they
# can never go stale or be duplicated. This table stores only the human part:
# whether a named user has acknowledged a specific alert (keyed by the alert's
# deterministic ``alert_key``). When the underlying condition clears the alert
# simply disappears; the acknowledgement row is harmless history.

class AlertAcknowledgement(db.Model):
    """Records that a user has acknowledged a derived alert."""
    __tablename__ = 'alert_acknowledgements'

    id = db.Column(db.Integer, primary_key=True)
    alert_key = db.Column(db.String(200), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    acknowledged_at = db.Column(db.DateTime, default=datetime.utcnow)
    note = db.Column(db.String(300))

    __table_args__ = (
        db.UniqueConstraint('alert_key', 'user_id', name='uq_alert_ack_user'),
    )

    user = db.relationship('User', backref=db.backref('alert_acks', lazy='dynamic'))

    def __repr__(self):
        return f'<AlertAcknowledgement {self.alert_key}>'
