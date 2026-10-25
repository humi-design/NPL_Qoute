from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, TextAreaField, IntegerField, FloatField, DateField, FileField, HiddenField
from wtforms.validators import DataRequired, Email, Length, Optional, NumberRange, ValidationError


# ============ Authentication Forms ============

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')


class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    full_name = StringField('Full Name', validators=[DataRequired(), Length(max=120)])
    role = SelectField('Role', choices=[
        ('admin', 'Administrator'),
        ('sales', 'Sales'),
        ('production', 'Production'),
        ('accounts', 'Accounts'),
        ('read_only', 'Read Only')
    ], validators=[DataRequired()])
    is_active = BooleanField('Active')
    password = PasswordField('Password')
    
    def validate_username(self, field):
        from app.models import User
        user = User.query.filter_by(username=field.data).first()
        if user and (not hasattr(self, 'edit_id') or user.id != self.edit_id):
            raise ValidationError('Username already exists.')


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired()])


# ============ Customer Forms ============

class CustomerForm(FlaskForm):
    customer_code = StringField('Customer Code', validators=[DataRequired(), Length(max=20)])
    company_name = StringField('Company Name', validators=[DataRequired(), Length(max=200)])
    contact_person = StringField('Contact Person', validators=[Length(max=120)])
    email = StringField('Email', validators=[Optional(), Email()])
    phone = StringField('Phone', validators=[Length(max=30)])
    mobile = StringField('Mobile', validators=[Length(max=30)])
    gst_vat = StringField('GST/VAT', validators=[Length(max=30)])
    country = StringField('Country', validators=[Length(max=100)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR - Indian Rupee'),
        ('USD', 'USD - US Dollar'),
        ('EUR', 'EUR - Euro'),
        ('GBP', 'GBP - British Pound'),
        ('AED', 'AED - UAE Dirham')
    ])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    incoterms = StringField('Incoterms', validators=[Length(max=50)])
    shipping_terms = TextAreaField('Shipping Terms', validators=[Optional()])
    billing_address = TextAreaField('Billing Address', validators=[Optional()])
    shipping_address = TextAreaField('Shipping Address', validators=[Optional()])
    delivery_address = TextAreaField('Delivery Address', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


# ============ Product Forms ============

class ProductForm(FlaskForm):
    internal_code = StringField('Internal Code', validators=[DataRequired(), Length(max=30)])
    product_name = StringField('Product Name', validators=[DataRequired(), Length(max=200)])
    customer_part_number = StringField('Customer Part Number', validators=[Length(max=100)])
    drawing_number = StringField('Drawing Number', validators=[Length(max=100)])
    revision = StringField('Revision', validators=[Length(max=10)])
    material = StringField('Material', validators=[Length(max=100)])
    weight = FloatField('Weight (grams)', validators=[Optional()])
    standard = StringField('Standard (DIN/ISO)', validators=[Length(max=50)])
    standard_number = StringField('Standard Number', validators=[Length(max=30)])
    thread = StringField('Thread', validators=[Length(max=50)])
    dimensions = TextAreaField('Dimensions (JSON)', validators=[Optional()])
    description = TextAreaField('Description', validators=[Optional()])
    manufacturing_template_id = SelectField('Manufacturing Template', choices=[('', 'Select Template')], validate_choice=False)
    status = SelectField('Status', choices=[
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('archived', 'Archived')
    ])


class ProductSearchForm(FlaskForm):
    search = StringField('Search', validators=[Length(max=200)])
    material = StringField('Material')
    standard = StringField('Standard')
    thread = StringField('Thread')
    status = SelectField('Status', choices=[
        ('', 'All'),
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('archived', 'Archived')
    ], validators=[Optional()])


# ============ Material Forms ============

class MaterialForm(FlaskForm):
    material_code = StringField('Material Code', validators=[DataRequired(), Length(max=30)])
    material_name = StringField('Material Name', validators=[DataRequired(), Length(max=100)])
    density = FloatField('Density (kg/m³)', validators=[DataRequired(), NumberRange(min=0)])
    grade = StringField('Grade', validators=[Length(max=50)])
    hsn_code = StringField('HSN Code', validators=[Length(max=20)])
    default_supplier = StringField('Default Supplier', validators=[Length(max=100)])
    supplier_id = SelectField('Supplier (master)', coerce=int, validators=[Optional()])
    default_rate = FloatField('Default Rate', validators=[Optional()])
    rate_unit = SelectField('Rate Unit', choices=[
        ('₹/kg', '₹/kg'),
        ('₹/bar', '₹/bar'),
        ('₹/bundle', '₹/bundle'),
        ('₹/piece', '₹/piece')
    ])
    remarks = TextAreaField('Remarks', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


class MaterialRateForm(FlaskForm):
    rate = FloatField('Rate', validators=[DataRequired(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[
        ('₹/kg', '₹/kg'),
        ('₹/bar', '₹/bar'),
        ('₹/bundle', '₹/bundle'),
        ('₹/piece', '₹/piece')
    ])
    supplier = StringField('Supplier', validators=[Length(max=100)])
    effective_date = DateField('Effective Date', validators=[DataRequired()])


# ============ Machine Forms ============

class MachineForm(FlaskForm):
    machine_code = StringField('Machine Code', validators=[DataRequired(), Length(max=30)])
    machine_name = StringField('Machine Name', validators=[DataRequired(), Length(max=100)])
    machine_type = StringField('Machine Type', validators=[Length(max=50)])
    hourly_rate = FloatField('Hourly Rate (₹/hr)', validators=[DataRequired(), NumberRange(min=0)])
    operator_name = StringField('Operator Name', validators=[Length(max=100)])
    power_consumption = FloatField('Power (kW)', validators=[Optional()])
    efficiency = FloatField('Efficiency (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    department = StringField('Department', validators=[Length(max=50)])
    default_setup_time = FloatField('Default Setup Time (min)', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


# ============ Vendor Forms ============

class VendorForm(FlaskForm):
    vendor_code = StringField('Vendor Code', validators=[DataRequired(), Length(max=30)])
    vendor_name = StringField('Vendor Name', validators=[DataRequired(), Length(max=200)])
    contact_person = StringField('Contact Person', validators=[Length(max=100)])
    email = StringField('Email', validators=[Optional(), Email()])
    phone = StringField('Phone', validators=[Length(max=30)])
    mobile = StringField('Mobile', validators=[Length(max=30)])
    gst = StringField('GST', validators=[Length(max=30)])
    address = TextAreaField('Address', validators=[Optional()])
    lead_time = IntegerField('Lead Time (days)', validators=[Optional()])
    default_currency = SelectField('Currency', choices=[
        ('INR', 'INR - Indian Rupee'),
        ('USD', 'USD - US Dollar'),
        ('EUR', 'EUR - Euro'),
        ('GBP', 'GBP - British Pound')
    ])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


class VendorRateForm(FlaskForm):
    service_name = StringField('Service Name', validators=[DataRequired(), Length(max=100)])
    rate = FloatField('Rate', validators=[DataRequired(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[
        ('₹/piece', '₹/piece'),
        ('₹/kg', '₹/kg'),
        ('₹/batch', '₹/batch')
    ])
    min_quantity = IntegerField('Minimum Quantity', validators=[Optional()])
    lead_time_days = IntegerField('Lead Time (days)', validators=[Optional()])
    effective_date = DateField('Effective Date', validators=[DataRequired()])
    notes = TextAreaField('Notes', validators=[Optional()])


class VendorWorkOrderForm(FlaskForm):
    work_order_number = StringField('Work Order Number', validators=[DataRequired(), Length(max=30)])
    vendor_id = SelectField('Vendor', choices=[('', 'Select Vendor')],
                            validate_choice=False, validators=[DataRequired()])
    process_id = SelectField('Process', choices=[('', 'Select Process')], validate_choice=False)
    product_id = SelectField('Product', choices=[('', 'Select Product')], validate_choice=False)

    title = StringField('Title', validators=[Length(max=200)])
    part_description = StringField('Part Description', validators=[Length(max=300)])
    drawing_number = StringField('Drawing Number', validators=[Length(max=100)])
    service_name = StringField('Service', validators=[Length(max=100)])

    quantity_ordered = IntegerField('Quantity Ordered', validators=[Optional(), NumberRange(min=0)])
    quantity_received = IntegerField('Quantity Received', validators=[Optional(), NumberRange(min=0)])
    quantity_rejected = IntegerField('Quantity Rejected', validators=[Optional(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[
        ('pcs', 'pcs'),
        ('kg', 'kg'),
        ('set', 'set'),
        ('batch', 'batch'),
        ('lot', 'lot')
    ])

    rate = FloatField('Rate', validators=[Optional(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'),
        ('USD', 'USD'),
        ('EUR', 'EUR'),
        ('GBP', 'GBP'),
        ('AED', 'AED')
    ])

    order_date = DateField('Order Date', validators=[Optional()])
    due_date = DateField('Due Date', validators=[Optional()])
    completed_date = DateField('Completed Date', validators=[Optional()])

    status = SelectField('Status', choices=[
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled')
    ])
    priority = SelectField('Priority', choices=[
        ('low', 'Low'),
        ('normal', 'Normal'),
        ('high', 'High'),
        ('urgent', 'Urgent')
    ])
    notes = TextAreaField('Notes', validators=[Optional()])


class VendorQualityIssueForm(FlaskForm):
    vendor_id = SelectField('Vendor', choices=[('', 'Select Vendor')], validate_choice=False)
    work_order_id = SelectField('Work Order', choices=[('', 'Not linked')], validate_choice=False)

    issue_date = DateField('Issue Date', validators=[DataRequired()])
    issue_type = SelectField('Issue Type', choices=[
        ('rejection', 'Rejection'),
        ('rework', 'Rework'),
        ('complaint', 'Complaint'),
        ('short_supply', 'Short Supply')
    ])
    severity = SelectField('Severity', choices=[
        ('minor', 'Minor'),
        ('major', 'Major'),
        ('critical', 'Critical')
    ])
    description = TextAreaField('Description', validators=[Optional()])
    quantity_affected = IntegerField('Quantity Affected', validators=[Optional(), NumberRange(min=0)])
    status = SelectField('Status', choices=[
        ('open', 'Open'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('closed', 'Closed')
    ])
    resolution = TextAreaField('Resolution', validators=[Optional()])
    resolved_date = DateField('Resolved Date', validators=[Optional()])


# ============ Process Forms ============

class ProcessForm(FlaskForm):
    process_code = StringField('Process Code', validators=[DataRequired(), Length(max=30)])
    process_name = StringField('Process Name', validators=[DataRequired(), Length(max=100)])
    process_type = SelectField('Process Type', choices=[
        ('internal', 'Internal'),
        ('vendor', 'Vendor')
    ])
    description = TextAreaField('Description', validators=[Optional()])
    machine_id = SelectField('Machine', choices=[('', 'Select Machine')], validate_choice=False)
    vendor_id = SelectField('Vendor', choices=[('', 'Select Vendor')], validate_choice=False)
    department = StringField('Department', validators=[Length(max=50)])
    setup_time = FloatField('Setup Time (min)', validators=[Optional()])
    cycle_time = FloatField('Cycle Time (sec/pc)', validators=[Optional()])
    cost_type = SelectField('Cost Type', choices=[
        ('cycle_time', 'Cycle Time Based'),
        ('per_piece', 'Per Piece'),
        ('per_kg', 'Per Kg'),
        ('per_batch', 'Per Batch'),
        ('percentage', 'Percentage'),
        ('manual', 'Manual Entry'),
        ('formula', 'Formula Based')
    ])
    cost_value = FloatField('Cost Value', validators=[Optional()])
    cost_formula = StringField('Cost Formula', validators=[Optional(), Length(max=200)])
    is_active = BooleanField('Active', default=True)
    is_custom = BooleanField('Custom Process', default=False)


class ProcessTemplateForm(FlaskForm):
    template_code = StringField('Template Code', validators=[DataRequired(), Length(max=30)])
    template_name = StringField('Template Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


# ============ RFQ Forms ============

class RFQForm(FlaskForm):
    rfq_number = StringField('RFQ Number', validators=[DataRequired(), Length(max=30)])
    customer_id = SelectField('Customer', choices=[('', 'Select Customer')], validators=[DataRequired()])
    subject = StringField('Subject', validators=[Length(max=300)])
    description = TextAreaField('Description', validators=[Optional()])
    status = SelectField('Status', choices=[
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('converted', 'Converted'),
        ('lost', 'Lost')
    ])
    priority = SelectField('Priority', choices=[
        ('low', 'Low'),
        ('normal', 'Normal'),
        ('high', 'High'),
        ('urgent', 'Urgent')
    ])
    due_date = DateField('Due Date', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])


class RFQItemForm(FlaskForm):
    product_id = SelectField('Product', choices=[('', 'Select Product')], validate_choice=False)
    part_description = StringField('Part Description', validators=[Length(max=300)])
    drawing_number = StringField('Drawing Number', validators=[Length(max=100)])
    material = StringField('Material', validators=[Length(max=100)])
    quantity = IntegerField('Quantity', validators=[DataRequired()])
    target_price = FloatField('Target Price', validators=[Optional()])
    required_date = DateField('Required Date', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])


# ============ Quotation Forms ============

class QuotationForm(FlaskForm):
    quotation_number = StringField('Quotation Number', validators=[DataRequired(), Length(max=30)])
    quotation_version = StringField('Version', validators=[Length(max=10)])
    customer_id = SelectField('Customer', choices=[('', 'Select Customer')], validators=[DataRequired()])
    quotation_date = DateField('Quotation Date', validators=[DataRequired()])
    valid_until = DateField('Valid Until', validators=[Optional()])
    status = SelectField('Status', default='draft', choices=[
        ('draft', 'Draft'),
        ('sent', 'Sent'),
        ('won', 'Won'),
        ('lost', 'Lost'),
        ('revised', 'Revised')
    ])
    priority = SelectField('Priority', choices=[
        ('low', 'Low'),
        ('normal', 'Normal'),
        ('high', 'High'),
        ('urgent', 'Urgent')
    ])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'),
        ('USD', 'USD'),
        ('EUR', 'EUR'),
        ('GBP', 'GBP'),
        ('AED', 'AED')
    ])
    exchange_rate = FloatField('Exchange Rate', validators=[Optional()])
    billing_address = TextAreaField('Billing Address', validators=[Optional()])
    shipping_address = TextAreaField('Shipping Address', validators=[Optional()])
    payment_terms = TextAreaField('Payment Terms', validators=[Optional()])
    delivery_terms = TextAreaField('Delivery Terms', validators=[Optional()])
    warranty = StringField('Warranty', validators=[Length(max=100)])
    tax_percent = FloatField('Tax (%)', validators=[Optional()])
    internal_notes = TextAreaField('Internal Notes', validators=[Optional()])
    terms_conditions = TextAreaField('Terms & Conditions', validators=[Optional()])


class QuotationItemForm(FlaskForm):
    product_id = SelectField('Product', choices=[('', 'Select Product')], validate_choice=False)
    part_description = StringField('Part Description', validators=[Length(max=300)])
    drawing_number = StringField('Drawing Number', validators=[Length(max=100)])
    material = StringField('Material', validators=[Length(max=100)])
    
    # Quantity breaks
    quantity_1 = IntegerField('Qty 1', validators=[Optional()])
    quantity_2 = IntegerField('Qty 2', validators=[Optional()])
    quantity_3 = IntegerField('Qty 3', validators=[Optional()])
    quantity_4 = IntegerField('Qty 4', validators=[Optional()])
    quantity_5 = IntegerField('Qty 5', validators=[Optional()])
    
    # Prices
    unit_price_1 = FloatField('Price 1', validators=[Optional()])
    unit_price_2 = FloatField('Price 2', validators=[Optional()])
    unit_price_3 = FloatField('Price 3', validators=[Optional()])
    unit_price_4 = FloatField('Price 4', validators=[Optional()])
    unit_price_5 = FloatField('Price 5', validators=[Optional()])
    
    # Tooling
    tooling_cost = FloatField('Tooling Cost', validators=[Optional()])
    tooling_amortization_qty = IntegerField('Tooling Amortization Qty', validators=[Optional()])


# ============ Material Calculator Forms ============

class RawMaterialCalculatorForm(FlaskForm):
    # Stock type
    stock_type = SelectField('Stock Type', choices=[
        ('round_bar', 'Round Bar'),
        ('hex_bar', 'Hex Bar'),
        ('square_bar', 'Square Bar'),
        ('tube', 'Tube'),
        ('pipe', 'Pipe'),
        ('sheet', 'Sheet'),
        ('flat', 'Flat'),
        ('wire', 'Wire'),
        ('coil', 'Coil'),
        ('custom', 'Custom')
    ])
    
    # Dimensions
    finished_length = FloatField('Finished Length (mm)', validators=[DataRequired()])
    diameter = FloatField('Diameter (mm)', validators=[Optional()])
    af = FloatField('AF / Width (mm)', validators=[Optional()])
    side = FloatField('Side (mm)', validators=[Optional()])
    od = FloatField('OD (mm)', validators=[Optional()])
    id = FloatField('ID (mm)', validators=[Optional()])
    thickness = FloatField('Thickness (mm)', validators=[Optional()])
    width = FloatField('Width (mm)', validators=[Optional()])
    height = FloatField('Height (mm)', validators=[Optional()])
    length = FloatField('Length (mm)', validators=[Optional()])
    
    # Material
    material_id = SelectField('Material', choices=[('', 'Select Material')], validate_choice=False)
    custom_density = FloatField('Custom Density (kg/m³)', validators=[Optional()])
    
    # Allowances
    parting = FloatField('Parting (mm)', validators=[Optional()], default=3)
    facing = FloatField('Facing (mm)', validators=[Optional()], default=2)
    machining_allowance = FloatField('Machining Allowance (mm)', validators=[Optional()], default=1)
    grinding_allowance = FloatField('Grinding Allowance (mm)', validators=[Optional()], default=0)
    chamfer_allowance = FloatField('Chamfer Allowance (mm)', validators=[Optional()], default=0.5)
    
    # Custom allowances
    custom_allowance_1_name = StringField('Custom Allowance 1 Name', validators=[Optional()])
    custom_allowance_1_value = FloatField('Custom Allowance 1 Value (mm)', validators=[Optional()])
    custom_allowance_2_name = StringField('Custom Allowance 2 Name', validators=[Optional()])
    custom_allowance_2_value = FloatField('Custom Allowance 2 Value (mm)', validators=[Optional()])
    custom_allowance_3_name = StringField('Custom Allowance 3 Name', validators=[Optional()])
    custom_allowance_3_value = FloatField('Custom Allowance 3 Value (mm)', validators=[Optional()])
    
    # Bar info
    bar_length = FloatField('Bar Length (mm)', validators=[Optional()], default=3000)
    rate_per_kg = FloatField('Rate per kg (₹)', validators=[Optional()])
    
    # Weight override
    override_weight = BooleanField('Override Weight')
    manual_weight = FloatField('Manual Weight (grams)', validators=[Optional()])


# ============ Allowance Form ============

class AllowanceForm(FlaskForm):
    allowance_code = StringField('Allowance Code', validators=[DataRequired(), Length(max=30)])
    allowance_name = StringField('Allowance Name', validators=[DataRequired(), Length(max=100)])
    allowance_type = SelectField('Type', choices=[
        ('machining', 'Machining'),
        ('grinding', 'Grinding'),
        ('facing', 'Facing'),
        ('custom', 'Custom')
    ])
    value = FloatField('Value (mm)', validators=[DataRequired(), NumberRange(min=0)])
    is_active = BooleanField('Active', default=True)


# ============ Settings Forms ============

class CurrencyForm(FlaskForm):
    currency_code = StringField('Currency Code', validators=[DataRequired(), Length(max=10)])
    currency_name = StringField('Currency Name', validators=[DataRequired(), Length(max=50)])
    symbol = StringField('Symbol', validators=[Length(max=10)])
    exchange_rate = FloatField('Exchange Rate', validators=[DataRequired()])
    is_base = BooleanField('Base Currency')
    is_active = BooleanField('Active', default=True)


class SystemSettingForm(FlaskForm):
    setting_key = StringField('Setting Key', validators=[DataRequired(), Length(max=100)])
    setting_value = TextAreaField('Setting Value', validators=[Optional()])
    description = StringField('Description', validators=[Length(max=200)])


# ============ Supplier / Master Data Forms ============

class SupplierTypeForm(FlaskForm):
    type_code = StringField('Type Code', validators=[DataRequired(), Length(max=30)])
    type_name = StringField('Type Name', validators=[DataRequired(), Length(max=100)])
    description = TextAreaField('Description', validators=[Optional()])
    sort_order = IntegerField('Sort Order', validators=[Optional()], default=0)
    is_active = BooleanField('Active', default=True)


class SupplierForm(FlaskForm):
    supplier_code = StringField('Supplier Code', validators=[DataRequired(), Length(max=30)])
    company_name = StringField('Company Name', validators=[DataRequired(), Length(max=200)])
    supplier_type_id = SelectField('Supplier Type', choices=[('', 'Select Type')],
                                   validate_choice=False)

    # Contact
    contact_person = StringField('Contact Person', validators=[Length(max=120)])
    email = StringField('Email', validators=[Optional(), Email()])
    phone = StringField('Phone', validators=[Length(max=30)])
    mobile = StringField('Mobile', validators=[Length(max=30)])
    website = StringField('Website', validators=[Length(max=200)])

    # Address
    address_line1 = StringField('Address Line 1', validators=[Length(max=200)])
    address_line2 = StringField('Address Line 2', validators=[Length(max=200)])
    city = StringField('City', validators=[Length(max=100)])
    state = StringField('State', validators=[Length(max=100)])
    postal_code = StringField('Postal Code', validators=[Length(max=20)])
    country = StringField('Country', validators=[Length(max=100)])

    # Tax / business identifiers
    gst_vat = StringField('GST/VAT', validators=[Length(max=30)])
    pan = StringField('PAN', validators=[Length(max=30)])
    cin = StringField('CIN', validators=[Length(max=30)])
    tax_id = StringField('Tax ID', validators=[Length(max=50)])

    # Commercial terms
    default_currency = SelectField('Currency', choices=[
        ('INR', 'INR - Indian Rupee'),
        ('USD', 'USD - US Dollar'),
        ('EUR', 'EUR - Euro'),
        ('GBP', 'GBP - British Pound'),
        ('AED', 'AED - UAE Dirham')
    ])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    lead_time = IntegerField('Lead Time (days)', validators=[Optional(), NumberRange(min=0)])

    document_references = TextAreaField('Document References', validators=[Optional()])
    notes = TextAreaField('Notes', validators=[Optional()])
    is_active = BooleanField('Active', default=True)


class SupplierMaterialForm(FlaskForm):
    material_id = SelectField('Material', choices=[('', 'Select Material')], validate_choice=False)
    grade = StringField('Grade', validators=[Length(max=50)])
    rate = FloatField('Rate', validators=[Optional(), NumberRange(min=0)])
    rate_unit = SelectField('Rate Unit', choices=[
        ('₹/kg', '₹/kg'),
        ('₹/bar', '₹/bar'),
        ('₹/bundle', '₹/bundle'),
        ('₹/piece', '₹/piece')
    ])
    lead_time = IntegerField('Lead Time (days)', validators=[Optional(), NumberRange(min=0)])
    notes = TextAreaField('Notes', validators=[Optional()])
    is_preferred = BooleanField('Preferred Supplier')


# ============ Procurement / Payables Forms ============

class SupplierPurchaseOrderForm(FlaskForm):
    order_date = DateField('Order Date', validators=[DataRequired()])
    expected_date = DateField('Expected Date', validators=[Optional()])
    status = SelectField('Status', choices=[
        ('draft', 'Draft'),
        ('sent', 'Sent'),
        ('partially_received', 'Partially Received'),
        ('received', 'Received'),
        ('cancelled', 'Cancelled'),
        ('closed', 'Closed'),
    ])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')
    ])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])


class SupplierPurchaseOrderLineForm(FlaskForm):
    material_id = SelectField('Material', choices=[('', 'Select Material')],
                              validate_choice=False)
    description = StringField('Description', validators=[Length(max=300)])
    grade = StringField('Grade', validators=[Length(max=50)])
    quantity_ordered = FloatField('Quantity', validators=[Optional(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[('kg', 'kg'), ('pcs', 'pcs'), ('m', 'm'),
                                        ('bar', 'bar'), ('bundle', 'bundle')])
    rate = FloatField('Rate', validators=[Optional(), NumberRange(min=0)])
    rate_unit = SelectField('Rate Unit', choices=[
        ('₹/kg', '₹/kg'), ('₹/bar', '₹/bar'), ('₹/bundle', '₹/bundle'),
        ('₹/piece', '₹/piece')])
    notes = TextAreaField('Notes', validators=[Optional()])


class SupplierOrderReceiptForm(FlaskForm):
    receipt_date = DateField('Receipt Date', validators=[DataRequired()])
    supplier_document = StringField('Supplier Challan/DC No.', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])


class SupplierReceiptLineForm(FlaskForm):
    """Receipt line capturing quantities, batch/heat/lot and inspection."""
    po_line_id = SelectField('PO Line', choices=[('', 'Select Line')],
                             validate_choice=False)
    quantity_received = FloatField('Received Qty', validators=[Optional(), NumberRange(min=0)])
    quantity_rejected = FloatField('Rejected Qty', validators=[Optional(), NumberRange(min=0)])
    batch_number = StringField('Batch No.', validators=[Length(max=60)])
    heat_number = StringField('Heat No.', validators=[Length(max=60)])
    lot_number = StringField('Lot No.', validators=[Length(max=60)])
    certificate_ref = StringField('Certificate Ref.', validators=[Length(max=120)])
    inspection_status = SelectField('Inspection', choices=[
        ('pending', 'Pending'), ('passed', 'Passed'),
        ('failed', 'Failed'), ('partial', 'Partial')])
    inspection_notes = TextAreaField('Inspection Notes', validators=[Optional()])


class SupplierInvoiceForm(FlaskForm):
    invoice_number = StringField('Invoice Number', validators=[DataRequired(), Length(max=60)])
    po_id = SelectField('Purchase Order', choices=[('', 'Not linked')], validate_choice=False)
    invoice_date = DateField('Invoice Date', validators=[DataRequired()])
    due_date = DateField('Due Date', validators=[Optional()])
    subtotal = FloatField('Subtotal', validators=[Optional(), NumberRange(min=0)])
    tax_amount = FloatField('Tax', validators=[Optional(), NumberRange(min=0)])
    total_amount = FloatField('Total', validators=[DataRequired(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')])
    status = SelectField('Status', choices=[
        ('draft', 'Draft'), ('posted', 'Posted'), ('cancelled', 'Cancelled')])
    notes = TextAreaField('Notes', validators=[Optional()])


class SupplierPaymentForm(FlaskForm):
    payment_date = DateField('Payment Date', validators=[DataRequired()])
    amount = FloatField('Amount', validators=[DataRequired(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')])
    method = SelectField('Method', choices=[
        ('', 'Select'), ('bank', 'Bank Transfer'), ('upi', 'UPI'),
        ('cheque', 'Cheque'), ('cash', 'Cash')], validators=[Optional()])
    reference = StringField('Reference', validators=[Length(max=100)])
    is_advance = BooleanField('Advance / Credit')
    notes = TextAreaField('Notes', validators=[Optional()])


class PaymentAllocationForm(FlaskForm):
    """Applies a payment to an invoice (one row per invoice)."""
    invoice_id = SelectField('Invoice', choices=[('', 'Select Invoice')],
                             validate_choice=False)
    amount = FloatField('Amount', validators=[DataRequired(), NumberRange(min=0.01)])


class VendorInvoiceForm(FlaskForm):
    invoice_number = StringField('Invoice Number', validators=[DataRequired(), Length(max=60)])
    work_order_id = SelectField('Work Order', choices=[('', 'Not linked')],
                                validate_choice=False)
    invoice_date = DateField('Invoice Date', validators=[DataRequired()])
    due_date = DateField('Due Date', validators=[Optional()])
    subtotal = FloatField('Subtotal', validators=[Optional(), NumberRange(min=0)])
    tax_amount = FloatField('Tax', validators=[Optional(), NumberRange(min=0)])
    total_amount = FloatField('Total', validators=[DataRequired(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')])
    status = SelectField('Status', choices=[
        ('draft', 'Draft'), ('posted', 'Posted'), ('cancelled', 'Cancelled')])
    notes = TextAreaField('Notes', validators=[Optional()])


class VendorPaymentForm(FlaskForm):
    payment_date = DateField('Payment Date', validators=[DataRequired()])
    amount = FloatField('Amount', validators=[DataRequired(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')])
    method = SelectField('Method', choices=[
        ('', 'Select'), ('bank', 'Bank Transfer'), ('upi', 'UPI'),
        ('cheque', 'Cheque'), ('cash', 'Cash')], validators=[Optional()])
    reference = StringField('Reference', validators=[Length(max=100)])
    is_advance = BooleanField('Advance / Credit')
    notes = TextAreaField('Notes', validators=[Optional()])



# ============ Orders & Manufacturing Tracking Forms ============

class OrderForm(FlaskForm):
    customer_id = SelectField('Customer', choices=[('', 'Select Customer')],
                              validate_choice=False, validators=[DataRequired()])
    quotation_id = SelectField('Quotation', choices=[('', 'Not linked')],
                               validate_choice=False)
    product_id = SelectField('Product', choices=[('', 'Not linked')],
                             validate_choice=False)

    part_description = StringField('Part Description', validators=[Length(max=300)])
    drawing_number = StringField('Drawing Number', validators=[Length(max=100)])
    revision = StringField('Revision', validators=[Length(max=10)])
    specifications = TextAreaField('Specifications', validators=[Optional()])
    material_grade = StringField('Material Grade', validators=[Length(max=100)])
    standard = StringField('Standard', validators=[Length(max=50)])

    quantity_ordered = IntegerField('Quantity Ordered',
                                    validators=[DataRequired(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[
        ('pcs', 'pcs'), ('kg', 'kg'), ('set', 'set'), ('batch', 'batch')])
    rate = FloatField('Rate', validators=[Optional(), NumberRange(min=0)])
    currency = SelectField('Currency', choices=[
        ('INR', 'INR'), ('USD', 'USD'), ('EUR', 'EUR'), ('GBP', 'GBP'), ('AED', 'AED')])

    order_date = DateField('Order Date', validators=[DataRequired()])
    due_date = DateField('Due Date', validators=[Optional()])
    planned_start = DateField('Planned Start', validators=[Optional()])
    actual_start = DateField('Actual Start', validators=[Optional()])
    actual_completion = DateField('Actual Completion', validators=[Optional()])

    status = SelectField('Status', choices=[
        ('draft', 'Draft'), ('confirmed', 'Confirmed'),
        ('in_production', 'In Production'), ('completed', 'Completed'),
        ('on_hold', 'On Hold'), ('cancelled', 'Cancelled')])
    priority = SelectField('Priority', choices=[
        ('low', 'Low'), ('normal', 'Normal'), ('high', 'High'), ('urgent', 'Urgent')])
    notes = TextAreaField('Notes', validators=[Optional()])


class OrderOperationForm(FlaskForm):
    sequence = IntegerField('Sequence', validators=[DataRequired(), NumberRange(min=1)])
    process_id = SelectField('Process', choices=[('', 'Not linked')], validate_choice=False)
    operation_name = StringField('Operation Name',
                                 validators=[DataRequired(), Length(max=100)])
    is_vendor = BooleanField('Subcontracted to a vendor')
    vendor_id = SelectField('Vendor', choices=[('', 'Not linked')], validate_choice=False)
    machine_id = SelectField('Machine', choices=[('', 'Not linked')], validate_choice=False)

    quantity_planned = IntegerField('Planned Qty', validators=[Optional(), NumberRange(min=0)])
    planned_issue_date = DateField('Planned Issue Date', validators=[Optional()])
    due_date = DateField('Due Date', validators=[Optional()])
    rate = FloatField('Rate', validators=[Optional(), NumberRange(min=0)])

    actual_start = DateField('Actual Start', validators=[Optional()])
    actual_end = DateField('Actual End', validators=[Optional()])

    status = SelectField('Status', choices=[
        ('pending', 'Pending'), ('in_progress', 'In Progress'),
        ('completed', 'Completed'), ('cancelled', 'Cancelled')])
    is_final = BooleanField('Final operation (feeds finished stock)')
    notes = TextAreaField('Notes', validators=[Optional()])


class OperationTransactionForm(FlaskForm):
    transaction_type = SelectField('Movement', choices=[
        ('issue', 'Issue'), ('return', 'Return'), ('accept', 'Accept'),
        ('reject', 'Reject'), ('rework', 'Rework'), ('scrap', 'Scrap')])
    transaction_date = DateField('Date', validators=[DataRequired()])
    quantity = IntegerField('Quantity', validators=[DataRequired(), NumberRange(min=1)])
    reference = StringField('Challan / DC Ref', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])


class MaterialLotForm(FlaskForm):
    lot_number = StringField('Lot Number', validators=[DataRequired(), Length(max=60)])
    material_id = SelectField('Material', choices=[('', 'Not linked')], validate_choice=False)
    order_id = SelectField('Order', choices=[('', 'Not linked')], validate_choice=False)
    supplier_id = SelectField('Supplier', choices=[('', 'Not linked')], validate_choice=False)
    po_id = SelectField('Purchase Order', choices=[('', 'Not linked')], validate_choice=False)
    receipt_id = SelectField('Goods Receipt', choices=[('', 'Not linked')],
                             validate_choice=False)

    grade = StringField('Grade', validators=[Length(max=100)])
    specification = StringField('Specification', validators=[Length(max=200)])
    heat_number = StringField('Heat No.', validators=[Length(max=60)])
    batch_number = StringField('Batch No.', validators=[Length(max=60)])
    certificate_ref = StringField('Certificate Ref.', validators=[Length(max=120)])

    quantity_received = FloatField('Quantity Received',
                                   validators=[DataRequired(), NumberRange(min=0)])
    unit = SelectField('Unit', choices=[
        ('kg', 'kg'), ('pcs', 'pcs'), ('m', 'm'), ('bar', 'bar'), ('bundle', 'bundle')])
    receipt_date = DateField('Receipt Date', validators=[Optional()])


class MaterialMovementForm(FlaskForm):
    movement_type = SelectField('Movement', choices=[
        ('allocation', 'Allocate (reserve)'), ('issue', 'Issue'),
        ('return', 'Return'), ('scrap', 'Scrap'), ('shortage', 'Shortage'),
        ('adjustment', 'Adjustment')])
    movement_date = DateField('Date', validators=[DataRequired()])
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0.001)])
    order_id = SelectField('Order', choices=[('', 'Not linked')], validate_choice=False)
    order_operation_id = SelectField('Operation', choices=[('', 'Not linked')],
                                     validate_choice=False)
    reference = StringField('Reference', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])


class InspectionForm(FlaskForm):
    inspection_type = SelectField('Inspection Type', choices=[
        ('incoming', 'Incoming'), ('in_process', 'In-Process'), ('final', 'Final')])
    inspection_date = DateField('Inspection Date', validators=[DataRequired()])
    order_operation_id = SelectField('Operation', choices=[('', 'Not linked')],
                                     validate_choice=False)
    lot_id = SelectField('Material Lot', choices=[('', 'Not linked')],
                         validate_choice=False)

    acceptance_criteria = TextAreaField('Acceptance Criteria', validators=[Optional()])
    tolerance = StringField('Tolerance', validators=[Length(max=100)])
    measurements = TextAreaField('Measurements', validators=[Optional()])
    sample_size = IntegerField('Sample Size', validators=[Optional(), NumberRange(min=0)])

    quantity_inspected = IntegerField('Qty Inspected', validators=[Optional(), NumberRange(min=0)])
    quantity_accepted = IntegerField('Qty Accepted', validators=[Optional(), NumberRange(min=0)])
    quantity_rejected = IntegerField('Qty Rejected', validators=[Optional(), NumberRange(min=0)])
    quantity_rework = IntegerField('Qty Rework', validators=[Optional(), NumberRange(min=0)])

    result = SelectField('Result', choices=[
        ('pending', 'Pending'), ('passed', 'Passed'),
        ('failed', 'Failed'), ('partial', 'Partial')])
    notes = TextAreaField('Notes', validators=[Optional()])


class PackagingRecordForm(FlaskForm):
    packaging_date = DateField('Packaging Date', validators=[DataRequired()])
    packaging_type = SelectField('Packaging Type', choices=[
        ('box', 'Box'), ('carton', 'Carton'), ('pallet', 'Pallet'),
        ('bag', 'Bag'), ('bundle', 'Bundle')], validators=[Optional()])
    quantity_packed = IntegerField('Qty Packed', validators=[Optional(), NumberRange(min=0)])
    quantity_accepted = IntegerField('Qty Accepted',
                                     validators=[DataRequired(), NumberRange(min=0)])
    quantity_rejected = IntegerField('Qty Rejected', validators=[Optional(), NumberRange(min=0)])
    label_ref = StringField('Label Ref', validators=[Length(max=100)])

    warehouse_receipt_number = StringField('Warehouse Receipt No.', validators=[Length(max=50)])
    warehouse_receipt_date = DateField('Warehouse Receipt Date', validators=[Optional()])
    location = StringField('Location', validators=[Length(max=100)])

    status = SelectField('Status', choices=[
        ('packed', 'Packed'), ('warehoused', 'Warehoused'), ('dispatched', 'Dispatched')])
    notes = TextAreaField('Notes', validators=[Optional()])



# ============ Inventory: locations, transfers, adjustments ============

class StockLocationForm(FlaskForm):
    code = StringField('Code', validators=[DataRequired(), Length(max=30)])
    name = StringField('Name', validators=[DataRequired(), Length(max=120)])
    kind = SelectField('Location Type', choices=[
        ('raw_material', 'Raw Material Store'),
        ('wip', 'Work in Progress'),
        ('subcontractor', 'Subcontractor'),
        ('finished_goods', 'Finished Goods'),
        ('rejected', 'Rejected Stock'),
        ('scrap', 'Scrap'),
        ('other', 'Other')], validators=[DataRequired()])
    vendor_id = SelectField('Vendor (for subcontractor sites)',
                            choices=[('', 'Not linked')], validate_choice=False)
    address = StringField('Address', validators=[Length(max=300)])
    is_active = BooleanField('Active', default=True)
    notes = TextAreaField('Notes', validators=[Optional()])


class StockTransferForm(FlaskForm):
    transfer_date = DateField('Transfer Date', validators=[DataRequired()])
    from_location_id = SelectField('From Location', choices=[], validate_choice=False)
    to_location_id = SelectField('To Location', choices=[], validate_choice=False)
    material_lot_id = SelectField('Material Lot', choices=[('', 'Not linked')],
                                  validate_choice=False)
    order_id = SelectField('Order', choices=[('', 'Not linked')], validate_choice=False)
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0.001)])
    unit = StringField('Unit', validators=[Length(max=20)])
    status = SelectField('Status', choices=[
        ('pending', 'Pending'), ('in_transit', 'In Transit'),
        ('received', 'Received'), ('cancelled', 'Cancelled')])
    reference = StringField('Challan / Reference', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])

    def validate_to_location_id(self, field):
        if field.data and field.data == self.from_location_id.data:
            raise ValidationError('From and To locations must differ.')


class StockAdjustmentForm(FlaskForm):
    adjustment_date = DateField('Adjustment Date', validators=[DataRequired()])
    material_lot_id = SelectField('Material Lot', choices=[], validate_choice=False)
    location_id = SelectField('Location', choices=[('', 'Not linked')],
                              validate_choice=False)
    order_id = SelectField('Order', choices=[('', 'Not linked')], validate_choice=False)
    adjustment_type = SelectField('Adjustment Type', choices=[
        ('increase', 'Increase'), ('decrease', 'Decrease'),
        ('write_off', 'Write Off'), ('reversal', 'Reversal'),
        ('correction', 'Correction')], validators=[DataRequired()])
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0.001)])
    unit = StringField('Unit', validators=[Length(max=20)])
    reason = StringField('Reason', validators=[DataRequired(), Length(max=300)])
    reference = StringField('Reference', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])


# ============ Costing: actual cost entries ============

class OrderCostEntryForm(FlaskForm):
    category = SelectField('Cost Category', choices=[
        ('material', 'Material'), ('subcontracting', 'Subcontracting / Job Work'),
        ('machining', 'Machining'), ('finishing', 'Finishing'),
        ('inspection', 'Inspection'), ('packaging', 'Packaging'),
        ('freight', 'Freight'), ('other', 'Other')], validators=[DataRequired()])
    amount = FloatField('Amount', validators=[DataRequired()])
    currency = StringField('Currency', validators=[Length(max=10)])
    entry_date = DateField('Date', validators=[DataRequired()])
    description = StringField('Description', validators=[Length(max=300)])
    order_operation_id = SelectField('Operation', choices=[('', 'Not linked')],
                                     validate_choice=False)
    vendor_id = SelectField('Vendor', choices=[('', 'Not linked')], validate_choice=False)
    supplier_id = SelectField('Supplier', choices=[('', 'Not linked')],
                              validate_choice=False)
    notes = TextAreaField('Notes', validators=[Optional()])


# ============ Procurement: supplier RFQs & quotes ============

class SupplierRFQForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired(), Length(max=200)])
    material_id = SelectField('Material', choices=[('', 'Select Material')],
                              validate_choice=False)
    order_id = SelectField('Order', choices=[('', 'Not linked')], validate_choice=False)
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0.001)])
    unit = StringField('Unit', validators=[Length(max=20)])
    grade = StringField('Grade', validators=[Length(max=50)])
    rfq_date = DateField('RFQ Date', validators=[DataRequired()])
    due_date = DateField('Quote Due Date', validators=[Optional()])
    required_date = DateField('Required Date', validators=[Optional()])
    delivery_terms = StringField('Delivery Terms', validators=[Length(max=200)])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    currency = StringField('Currency', validators=[Length(max=10)])
    status = SelectField('Status', choices=[
        ('draft', 'Draft'), ('sent', 'Sent'), ('quoted', 'Quoted'),
        ('compared', 'Compared'), ('awarded', 'Awarded'), ('cancelled', 'Cancelled')])
    notes = TextAreaField('Notes', validators=[Optional()])


class SupplierQuoteForm(FlaskForm):
    supplier_id = SelectField('Supplier', choices=[('', 'Select Supplier')],
                              validate_choice=False)
    quote_reference = StringField('Supplier Quote Ref', validators=[Length(max=60)])
    quote_date = DateField('Quote Date', validators=[DataRequired()])
    valid_until = DateField('Valid Until', validators=[Optional()])
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0)])
    unit = StringField('Unit', validators=[Length(max=20)])
    unit_price = FloatField('Unit Price', validators=[DataRequired(), NumberRange(min=0)])
    currency = StringField('Currency', validators=[Length(max=10)])
    tax_percent = FloatField('Tax %', validators=[Optional(), NumberRange(min=0, max=100)])
    lead_time_days = IntegerField('Lead Time (days)', validators=[Optional(), NumberRange(min=0)])
    payment_terms = StringField('Payment Terms', validators=[Length(max=100)])
    delivery_terms = StringField('Delivery Terms', validators=[Length(max=200)])
    quality_rating = FloatField('Quality Rating (0-100)',
                                validators=[Optional(), NumberRange(min=0, max=100)])
    status = SelectField('Status', choices=[
        ('received', 'Received'), ('shortlisted', 'Shortlisted'),
        ('rejected', 'Rejected'), ('selected', 'Selected')])
    notes = TextAreaField('Notes', validators=[Optional()])


# ============ Documents & traceability ============

class DocumentForm(FlaskForm):
    """Upload or edit a controlled document linked to a record."""
    entity_type = SelectField('Linked To', choices=[
        ('product', 'Product / Part'),
        ('order', 'Sales Order'),
        ('lot', 'Material Lot / Batch'),
        ('inspection', 'Inspection'),
        ('dispatch', 'Dispatch'),
        ('supplier', 'Supplier'),
        ('vendor', 'Vendor'),
        ('customer', 'Customer'),
    ])
    entity_id = IntegerField('Record ID', validators=[Optional(), NumberRange(min=1)])
    title = StringField('Title', validators=[Length(max=200)])
    category = SelectField('Category', choices=[
        ('drawing', 'Drawing'),
        ('revision', 'Revision'),
        ('specification', 'Specification'),
        ('material_certificate', 'Material Certificate'),
        ('test_certificate', 'Test Certificate'),
        ('inspection_report', 'Inspection Report'),
        ('supplier_document', 'Supplier Document'),
        ('purchase_order', 'Purchase Order'),
        ('packing_list', 'Packing List'),
        ('invoice', 'Invoice'),
        ('other', 'Other'),
    ])
    revision = StringField('Revision', validators=[Length(max=20)])
    issue_date = DateField('Issue Date', validators=[Optional()])
    is_current = BooleanField('Current revision', default=True)
    notes = TextAreaField('Notes', validators=[Optional()])
    file = FileField('File')


class DocumentSearchForm(FlaskForm):
    """Filter form for the document register (GET)."""
    q = StringField('Search', validators=[Optional(), Length(max=100)])
    category = SelectField('Category', choices=[('', 'All categories')], validate_choice=False)
    entity_type = SelectField('Linked To', choices=[('', 'All records')], validate_choice=False)
    current_only = BooleanField('Current revisions only', default=True)


# ============ Dispatch ============

class DispatchForm(FlaskForm):
    order_id = SelectField('Sales Order', choices=[], coerce=int,
                           validate_choice=False)
    dispatch_date = DateField('Dispatch Date', validators=[DataRequired()])
    mode = SelectField('Mode', choices=[
        ('surface', 'Surface'), ('air', 'Air'), ('sea', 'Sea'),
        ('rail', 'Rail'), ('courier', 'Courier'),
        ('hand_delivery', 'Hand Delivery')])
    transporter = StringField('Transporter', validators=[Length(max=150)])
    vehicle_number = StringField('Vehicle Number', validators=[Length(max=50)])
    shipment_reference = StringField('Shipment Ref (LR/Docket/AWB)',
                                     validators=[Length(max=100)])
    eway_bill_number = StringField('E-Way Bill No.', validators=[Length(max=50)])
    invoice_number = StringField('Invoice No.', validators=[Length(max=60)])
    invoice_date = DateField('Invoice Date', validators=[Optional()])
    packing_list_ref = StringField('Packing List Ref', validators=[Length(max=60)])
    destination = StringField('Destination', validators=[Length(max=200)])
    status = SelectField('Status', choices=[
        ('prepared', 'Prepared'), ('dispatched', 'Dispatched'),
        ('in_transit', 'In Transit'), ('delivered', 'Delivered'),
        ('returned', 'Returned'), ('cancelled', 'Cancelled')])
    delivery_status = SelectField('Delivery Status', choices=[
        ('pending', 'Pending'), ('partial', 'Partial'), ('delivered', 'Delivered')])
    delivered_date = DateField('Delivered Date', validators=[Optional()])
    pod_reference = StringField('Proof of Delivery Ref', validators=[Length(max=100)])
    receiver_name = StringField('Received By', validators=[Length(max=120)])
    remarks = TextAreaField('Remarks', validators=[Optional()])


class DispatchLineForm(FlaskForm):
    packaging_record_id = SelectField('Packing Record', choices=[], coerce=int,
                                      validate_choice=False)
    quantity = IntegerField('Quantity', validators=[DataRequired(), NumberRange(min=1)])
    quantity_rejected = IntegerField('Qty Rejected',
                                     validators=[Optional(), NumberRange(min=0)])
    label_ref = StringField('Label Ref', validators=[Length(max=100)])
    notes = TextAreaField('Notes', validators=[Optional()])
