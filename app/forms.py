from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, TextAreaField, IntegerField, FloatField, DateField, FileField, HiddenField
from wtforms.validators import DataRequired, Email, Length, Optional, NumberRange, ValidationError
from wtforms_sqlalchemy.fields import QuerySelectField
from app.models import User, Customer, Product, Material, Machine, Vendor, Process, ProcessTemplate


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
    manufacturing_template_id = QuerySelectField('Manufacturing Template',
        query_factory=lambda: ProcessTemplate.query.filter_by(is_active=True).all(),
        get_label='template_name', allow_blank=True)
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


# ============ Process Forms ============

class ProcessForm(FlaskForm):
    process_code = StringField('Process Code', validators=[DataRequired(), Length(max=30)])
    process_name = StringField('Process Name', validators=[DataRequired(), Length(max=100)])
    process_type = SelectField('Process Type', choices=[
        ('internal', 'Internal'),
        ('vendor', 'Vendor')
    ])
    description = TextAreaField('Description', validators=[Optional()])
    machine_id = QuerySelectField('Machine',
        query_factory=lambda: Machine.query.filter_by(is_active=True).all(),
        get_label='machine_name', allow_blank=True)
    vendor_id = QuerySelectField('Vendor',
        query_factory=lambda: Vendor.query.filter_by(is_active=True).all(),
        get_label='vendor_name', allow_blank=True)
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
    customer_id = QuerySelectField('Customer',
        query_factory=lambda: Customer.query.filter_by(is_active=True).all(),
        get_label='company_name', validators=[DataRequired()])
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
    product_id = QuerySelectField('Product',
        query_factory=lambda: Product.query.filter_by(status='active').all(),
        get_label='product_name', allow_blank=True)
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
    customer_id = QuerySelectField('Customer',
        query_factory=lambda: Customer.query.filter_by(is_active=True).all(),
        get_label='company_name', validators=[DataRequired()])
    quotation_date = DateField('Quotation Date', validators=[DataRequired()])
    valid_until = DateField('Valid Until', validators=[Optional()])
    status = SelectField('Status', choices=[
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
    product_id = QuerySelectField('Product',
        query_factory=lambda: Product.query.filter_by(status='active').all(),
        get_label='product_name', allow_blank=True, description='Link to Product Library')
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
    material_id = QuerySelectField('Material',
        query_factory=lambda: Material.query.filter_by(is_active=True).all(),
        get_label='material_name', allow_blank=True)
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
