import os
import uuid
import pandas as pd
from datetime import datetime
from io import BytesIO
from werkzeug.utils import secure_filename
from flask import current_app, send_file

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'step', 'stp', 'iges', 'igs', 'dxf', 'dwg', 'zip'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_file(file, entity_type, entity_id):
    """Upload a file and return the file path."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        # Create directory if not exists
        upload_dir = os.path.join('uploads', entity_type, str(entity_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        
        return {
            'file_name': unique_filename,
            'original_name': filename,
            'file_path': file_path,
            'file_type': ext,
            'file_size': os.path.getsize(file_path)
        }
    return None


def delete_file(file_path):
    """Delete a file."""
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def get_file_extension(filename):
    """Get file extension."""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def is_image_file(filename):
    """Check if file is an image."""
    ext = get_file_extension(filename)
    return ext in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def is_pdf_file(filename):
    """Check if file is a PDF."""
    return get_file_extension(filename) == 'pdf'


def generate_quotation_number(prefix='NP'):
    """Generate unique quotation number."""
    from app.models import Quotation, db
    year = datetime.now().year
    last_quote = Quotation.query.filter(
        Quotation.quotation_number.like(f'{prefix}-{year}-%')
    ).order_by(Quotation.id.desc()).first()
    
    if last_quote:
        try:
            last_num = int(last_quote.quotation_number.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{year}-{new_num:06d}"


def generate_rfq_number(prefix='RFQ'):
    """Generate unique RFQ number."""
    from app.models import RFQ, db
    year = datetime.now().year
    last_rfq = RFQ.query.filter(
        RFQ.rfq_number.like(f'{prefix}-{year}-%')
    ).order_by(RFQ.id.desc()).first()
    
    if last_rfq:
        try:
            last_num = int(last_rfq.rfq_number.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{year}-{new_num:06d}"


def generate_product_code(prefix='PRD'):
    """Generate unique product code."""
    from app.models import Product, db
    last_product = Product.query.order_by(Product.id.desc()).first()
    
    if last_product:
        try:
            last_num = int(last_product.internal_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:05d}"


def generate_customer_code(prefix='CUST'):
    """Generate unique customer code."""
    from app.models import Customer, db
    last_customer = Customer.query.order_by(Customer.id.desc()).first()
    
    if last_customer:
        try:
            last_num = int(last_customer.customer_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:04d}"


def generate_vendor_code(prefix='VND'):
    """Generate unique vendor code."""
    from app.models import Vendor, db
    last_vendor = Vendor.query.order_by(Vendor.id.desc()).first()
    
    if last_vendor:
        try:
            last_num = int(last_vendor.vendor_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:04d}"


def generate_machine_code(prefix='MCH'):
    """Generate unique machine code."""
    from app.models import Machine, db
    last_machine = Machine.query.order_by(Machine.id.desc()).first()
    
    if last_machine:
        try:
            last_num = int(last_machine.machine_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:03d}"


def generate_material_code(prefix='MAT'):
    """Generate unique material code."""
    from app.models import Material, db
    last_material = Material.query.order_by(Material.id.desc()).first()
    
    if last_material:
        try:
            last_num = int(last_material.material_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:04d}"


def generate_process_code(prefix='PRC'):
    """Generate unique process code."""
    from app.models import Process, db
    last_process = Process.query.order_by(Process.id.desc()).first()
    
    if last_process:
        try:
            last_num = int(last_process.process_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:04d}"


def generate_template_code(prefix='TMP'):
    """Generate unique template code."""
    from app.models import ProcessTemplate, db
    last_template = ProcessTemplate.query.order_by(ProcessTemplate.id.desc()).first()
    
    if last_template:
        try:
            last_num = int(last_template.template_code.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{new_num:04d}"


# ============ Material Calculator ============

def calculate_blank_length(finished_length, parting=3, facing=2, machining_allowance=1, 
                           grinding_allowance=0.5, chamfer_allowance=0.5, custom_allowances=None):
    """
    Calculate blank length from finished dimensions.
    
    Args:
        finished_length: Finished part length in mm
        parting: Parting allowance in mm
        facing: Facing allowance (both ends)
        machining_allowance: General machining allowance
        grinding_allowance: Grinding allowance
        chamfer_allowance: Chamfer allowance
        custom_allowances: Dict of custom allowance name -> value
    """
    total_allowance = parting + facing + machining_allowance + grinding_allowance + chamfer_allowance
    
    if custom_allowances:
        total_allowance += sum(custom_allowances.values())
    
    return finished_length + total_allowance


def calculate_material_weight(volume_mm3, density_kg_m3):
    """
    Calculate material weight from volume and density.
    
    Args:
        volume_mm3: Volume in cubic mm
        density_kg_m3: Density in kg/m³
    
    Returns:
        Weight in grams
    """
    volume_m3 = volume_mm3 / 1e9  # Convert mm³ to m³
    weight_kg = volume_m3 * density_kg_m3
    return weight_kg * 1000  # Convert to grams


def calculate_round_bar_volume(diameter_mm, length_mm):
    """Calculate volume of round bar in mm³."""
    import math
    radius = diameter_mm / 2
    return math.pi * radius ** 2 * length_mm


def calculate_hex_bar_volume(af_mm, length_mm):
    """Calculate volume of hexagonal bar in mm³."""
    import math
    # Area of hexagon = (3√3/2) * (AF/2)²
    area = (3 * math.sqrt(3) / 2) * (af_mm / 2) ** 2
    return area * length_mm


def calculate_square_bar_volume(side_mm, length_mm):
    """Calculate volume of square bar in mm³."""
    return side_mm ** 2 * length_mm


def calculate_tube_volume(od_mm, id_mm, length_mm):
    """Calculate volume of tube/pipe in mm³."""
    import math
    outer_area = math.pi * (od_mm / 2) ** 2
    inner_area = math.pi * (id_mm / 2) ** 2 if id_mm > 0 else 0
    return (outer_area - inner_area) * length_mm


def calculate_sheet_volume(width_mm, thickness_mm, length_mm):
    """Calculate volume of sheet/flat in mm³."""
    return width_mm * thickness_mm * length_mm


def calculate_pieces_per_bar(blank_length_mm, bar_length_mm=3000):
    """Calculate how many pieces can be made from one bar."""
    return int(bar_length_mm / blank_length_mm) if blank_length_mm > 0 else 0


def calculate_waste_percentage(blank_length_mm, bar_length_mm=3000):
    """Calculate material waste percentage."""
    pieces = calculate_pieces_per_bar(blank_length_mm, bar_length_mm)
    if pieces == 0:
        return 100
    used_length = pieces * blank_length_mm
    waste = bar_length_mm - used_length
    return (waste / bar_length_mm) * 100


def calculate_utilization(blank_length_mm, bar_length_mm=3000):
    """Calculate material utilization percentage."""
    return 100 - calculate_waste_percentage(blank_length_mm, bar_length_mm)


def calculate_material_cost(weight_grams, rate_per_kg, rate_unit='₹/kg'):
    """
    Calculate material cost based on weight and rate.
    
    Args:
        weight_grams: Weight in grams
        rate_per_kg: Rate per kg
        rate_unit: Unit of rate (₹/kg, ₹/bar, ₹/bundle, ₹/piece)
    """
    weight_kg = weight_grams / 1000
    
    if rate_unit == '₹/kg':
        return weight_kg * rate_per_kg
    elif rate_unit == '₹/piece':
        return rate_per_kg  # Rate is per piece
    else:
        return weight_kg * rate_per_kg  # Default to per kg


# ============ Costing Engine ============

def calculate_operation_cost(process, quantity, machine_rate=None, vendor_rate=None):
    """
    Calculate cost for a single operation.
    
    Args:
        process: Process object with cost parameters
        quantity: Number of pieces
        machine_rate: Machine hourly rate (for internal operations)
        vendor_rate: Vendor rate (for vendor operations)
    """
    cost = 0
    
    if process.cost_type == 'cycle_time':
        if machine_rate:
            cycle_time_seconds = process.cycle_time
            setup_time_hours = process.setup_time / 60
            cycle_time_hours = (cycle_time_seconds * quantity) / 3600
            cost = machine_rate * (setup_time_hours + cycle_time_hours)
        cost = (process.cost_value or 0) * quantity
    
    elif process.cost_type == 'per_piece':
        cost = (process.cost_value or 0) * quantity
    
    elif process.cost_type == 'per_kg':
        # Requires weight - handled separately
        cost = process.cost_value or 0
    
    elif process.cost_type == 'per_batch':
        cost = process.cost_value or 0
    
    elif process.cost_type == 'percentage':
        # Percentage of material cost - handled separately
        cost = process.cost_value or 0
    
    elif process.cost_type == 'manual':
        cost = process.cost_value or 0
    
    elif process.cost_type == 'formula':
        # Evaluate formula - requires variables
        cost = process.cost_value or 0
    
    return cost


def calculate_tooling_cost(tool_cost, tool_life, quantity):
    """Calculate tooling cost per piece."""
    if tool_life > 0 and quantity > 0:
        return tool_cost / min(tool_life, quantity)
    return tool_cost / max(quantity, 1)


def calculate_total_cost(material_cost, machining_cost, vendor_cost, tooling_cost, 
                         overhead_percent, profit_percent, scrap_percent=0):
    """Calculate total costing including overheads and profit."""
    # Adjust for scrap
    adjusted_material = material_cost / (1 - scrap_percent / 100) if scrap_percent > 0 else material_cost
    
    # Direct costs
    direct_cost = adjusted_material + machining_cost + vendor_cost + tooling_cost
    
    # Overheads
    overhead_amount = direct_cost * (overhead_percent / 100)
    
    # Subtotal before profit
    subtotal = direct_cost + overhead_amount
    
    # Profit
    profit_amount = subtotal * (profit_percent / 100)
    
    # Final price
    total = subtotal + profit_amount
    
    # Margin
    margin = (total - direct_cost) / total * 100 if total > 0 else 0
    
    return {
        'material_cost': material_cost,
        'adjusted_material': adjusted_material,
        'machining_cost': machining_cost,
        'vendor_cost': vendor_cost,
        'tooling_cost': tooling_cost,
        'direct_cost': direct_cost,
        'overhead_percent': overhead_percent,
        'overhead_amount': overhead_amount,
        'subtotal': subtotal,
        'profit_percent': profit_percent,
        'profit_amount': profit_amount,
        'total': total,
        'margin_percent': margin
    }


# ============ PDF Generation ============

def generate_quotation_pdf(quotation):
    """Generate PDF for quotation."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    from reportlab.pdfgen import canvas
    from reportlab.qrcode import qr
    import io
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, 
                           topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    elements = []
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER)
    header_style = ParagraphStyle('CustomHeader', parent=styles['Heading2'], fontSize=12)
    
    # Header
    from flask import current_app
    company_name = current_app.config.get('COMPANY_NAME', 'NPL Fasteners')
    
    elements.append(Paragraph(f"<b>{company_name}</b>", title_style))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph("QUOTATION", ParagraphStyle('QuoteTitle', parent=styles['Heading1'], 
                                                            fontSize=24, alignment=TA_CENTER)))
    elements.append(Spacer(1, 10*mm))
    
    # Quotation details table
    quote_data = [
        ['Quotation Number:', quotation.quotation_number, 'Date:', str(quotation.quotation_date)],
        ['Customer:', quotation.customer.company_name if quotation.customer else '', '', ''],
        ['Valid Until:', str(quotation.valid_until) if quotation.valid_until else '', '', ''],
    ]
    
    quote_table = Table(quote_data, colWidths=[40*mm, 60*mm, 30*mm, 50*mm])
    quote_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(quote_table)
    elements.append(Spacer(1, 15*mm))
    
    # Items table
    items_header = ['#', 'Description', 'Material', 'Qty', 'Unit Price', 'Total']
    items_data = [items_header]
    
    for i, item in enumerate(quotation.items.all(), 1):
        row = [
            str(i),
            item.part_description or '',
            item.material or '',
            str(item.quantity_1) if item.quantity_1 else '',
            f"₹ {item.unit_price_1:.2f}" if item.unit_price_1 else '',
            f"₹ {(item.quantity_1 or 0) * (item.unit_price_1 or 0):.2f}"
        ]
        items_data.append(row)
    
    # Add totals
    items_data.append(['', '', '', '', 'Subtotal:', f"₹ {quotation.subtotal:.2f}"])
    items_data.append(['', '', '', '', 'Tax ({}%):'.format(int(quotation.tax_percent)), 
                       f"₹ {quotation.tax_amount:.2f}"])
    items_data.append(['', '', '', '', 'TOTAL:', f"₹ {quotation.total_amount:.2f}"])
    
    items_table = Table(items_data, colWidths=[10*mm, 70*mm, 30*mm, 20*mm, 30*mm, 30*mm])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -4), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (-2, -3), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.black),
        ('LINEABOVE', (4, -3), (-1, -3), 1, colors.black),
        ('LINEABOVE', (4, -1), (-1, -1), 1.5, colors.black),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 15*mm))
    
    # Terms and conditions
    if quotation.terms_conditions:
        elements.append(Paragraph("<b>Terms & Conditions:</b>", header_style))
        elements.append(Paragraph(quotation.terms_conditions, styles['Normal']))
        elements.append(Spacer(1, 10*mm))
    
    # Signature area
    sig_data = [
        ['', '', 'For ' + company_name],
        ['', '', ''],
        ['', '', 'Authorised Signatory'],
    ]
    sig_table = Table(sig_data, colWidths=[80*mm, 20*mm, 90*mm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(sig_table)
    
    # Page numbers
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============ Excel Export ============

def export_quotations_to_excel(quotations):
    """Export quotations to Excel."""
    data = []
    for q in quotations:
        for item in q.items.all():
            data.append({
                'Quotation No': q.quotation_number,
                'Date': str(q.quotation_date),
                'Customer': q.customer.company_name if q.customer else '',
                'Part Description': item.part_description,
                'Material': item.material,
                'Quantity': item.quantity_1,
                'Unit Price': item.unit_price_1,
                'Total': (item.quantity_1 or 0) * (item.unit_price_1 or 0),
                'Status': q.status,
                'Currency': q.currency
            })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Quotations', index=False)
    
    output.seek(0)
    return output


def export_products_to_excel(products):
    """Export products to Excel."""
    data = []
    for p in products:
        data.append({
            'Internal Code': p.internal_code,
            'Product Name': p.product_name,
            'Customer Part No': p.customer_part_number,
            'Drawing Number': p.drawing_number,
            'Material': p.material,
            'Weight (g)': p.weight,
            'Standard': p.standard,
            'Thread': p.thread,
            'Revision': p.revision,
            'Status': p.status,
            'Created': str(p.created_at)[:10]
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Products', index=False)
    
    output.seek(0)
    return output


def export_customers_to_excel(customers):
    """Export customers to Excel."""
    data = []
    for c in customers:
        data.append({
            'Customer Code': c.customer_code,
            'Company Name': c.company_name,
            'Contact Person': c.contact_person,
            'Email': c.email,
            'Phone': c.phone,
            'GST/VAT': c.gst_vat,
            'Country': c.country,
            'Currency': c.currency,
            'Payment Terms': c.payment_terms,
            'Status': 'Active' if c.is_active else 'Inactive'
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Customers', index=False)
    
    output.seek(0)
    return output


def export_materials_to_excel(materials):
    """Export materials to Excel."""
    data = []
    for m in materials:
        data.append({
            'Material Code': m.material_code,
            'Material Name': m.material_name,
            'Density (kg/m³)': m.density,
            'Grade': m.grade,
            'HSN Code': m.hsn_code,
            'Default Rate': m.default_rate,
            'Rate Unit': m.rate_unit,
            'Default Supplier': m.default_supplier,
            'Status': 'Active' if m.is_active else 'Inactive'
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Materials', index=False)
    
    output.seek(0)
    return output


# ============ Currency Conversion ============

def convert_currency(amount, from_currency, to_currency, rates=None):
    """Convert amount between currencies."""
    if from_currency == to_currency:
        return amount
    
    if rates is None:
        rates = {
            'USD': 83.0,
            'EUR': 90.0,
            'GBP': 100.0,
            'AED': 22.5,
            'INR': 1.0
        }
    
    # Convert to INR first
    inr_amount = amount
    if from_currency != 'INR':
        rate = rates.get(from_currency, 1)
        inr_amount = amount * rate
    
    # Convert from INR to target
    if to_currency == 'INR':
        return inr_amount
    
    rate = rates.get(to_currency, 1)
    return inr_amount / rate


# ============ Revision History ============

def track_revision(entity_type, entity_id, field_name, old_value, new_value, user_id):
    """Track changes in revision history."""
    from app.models import RevisionHistory, db
    
    history = RevisionHistory(
        entity_type=entity_type,
        entity_id=entity_id,
        field_name=field_name,
        old_value=str(old_value) if old_value else None,
        new_value=str(new_value) if new_value else None,
        changed_by=user_id
    )
    db.session.add(history)
    return history


# ============ Notifications ============

def create_notification(user_id, title, message, notification_type, reference_type=None, reference_id=None):
    """Create a notification for user."""
    from app.models import Notification, db
    
    notification = Notification(
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
        reference_type=reference_type,
        reference_id=reference_id
    )
    db.session.add(notification)
    return notification
