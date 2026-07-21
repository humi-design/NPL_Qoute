from flask import jsonify, request
from flask_login import login_required, current_user
from datetime import datetime
from app.api import api_bp
from app.models import db, Customer, Product, Quotation, RFQ, Material, Machine, Vendor, Process


# ============ Customers API ============

@api_bp.route('/customers', methods=['GET'])
@login_required
def get_customers():
    """Get all customers."""
    customers = Customer.query.filter_by(is_active=True).all()
    return jsonify([
        {
            'id': c.id,
            'code': c.customer_code,
            'name': c.company_name,
            'email': c.email,
            'phone': c.phone,
            'currency': c.currency
        }
        for c in customers
    ])


@api_bp.route('/customers/<int:id>', methods=['GET'])
@login_required
def get_customer(id):
    """Get customer by ID."""
    customer = Customer.query.get_or_404(id)
    return jsonify({
        'id': customer.id,
        'code': customer.customer_code,
        'name': customer.company_name,
        'contact_person': customer.contact_person,
        'email': customer.email,
        'phone': customer.phone,
        'mobile': customer.mobile,
        'gst': customer.gst_vat,
        'country': customer.country,
        'currency': customer.currency,
        'payment_terms': customer.payment_terms,
        'billing_address': customer.billing_address,
        'shipping_address': customer.shipping_address
    })


# ============ Products API ============

@api_bp.route('/products', methods=['GET'])
@login_required
def get_products():
    """Get all products."""
    material = request.args.get('material', '')
    standard = request.args.get('standard', '')
    
    query = Product.query.filter_by(status='active')
    
    if material:
        query = query.filter(Product.material.ilike(f'%{material}%'))
    if standard:
        query = query.filter(Product.standard.ilike(f'%{standard}%'))
    
    products = query.all()
    return jsonify([
        {
            'id': p.id,
            'code': p.internal_code,
            'name': p.product_name,
            'material': p.material,
            'drawing': p.drawing_number,
            'standard': p.standard,
            'thread': p.thread,
            'weight': p.weight
        }
        for p in products
    ])


@api_bp.route('/products/<int:id>', methods=['GET'])
@login_required
def get_product(id):
    """Get product by ID."""
    product = Product.query.get_or_404(id)
    return jsonify({
        'id': product.id,
        'code': product.internal_code,
        'name': product.product_name,
        'customer_part_number': product.customer_part_number,
        'drawing_number': product.drawing_number,
        'revision': product.revision,
        'material': product.material,
        'weight': product.weight,
        'standard': product.standard,
        'standard_number': product.standard_number,
        'thread': product.thread,
        'dimensions': product.dimensions,
        'description': product.description
    })


# ============ Materials API ============

@api_bp.route('/materials', methods=['GET'])
@login_required
def get_materials():
    """Get all materials."""
    materials = Material.query.filter_by(is_active=True).all()
    return jsonify([
        {
            'id': m.id,
            'code': m.material_code,
            'name': m.material_name,
            'density': m.density,
            'grade': m.grade,
            'rate': m.default_rate,
            'unit': m.rate_unit
        }
        for m in materials
    ])


# ============ Machines API ============

@api_bp.route('/machines', methods=['GET'])
@login_required
def get_machines():
    """Get all machines."""
    machines = Machine.query.filter_by(is_active=True).all()
    return jsonify([
        {
            'id': m.id,
            'code': m.machine_code,
            'name': m.machine_name,
            'type': m.machine_type,
            'hourly_rate': m.hourly_rate,
            'department': m.department
        }
        for m in machines
    ])


# ============ Vendors API ============

@api_bp.route('/vendors', methods=['GET'])
@login_required
def get_vendors():
    """Get all vendors."""
    vendors = Vendor.query.filter_by(is_active=True).all()
    return jsonify([
        {
            'id': v.id,
            'code': v.vendor_code,
            'name': v.vendor_name,
            'contact': v.contact_person,
            'email': v.email,
            'lead_time': v.lead_time
        }
        for v in vendors
    ])


# ============ Processes API ============

@api_bp.route('/processes', methods=['GET'])
@login_required
def get_processes():
    """Get all processes."""
    processes = Process.query.filter_by(is_active=True).all()
    return jsonify([
        {
            'id': p.id,
            'code': p.process_code,
            'name': p.process_name,
            'type': p.process_type,
            'machine_id': p.machine_id,
            'vendor_id': p.vendor_id,
            'setup_time': p.setup_time,
            'cycle_time': p.cycle_time,
            'cost_type': p.cost_type,
            'cost_value': p.cost_value
        }
        for p in processes
    ])


# ============ Quotations API ============

@api_bp.route('/quotations', methods=['GET'])
@login_required
def get_quotations():
    """Get quotations with filters."""
    status = request.args.get('status', '')
    customer_id = request.args.get('customer_id', type=int)
    
    query = Quotation.query
    
    if status:
        query = query.filter(Quotation.status == status)
    if customer_id:
        query = query.filter(Quotation.customer_id == customer_id)
    
    quotations = query.order_by(Quotation.quotation_date.desc()).limit(50).all()
    
    return jsonify([
        {
            'id': q.id,
            'number': q.quotation_number,
            'version': q.quotation_version,
            'customer_id': q.customer_id,
            'customer_name': q.customer.company_name if q.customer else None,
            'date': q.quotation_date.isoformat(),
            'valid_until': q.valid_until.isoformat() if q.valid_until else None,
            'status': q.status,
            'currency': q.currency,
            'total': q.total_amount,
            'margin': q.margin_percent
        }
        for q in quotations
    ])


@api_bp.route('/quotations/<int:id>', methods=['GET'])
@login_required
def get_quotation(id):
    """Get quotation by ID."""
    quotation = Quotation.query.get_or_404(id)
    
    return jsonify({
        'id': quotation.id,
        'number': quotation.quotation_number,
        'version': quotation.quotation_version,
        'customer_id': quotation.customer_id,
        'customer_name': quotation.customer.company_name if quotation.customer else None,
        'date': quotation.quotation_date.isoformat(),
        'valid_until': quotation.valid_until.isoformat() if quotation.valid_until else None,
        'status': quotation.status,
        'currency': quotation.currency,
        'exchange_rate': quotation.exchange_rate,
        'subtotal': quotation.subtotal,
        'tax_percent': quotation.tax_percent,
        'tax_amount': quotation.tax_amount,
        'total': quotation.total_amount,
        'margin': quotation.margin_percent,
        'payment_terms': quotation.payment_terms,
        'delivery_terms': quotation.delivery_terms,
        'items': [
            {
                'id': item.id,
                'product_id': item.product_id,
                'description': item.part_description,
                'drawing': item.drawing_number,
                'material': item.material,
                'qty_1': item.quantity_1,
                'price_1': item.unit_price_1,
                'qty_2': item.quantity_2,
                'price_2': item.unit_price_2,
                'qty_3': item.quantity_3,
                'price_3': item.unit_price_3
            }
            for item in quotation.items.all()
        ]
    })


# ============ Dashboard API ============

@api_bp.route('/dashboard/stats', methods=['GET'])
@login_required
def dashboard_stats():
    """Get dashboard statistics."""
    today = datetime.now().date()
    current_year = datetime.now().year
    
    from sqlalchemy import func
    
    stats = {
        'todays_quotations': Quotation.query.filter(
            func.date(Quotation.quotation_date) == today
        ).count(),
        'pending_rfqs': RFQ.query.filter(RFQ.status.in_(['pending', 'in_progress'])).count(),
        'total_customers': Customer.query.count(),
        'total_products': Product.query.filter_by(status='active').count(),
        'monthly_sales': db.session.query(func.sum(Quotation.total_amount)).filter(
            Quotation.status == 'won',
            func.extract('year', Quotation.won_date) == current_year
        ).scalar() or 0
    }
    
    return jsonify(stats)


# ============ Calculator API ============

@api_bp.route('/calculator/material', methods=['POST'])
@login_required
def calculate_material():
    """Calculate material requirements."""
    from app.utils import (
        calculate_blank_length, calculate_material_weight, 
        calculate_round_bar_volume, calculate_hex_bar_volume,
        calculate_square_bar_volume, calculate_tube_volume,
        calculate_sheet_volume, calculate_pieces_per_bar,
        calculate_waste_percentage, calculate_utilization
    )
    
    data = request.get_json()
    
    stock_type = data.get('stock_type', 'round_bar')
    finished_length = float(data.get('finished_length', 0))
    diameter = float(data.get('diameter', 0))
    parting = float(data.get('parting', 3))
    facing = float(data.get('facing', 2))
    machining = float(data.get('machining_allowance', 1))
    grinding = float(data.get('grinding_allowance', 0))
    chamfer = float(data.get('chamfer_allowance', 0.5))
    bar_length = float(data.get('bar_length', 3000))
    density = float(data.get('density', 7850))
    
    # Calculate blank length
    blank_length = calculate_blank_length(
        finished_length, parting, facing, machining, grinding, chamfer
    )
    
    # Calculate volume
    if stock_type == 'round_bar':
        volume = calculate_round_bar_volume(diameter, blank_length)
    elif stock_type == 'hex_bar':
        volume = calculate_hex_bar_volume(diameter, blank_length)
    elif stock_type == 'square_bar':
        volume = calculate_square_bar_volume(diameter, blank_length)
    elif stock_type in ['tube', 'pipe']:
        volume = calculate_tube_volume(diameter, data.get('id', 0), blank_length)
    elif stock_type == 'sheet':
        volume = calculate_sheet_volume(diameter, data.get('thickness', 0), blank_length)
    else:
        volume = calculate_round_bar_volume(diameter, blank_length)
    
    # Calculate weight
    weight_grams = calculate_material_weight(volume, density)
    
    # Calculate utilization
    pieces = calculate_pieces_per_bar(blank_length, bar_length)
    waste = calculate_waste_percentage(blank_length, bar_length)
    utilization = calculate_utilization(blank_length, bar_length)
    
    return jsonify({
        'blank_length': round(blank_length, 2),
        'volume_mm3': round(volume, 2),
        'weight_grams': round(weight_grams, 4),
        'weight_kg': round(weight_grams / 1000, 6),
        'pieces_per_bar': pieces,
        'waste_percentage': round(waste, 2),
        'utilization': round(utilization, 2)
    })


@api_bp.route('/calculator/costing', methods=['POST'])
@login_required
def calculate_costing():
    """Calculate total costing."""
    from app.utils import calculate_total_cost
    
    data = request.get_json()
    
    result = calculate_total_cost(
        material_cost=data.get('material_cost', 0),
        machining_cost=data.get('machining_cost', 0),
        vendor_cost=data.get('vendor_cost', 0),
        tooling_cost=data.get('tooling_cost', 0),
        overhead_percent=data.get('overhead_percent', 15),
        profit_percent=data.get('profit_percent', 20),
        scrap_percent=data.get('scrap_percent', 2)
    )
    
    return jsonify(result)
