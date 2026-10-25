from flask import jsonify, request
from flask_login import login_required, current_user
from datetime import datetime
from app.api import api_bp
from app.models import (
    db, Customer, Product, Quotation, RFQ, Material, Machine, Vendor, Process,
    VendorWorkOrder, Supplier, SupplierType
)


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
    """Get all vendors (optionally with work summary)."""
    from app.utils import compute_vendor_summary

    include_summary = request.args.get('summary', '0') in ('1', 'true', 'yes')
    vendors = Vendor.query.filter_by(is_active=True).all()

    payload = []
    for v in vendors:
        item = {
            'id': v.id,
            'code': v.vendor_code,
            'name': v.vendor_name,
            'contact': v.contact_person,
            'email': v.email,
            'lead_time': v.lead_time
        }
        if include_summary:
            item['summary'] = compute_vendor_summary(v)
        payload.append(item)

    return jsonify(payload)


@api_bp.route('/vendors/<int:id>', methods=['GET'])
@login_required
def get_vendor(id):
    """Get a single vendor with work summary and recent work orders."""
    from app.utils import compute_vendor_summary, compute_vendor_financial_summary

    vendor = Vendor.query.get_or_404(id)
    summary = compute_vendor_summary(vendor)
    financials = compute_vendor_financial_summary(vendor)

    work_orders = vendor.work_orders.order_by(VendorWorkOrder.id.desc()).limit(50).all()

    return jsonify({
        'id': vendor.id,
        'code': vendor.vendor_code,
        'name': vendor.vendor_name,
        'contact': vendor.contact_person,
        'email': vendor.email,
        'lead_time': vendor.lead_time,
        'summary': summary,
        'financials': financials,
        'work_orders': [
            {
                'id': w.id,
                'number': w.work_order_number,
                'title': w.title,
                'service': w.service_name,
                'status': w.status,
                'quantity_ordered': w.quantity_ordered,
                'quantity_received': w.quantity_received,
                'pending_quantity': w.pending_quantity,
                'due_date': w.due_date.isoformat() if w.due_date else None,
                'is_overdue': w.is_overdue,
            }
            for w in work_orders
        ],
    })


@api_bp.route('/vendors/work-orders', methods=['GET'])
@login_required
def get_vendor_work_orders():
    """Get vendor work orders with the Current / Past / All view filter."""
    view = request.args.get('view', 'all')
    vendor_id = request.args.get('vendor_id', type=int)

    query = VendorWorkOrder.query
    if view == 'current':
        query = query.filter(VendorWorkOrder.status.in_(VendorWorkOrder.OPEN_STATUSES))
    elif view == 'past':
        query = query.filter(VendorWorkOrder.status.in_(VendorWorkOrder.COMPLETE_STATUSES))

    if vendor_id:
        query = query.filter(VendorWorkOrder.vendor_id == vendor_id)

    work_orders = query.order_by(VendorWorkOrder.id.desc()).limit(500).all()

    return jsonify([
        {
            'id': w.id,
            'number': w.work_order_number,
            'vendor_id': w.vendor_id,
            'vendor': w.vendor.vendor_name if w.vendor else None,
            'title': w.title,
            'service': w.service_name,
            'status': w.status,
            'quantity_ordered': w.quantity_ordered,
            'quantity_received': w.quantity_received,
            'pending_quantity': w.pending_quantity,
            'order_date': w.order_date.isoformat() if w.order_date else None,
            'due_date': w.due_date.isoformat() if w.due_date else None,
            'is_overdue': w.is_overdue,
        }
        for w in work_orders
    ])


# ============ Suppliers API ============

@api_bp.route('/suppliers', methods=['GET'])
@login_required
def get_suppliers():
    """Get all suppliers (optionally with material summary)."""
    from app.utils import compute_supplier_summary

    include_summary = request.args.get('summary', '0') in ('1', 'true', 'yes')
    type_id = request.args.get('type', type=int)
    status = request.args.get('status', '')

    query = Supplier.query
    if status == 'active':
        query = query.filter(Supplier.is_active.is_(True))
    elif status == 'inactive':
        query = query.filter(Supplier.is_active.is_(False))
    else:
        query = query.filter(Supplier.is_active.is_(True))

    if type_id:
        query = query.filter(Supplier.supplier_type_id == type_id)

    suppliers = query.order_by(Supplier.company_name).all()

    payload = []
    for s in suppliers:
        item = {
            'id': s.id,
            'code': s.supplier_code,
            'name': s.company_name,
            'type': s.supplier_type.type_name if s.supplier_type else None,
            'contact': s.contact_person,
            'email': s.email,
            'phone': s.phone,
            'city': s.city,
            'gst_vat': s.gst_vat,
            'currency': s.default_currency,
            'lead_time': s.lead_time,
            'payment_terms': s.payment_terms,
        }
        if include_summary:
            item['summary'] = compute_supplier_summary(s)
        payload.append(item)

    return jsonify(payload)


@api_bp.route('/suppliers/<int:id>', methods=['GET'])
@login_required
def get_supplier(id):
    """Get a single supplier with supplied materials, procurement and summary."""
    from app.utils import (compute_supplier_summary,
                           compute_supplier_procurement_summary)
    from app.models import (SupplierPurchaseOrder, SupplierOrderReceipt,
                            SupplierOrderReceiptLine, SupplierInvoice)

    supplier = Supplier.query.get_or_404(id)
    links = supplier.supplied_materials.all()

    pos = SupplierPurchaseOrder.query.filter_by(supplier_id=id).order_by(
        SupplierPurchaseOrder.id.desc()).all()
    batch_lines = (SupplierOrderReceiptLine.query
                   .join(SupplierOrderReceipt)
                   .filter(SupplierOrderReceipt.supplier_id == id).all())
    invoices = SupplierInvoice.query.filter_by(supplier_id=id).all()

    return jsonify({
        'id': supplier.id,
        'code': supplier.supplier_code,
        'name': supplier.company_name,
        'type': supplier.supplier_type.type_name if supplier.supplier_type else None,
        'contact': supplier.contact_person,
        'email': supplier.email,
        'phone': supplier.phone,
        'mobile': supplier.mobile,
        'address': supplier.full_address,
        'gst_vat': supplier.gst_vat,
        'pan': supplier.pan,
        'currency': supplier.default_currency,
        'payment_terms': supplier.payment_terms,
        'lead_time': supplier.lead_time,
        'is_active': supplier.is_active,
        'summary': compute_supplier_summary(supplier),
        'procurement': compute_supplier_procurement_summary(supplier),
        'purchase_orders': [
            {
                'id': po.id,
                'number': po.po_number,
                'order_date': po.order_date.isoformat() if po.order_date else None,
                'expected_date': po.expected_date.isoformat() if po.expected_date else None,
                'status': po.status,
                'ordered_quantity': po.ordered_quantity,
                'received_quantity': po.received_quantity,
                'accepted_quantity': po.accepted_quantity,
                'rejected_quantity': po.rejected_quantity,
                'pending_quantity': po.pending_quantity,
                'value': po.total_value,
                'is_overdue': po.is_overdue,
                'days_overdue': po.days_overdue,
            }
            for po in pos
        ],
        'batches': [
            {
                'id': b.id,
                'material_id': b.material_id,
                'material': b.material.material_name if b.material else None,
                'batch_number': b.batch_number,
                'heat_number': b.heat_number,
                'lot_number': b.lot_number,
                'certificate_ref': b.certificate_ref,
                'quantity_received': b.quantity_received,
                'quantity_rejected': b.quantity_rejected,
                'inspection_status': b.inspection_status,
            }
            for b in batch_lines
        ],
        'invoices': [
            {
                'id': i.id,
                'number': i.invoice_number,
                'invoice_date': i.invoice_date.isoformat() if i.invoice_date else None,
                'due_date': i.due_date.isoformat() if i.due_date else None,
                'total_amount': i.total_amount,
                'paid_amount': i.paid_amount,
                'outstanding_amount': i.outstanding_amount,
                'ageing_bucket': i.ageing_bucket,
                'status': i.status,
            }
            for i in invoices
        ],
        'supplied_materials': [
            {
                'id': link.id,
                'material_id': link.material_id,
                'material': link.material.material_name if link.material else None,
                'material_code': link.material.material_code if link.material else None,
                'grade': link.grade,
                'rate': link.rate,
                'rate_unit': link.rate_unit,
                'lead_time': link.lead_time,
                'is_preferred': link.is_preferred,
            }
            for link in links
        ],
    })


@api_bp.route('/supplier-types', methods=['GET'])
@login_required
def get_supplier_types():
    """Get all active supplier types (for dropdowns/integrations)."""
    types = SupplierType.query.filter_by(is_active=True).order_by(
        SupplierType.sort_order, SupplierType.type_name).all()

    return jsonify([
        {'id': t.id, 'code': t.type_code, 'name': t.type_name,
         'supplier_count': t.suppliers.count()}
        for t in types
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
