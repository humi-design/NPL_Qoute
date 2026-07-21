from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from app.main import main_bp
from app.models import (
    db, Customer, RFQ, Product, Material, Quotation, QuotationItem,
    ProductCostHistory, VendorRate, MaterialRate
)


@main_bp.route('/')
@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard view."""
    today = datetime.now().date()
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Today's quotations
    todays_quotations = Quotation.query.filter(
        func.date(Quotation.quotation_date) == today
    ).count()
    
    # Pending RFQs
    pending_rfqs = RFQ.query.filter(
        RFQ.status.in_(['pending', 'in_progress'])
    ).count()
    
    # Won quotations this month
    won_quotations = Quotation.query.filter(
        Quotation.status == 'won',
        func.extract('year', Quotation.won_date) == current_year,
        func.extract('month', Quotation.won_date) == current_month
    ).count()
    
    # Lost quotations this month
    lost_quotations = Quotation.query.filter(
        Quotation.status == 'lost',
        func.extract('year', Quotation.lost_date) == current_year,
        func.extract('month', Quotation.lost_date) == current_month
    ).count()
    
    # Monthly sales (won quotations total)
    monthly_sales = db.session.query(func.sum(Quotation.total_amount)).filter(
        Quotation.status == 'won',
        func.extract('year', Quotation.won_date) == current_year,
        func.extract('month', Quotation.won_date) == current_month
    ).scalar() or 0
    
    # Average margin
    avg_margin = db.session.query(func.avg(Quotation.margin_percent)).filter(
        Quotation.status == 'won',
        Quotation.margin_percent > 0
    ).scalar() or 0
    
    # Material cost this month
    material_cost = db.session.query(func.sum(ProductCostHistory.raw_material_cost)).filter(
        func.extract('year', ProductCostHistory.recorded_at) == current_year,
        func.extract('month', ProductCostHistory.recorded_at) == current_month
    ).scalar() or 0
    
    # Recent quotations
    recent_quotations = Quotation.query.order_by(
        Quotation.created_at.desc()
    ).limit(10).all()
    
    # Recent products
    recent_products = Product.query.order_by(
        Product.created_at.desc()
    ).limit(10).all()
    
    # Recent RFQs
    recent_rfqs = RFQ.query.order_by(
        RFQ.created_at.desc()
    ).limit(5).all()
    
    # Monthly quotation data for chart (last 6 months)
    monthly_data = []
    for i in range(5, -1, -1):
        month_date = today.replace(day=1) - timedelta(days=i * 30)
        month_name = month_date.strftime('%b')
        
        won_count = Quotation.query.filter(
            Quotation.status == 'won',
            func.extract('year', Quotation.won_date) == month_date.year,
            func.extract('month', Quotation.won_date) == month_date.month
        ).count()
        
        lost_count = Quotation.query.filter(
            Quotation.status == 'lost',
            func.extract('year', Quotation.lost_date) == month_date.year,
            func.extract('month', Quotation.lost_date) == month_date.month
        ).count()
        
        monthly_data.append({
            'month': month_name,
            'won': won_count,
            'lost': lost_count
        })
    
    # Top customers by quotation value
    top_customers = db.session.query(
        Customer.id,
        Customer.company_name,
        func.sum(Quotation.total_amount).label('total')
    ).join(Quotation).filter(
        Quotation.status == 'won'
    ).group_by(Customer.id).order_by(
        desc('total')
    ).limit(5).all()
    
    # Top products
    top_products = db.session.query(
        Product.id,
        Product.internal_code,
        Product.product_name,
        func.sum(QuotationItem.quantity_1).label('total_qty')
    ).join(QuotationItem).join(Quotation).filter(
        Quotation.status == 'won'
    ).group_by(Product.id).order_by(
        desc('total_qty')
    ).limit(5).all()
    
    # Notifications - now handled by context processor
    # (no need to query here, use the injected notifications variable)
    
    return render_template('main/dashboard.html',
                         page_title='Dashboard',
                         todays_quotations=todays_quotations,
                         pending_rfqs=pending_rfqs,
                         won_quotations=won_quotations,
                         lost_quotations=lost_quotations,
                         monthly_sales=monthly_sales,
                         avg_margin=avg_margin,
                         material_cost=material_cost,
                         recent_quotations=recent_quotations,
                         recent_products=recent_products,
                         recent_rfqs=recent_rfqs,
                         monthly_data=monthly_data,
                         top_customers=top_customers,
                         top_products=top_products)


@main_bp.route('/search')
@login_required
def search():
    """Global search across all entities."""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify({'results': []})
    
    results = {
        'customers': [],
        'products': [],
        'quotations': [],
        'rfqs': [],
        'vendors': []
    }
    
    # Search customers
    customers = Customer.query.filter(
        Customer.company_name.ilike(f'%{query}%') |
        Customer.customer_code.ilike(f'%{query}%') |
        Customer.contact_person.ilike(f'%{query}%')
    ).limit(5).all()
    results['customers'] = [
        {'id': c.id, 'code': c.customer_code, 'name': c.company_name, 'type': 'customer'}
        for c in customers
    ]
    
    # Search products
    products = Product.query.filter(
        Product.product_name.ilike(f'%{query}%') |
        Product.internal_code.ilike(f'%{query}%') |
        Product.drawing_number.ilike(f'%{query}%') |
        Product.customer_part_number.ilike(f'%{query}%')
    ).limit(5).all()
    results['products'] = [
        {'id': p.id, 'code': p.internal_code, 'name': p.product_name, 'type': 'product'}
        for p in products
    ]
    
    # Search quotations
    quotations = Quotation.query.filter(
        Quotation.quotation_number.ilike(f'%{query}%')
    ).limit(5).all()
    results['quotations'] = [
        {'id': q.id, 'number': q.quotation_number, 'customer': q.customer.company_name if q.customer else '', 'type': 'quotation'}
        for q in quotations
    ]
    
    # Search RFQs
    rfqs = RFQ.query.filter(
        RFQ.rfq_number.ilike(f'%{query}%') |
        RFQ.subject.ilike(f'%{query}%')
    ).limit(5).all()
    results['rfqs'] = [
        {'id': r.id, 'number': r.rfq_number, 'subject': r.subject or '', 'type': 'rfq'}
        for r in rfqs
    ]
    
    return jsonify({'results': results})


@main_bp.route('/calculator')
@login_required
def calculator():
    """Raw material calculator page."""
    materials = Material.query.filter_by(is_active=True).all()
    return render_template('main/calculator.html', 
                         page_title='Raw Material Calculator',
                         materials=materials)


@main_bp.route('/calculator/calculate', methods=['POST'])
@login_required
def calculate_material():
    """Calculate material requirements."""
    import math
    from app.utils import (
        calculate_blank_length, calculate_material_weight, 
        calculate_round_bar_volume, calculate_hex_bar_volume,
        calculate_square_bar_volume, calculate_tube_volume,
        calculate_sheet_volume, calculate_pieces_per_bar,
        calculate_waste_percentage, calculate_utilization,
        calculate_material_cost
    )
    
    stock_type = request.form.get('stock_type')
    finished_length = float(request.form.get('finished_length', 0))
    
    # Get dimensions based on stock type
    diameter = float(request.form.get('diameter') or 0)
    af = float(request.form.get('af') or 0)
    side = float(request.form.get('side') or 0)
    od = float(request.form.get('od') or 0)
    id_val = float(request.form.get('id') or 0)
    thickness = float(request.form.get('thickness') or 0)
    width = float(request.form.get('width') or 0)
    
    # Allowances
    parting = float(request.form.get('parting') or 3)
    facing = float(request.form.get('facing') or 2)
    machining_allowance = float(request.form.get('machining_allowance') or 1)
    grinding_allowance = float(request.form.get('grinding_allowance') or 0)
    chamfer_allowance = float(request.form.get('chamfer_allowance') or 0.5)
    
    # Custom allowances
    custom_allowances = {}
    for i in range(1, 4):
        name = request.form.get(f'custom_allowance_{i}_name')
        value = request.form.get(f'custom_allowance_{i}_value')
        if name and value:
            custom_allowances[name] = float(value)
    
    # Bar info
    bar_length = float(request.form.get('bar_length') or 3000)
    
    # Calculate blank length
    blank_length = calculate_blank_length(
        finished_length, parting, facing, machining_allowance,
        grinding_allowance, chamfer_allowance, custom_allowances
    )
    
    # Calculate volume based on stock type
    if stock_type == 'round_bar':
        dim = diameter or af
        volume = calculate_round_bar_volume(dim, blank_length)
    elif stock_type == 'hex_bar':
        volume = calculate_hex_bar_volume(af or diameter, blank_length)
    elif stock_type == 'square_bar':
        volume = calculate_square_bar_volume(side or diameter, blank_length)
    elif stock_type in ['tube', 'pipe']:
        volume = calculate_tube_volume(od or diameter, id_val, blank_length)
    elif stock_type == 'sheet':
        volume = calculate_sheet_volume(width, thickness, blank_length)
    else:
        # Default to round bar
        volume = calculate_round_bar_volume(diameter, blank_length)
    
    # Get material density
    material_id = request.form.get('material_id')
    custom_density = float(request.form.get('custom_density') or 0)
    
    if material_id:
        material = Material.query.get(int(material_id))
        density = material.density if material else custom_density
        rate = material.default_rate or 0
        rate_unit = material.rate_unit or '₹/kg'
    else:
        density = custom_density
        rate = float(request.form.get('rate_per_kg') or 0)
        rate_unit = '₹/kg'
    
    # Calculate weight
    weight_grams = calculate_material_weight(volume, density)
    
    # Pieces per bar
    pieces = calculate_pieces_per_bar(blank_length, bar_length)
    
    # Waste and utilization
    waste_pct = calculate_waste_percentage(blank_length, bar_length)
    utilization = calculate_utilization(blank_length, bar_length)
    
    # Material cost
    material_cost = calculate_material_cost(weight_grams, rate, rate_unit)
    
    return jsonify({
        'success': True,
        'blank_length': round(blank_length, 2),
        'volume_mm3': round(volume, 2),
        'weight_grams': round(weight_grams, 4),
        'weight_kg': round(weight_grams / 1000, 6),
        'pieces_per_bar': pieces,
        'waste_percentage': round(waste_pct, 2),
        'utilization': round(utilization, 2),
        'material_cost': round(material_cost, 4),
        'density': density
    })


@main_bp.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for dashboard statistics."""
    current_year = datetime.now().year
    
    # Get various stats
    stats = {
        'total_customers': Customer.query.count(),
        'total_products': Product.query.filter_by(status='active').count(),
        'total_quotations': Quotation.query.count(),
        'total_rfqs': RFQ.query.count(),
        'monthly_sales': db.session.query(func.sum(Quotation.total_amount)).filter(
            Quotation.status == 'won',
            func.extract('year', Quotation.won_date) == current_year
        ).scalar() or 0,
        'pending_rfqs': RFQ.query.filter(RFQ.status == 'pending').count(),
        'draft_quotations': Quotation.query.filter(Quotation.status == 'draft').count()
    }
    
    return jsonify(stats)
