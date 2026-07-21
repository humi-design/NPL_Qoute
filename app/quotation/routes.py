from flask import render_template, redirect, url_for, request, flash, jsonify, send_file
from flask_login import login_required
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from app.quotation import quotation_bp
from app.models import (
    db, Quotation, QuotationItem, QuotationOperation, Product,
    Customer, Process, Machine, Material, ProductCostHistory, Attachment
)
from app.forms import QuotationForm, QuotationItemForm
from app.utils import (
    generate_quotation_number, export_quotations_to_excel,
    generate_product_code, upload_file, calculate_total_cost
)
from app.extensions import current_app


@quotation_bp.route('/')
@login_required
def index():
    """Quotation list page."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    status = request.args.get('status', '')
    search = request.args.get('search', '')
    
    query = Quotation.query
    
    if status:
        query = query.filter(Quotation.status == status)
    
    if search:
        query = query.filter(Quotation.quotation_number.ilike(f'%{search}%'))
    
    quotations = query.order_by(Quotation.quotation_date.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('quotation/index.html', 
                         quotations=quotations, 
                         status=status, 
                         search=search)


@quotation_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new quotation."""
    form = QuotationForm()
    form.quotation_number.data = generate_quotation_number()
    form.quotation_date.data = datetime.now().date()
    form.valid_until.data = (datetime.now() + timedelta(days=30)).date()
    form.currency.data = 'INR'
    
    # Pre-select customer if provided
    customer_id = request.args.get('customer', type=int)
    if customer_id:
        customer = Customer.query.get(customer_id)
        if customer:
            form.customer_id.data = customer
            form.billing_address.data = customer.billing_address
            form.shipping_address.data = customer.shipping_address
            form.payment_terms.data = customer.payment_terms
            form.currency.data = customer.currency
    
    # Pre-select product if provided
    product_id = request.args.get('product', type=int)
    
    if form.validate_on_submit():
        quotation = Quotation(
            quotation_number=form.quotation_number.data,
            quotation_version='V1',
            customer_id=form.customer_id.data.id,
            quotation_date=form.quotation_date.data,
            valid_until=form.valid_until.data,
            status='draft',
            priority=form.priority.data,
            currency=form.currency.data,
            exchange_rate=form.exchange_rate.data or 1.0,
            billing_address=form.billing_address.data,
            shipping_address=form.shipping_address.data,
            payment_terms=form.payment_terms.data,
            delivery_terms=form.delivery_terms.data,
            warranty=form.warranty.data,
            tax_percent=form.tax_percent.data or 18,
            internal_notes=form.internal_notes.data,
            terms_conditions=form.terms_conditions.data,
            created_by=current_user.id
        )
        
        db.session.add(quotation)
        db.session.commit()
        
        flash(f'Quotation "{quotation.quotation_number}" created.', 'success')
        return redirect(url_for('quotation.view', id=quotation.id))
    
    # Get default terms
    default_terms = """1. Prices are exclusive of GST
2. Delivery: Ex-works
3. Payment: 100% advance
4. Validity: 30 days from quotation date
5. Tooling: To be quoted separately if applicable"""
    
    return render_template('quotation/form.html', 
                         form=form, 
                         quotation=None,
                         product_id=product_id,
                         default_terms=default_terms)


@quotation_bp.route('/<int:id>')
@login_required
def view(id):
    """View quotation details."""
    quotation = Quotation.query.get_or_404(id)
    items = quotation.items.all()
    attachments = Attachment.query.filter_by(entity_type='quotation', entity_id=id).all()
    
    # Calculate totals
    subtotal = sum((item.quantity_1 or 0) * (item.unit_price_1 or 0) for item in items)
    tax_amount = subtotal * (quotation.tax_percent or 18) / 100
    total = subtotal + tax_amount
    
    return render_template('quotation/view.html', 
                         quotation=quotation, 
                         items=items,
                         attachments=attachments,
                         subtotal=subtotal,
                         tax_amount=tax_amount,
                         total=total)


@quotation_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit quotation."""
    quotation = Quotation.query.get_or_404(id)
    
    if quotation.status not in ['draft']:
        flash('Only draft quotations can be edited.', 'warning')
        return redirect(url_for('quotation.view', id=id))
    
    form = QuotationForm(obj=quotation)
    
    if form.validate_on_submit():
        quotation.customer_id = form.customer_id.data.id
        quotation.quotation_date = form.quotation_date.data
        quotation.valid_until = form.valid_until.data
        quotation.priority = form.priority.data
        quotation.currency = form.currency.data
        quotation.exchange_rate = form.exchange_rate.data or 1.0
        quotation.billing_address = form.billing_address.data
        quotation.shipping_address = form.shipping_address.data
        quotation.payment_terms = form.payment_terms.data
        quotation.delivery_terms = form.delivery_terms.data
        quotation.warranty = form.warranty.data
        quotation.tax_percent = form.tax_percent.data or 18
        quotation.internal_notes = form.internal_notes.data
        quotation.terms_conditions = form.terms_conditions.data
        
        db.session.commit()
        
        flash(f'Quotation "{quotation.quotation_number}" updated.', 'success')
        return redirect(url_for('quotation.view', id=id))
    
    return render_template('quotation/form.html', form=form, quotation=quotation)


@quotation_bp.route('/<int:id>/item/add', methods=['GET', 'POST'])
@login_required
def add_item(id):
    """Add item to quotation."""
    quotation = Quotation.query.get_or_404(id)
    
    if request.method == 'POST':
        product_id = request.form.get('product_id', type=int)
        product = Product.query.get(product_id) if product_id else None
        
        item = QuotationItem(
            quotation_id=quotation.id,
            product_id=product_id,
            part_description=request.form.get('part_description'),
            drawing_number=request.form.get('drawing_number') or (product.drawing_number if product else ''),
            material=request.form.get('material') or (product.material if product else ''),
            quantity_1=request.form.get('quantity_1', type=int) or 100,
            unit_price_1=request.form.get('unit_price_1', type=float) or 0,
            tooling_cost=request.form.get('tooling_cost', type=float) or 0,
            sequence=quotation.items.count()
        )
        
        # Copy from product if available
        if product and not item.part_description:
            item.part_description = product.product_name
        
        db.session.add(item)
        db.session.commit()
        
        # Update quotation totals
        update_quotation_totals(quotation)
        
        flash('Item added.', 'success')
    
    return redirect(url_for('quotation.view', id=id))


@quotation_bp.route('/item/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_item(id):
    """Edit quotation item."""
    item = QuotationItem.query.get_or_404(id)
    
    if request.method == 'POST':
        item.part_description = request.form.get('part_description')
        item.drawing_number = request.form.get('drawing_number')
        item.material = request.form.get('material')
        item.quantity_1 = request.form.get('quantity_1', type=int)
        item.quantity_2 = request.form.get('quantity_2', type=int)
        item.quantity_3 = request.form.get('quantity_3', type=int)
        item.unit_price_1 = request.form.get('unit_price_1', type=float)
        item.unit_price_2 = request.form.get('unit_price_2', type=float)
        item.unit_price_3 = request.form.get('unit_price_3', type=float)
        item.tooling_cost = request.form.get('tooling_cost', type=float)
        item.material_cost = request.form.get('material_cost', type=float)
        item.machining_cost = request.form.get('machining_cost', type=float)
        
        db.session.commit()
        
        # Update quotation totals
        update_quotation_totals(item.quotation)
        
        flash('Item updated.', 'success')
        return redirect(url_for('quotation.view', id=item.quotation_id))
    
    return render_template('quotation/item_form.html', item=item)


@quotation_bp.route('/item/<int:id>/delete', methods=['POST'])
@login_required
def delete_item(id):
    """Delete quotation item."""
    item = QuotationItem.query.get_or_404(id)
    quotation_id = item.quotation_id
    
    db.session.delete(item)
    db.session.commit()
    
    # Update quotation totals
    quotation = Quotation.query.get(quotation_id)
    update_quotation_totals(quotation)
    
    flash('Item deleted.', 'success')
    return redirect(url_for('quotation.view', id=quotation_id))


@quotation_bp.route('/<int:id>/status/<status>')
@login_required
def change_status(id, status):
    """Change quotation status."""
    quotation = Quotation.query.get_or_404(id)
    
    old_status = quotation.status
    quotation.status = status
    
    if status == 'sent':
        quotation.sent_date = datetime.utcnow()
    elif status == 'won':
        quotation.won_date = datetime.utcnow()
    elif status == 'lost':
        quotation.lost_date = datetime.utcnow()
    
    db.session.commit()
    
    flash(f'Quotation status changed to {status}.', 'success')
    return redirect(url_for('quotation.view', id=id))


@quotation_bp.route('/<int:id>/revise', methods=['GET', 'POST'])
@login_required
def revise(id):
    """Create revision of quotation."""
    original = Quotation.query.get_or_404(id)
    
    # Get next version number
    version_num = 1
    existing = Quotation.query.filter_by(quotation_ref=original.quotation_number).all()
    if existing:
        versions = [int(q.quotation_version.replace('V', '')) for q in existing]
        version_num = max(versions) + 1
    
    new_quote = Quotation(
        quotation_number=generate_quotation_number(),
        quotation_version=f'V{version_num}',
        quotation_ref=original.quotation_number,
        customer_id=original.customer_id,
        quotation_date=datetime.now().date(),
        valid_until=(datetime.now() + timedelta(days=30)).date(),
        status='draft',
        priority=original.priority,
        currency=original.currency,
        exchange_rate=original.exchange_rate,
        billing_address=original.billing_address,
        shipping_address=original.shipping_address,
        payment_terms=original.payment_terms,
        delivery_terms=original.delivery_terms,
        warranty=original.warranty,
        tax_percent=original.tax_percent,
        terms_conditions=original.terms_conditions,
        created_by=current_user.id
    )
    
    db.session.add(new_quote)
    db.session.commit()
    
    # Copy items
    for item in original.items.all():
        new_item = QuotationItem(
            quotation_id=new_quote.id,
            product_id=item.product_id,
            part_description=item.part_description,
            drawing_number=item.drawing_number,
            material=item.material,
            quantity_1=item.quantity_1,
            quantity_2=item.quantity_2,
            quantity_3=item.quantity_3,
            unit_price_1=item.unit_price_1,
            unit_price_2=item.unit_price_2,
            unit_price_3=item.unit_price_3,
            tooling_cost=item.tooling_cost,
            material_cost=item.material_cost,
            machining_cost=item.machining_cost,
            sequence=item.sequence
        )
        db.session.add(new_item)
    
    # Update original status
    original.status = 'revised'
    db.session.commit()
    
    flash(f'Revision {new_quote.quotation_version} created.', 'success')
    return redirect(url_for('quotation.view', id=new_quote.id))


@quotation_bp.route('/<int:id>/pdf')
@login_required
def pdf(id):
    """Generate PDF for quotation."""
    from app.utils import generate_quotation_pdf
    
    quotation = Quotation.query.get_or_404(id)
    pdf_buffer = generate_quotation_pdf(quotation)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{quotation.quotation_number}.pdf'
    )


@quotation_bp.route('/export')
@login_required
def export():
    """Export quotations to Excel."""
    quotations = Quotation.query.order_by(Quotation.quotation_date.desc()).all()
    output = export_quotations_to_excel(quotations)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'quotations_export_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )


@quotation_bp.route('/<int:id>/cost-item', methods=['POST'])
@login_required
def cost_item(id):
    """Calculate cost for quotation item."""
    quotation_id = request.form.get('quotation_id', type=int)
    material_cost = request.form.get('material_cost', type=float) or 0
    machining_cost = request.form.get('machining_cost', type=float) or 0
    vendor_cost = request.form.get('vendor_cost', type=float) or 0
    tooling_cost = request.form.get('tooling_cost', type=float) or 0
    overhead = float(current_app.config.get('DEFAULT_OVERHEAD_PERCENT', 15))
    profit = float(current_app.config.get('DEFAULT_PROFIT_PERCENT', 20))
    scrap = float(current_app.config.get('DEFAULT_SCRAP_PERCENT', 2))
    
    result = calculate_total_cost(
        material_cost, machining_cost, vendor_cost, tooling_cost,
        overhead, profit, scrap
    )
    
    return jsonify(result)


def update_quotation_totals(quotation):
    """Update quotation subtotal, tax, and total."""
    items = quotation.items.all()
    
    subtotal = 0
    for item in items:
        qty = item.quantity_1 or 0
        price = item.unit_price_1 or 0
        subtotal += qty * price
    
    quotation.subtotal = subtotal
    quotation.tax_amount = subtotal * (quotation.tax_percent or 18) / 100
    quotation.total_amount = quotation.subtotal + quotation.tax_amount
    
    db.session.commit()
