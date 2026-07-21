from flask import render_template, redirect, url_for, request, flash, jsonify, send_file
from flask_login import login_required, current_user
from sqlalchemy import or_
from app.customer import customer_bp
from app.models import db, Customer, Quotation, Attachment
from app.forms import CustomerForm
from app.utils import generate_customer_code, upload_file, delete_file, export_customers_to_excel


@customer_bp.route('/')
@login_required
def index():
    """Customer list page."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    search = request.args.get('search', '')
    status = request.args.get('status', '')
    
    query = Customer.query
    
    if search:
        query = query.filter(
            or_(
                Customer.company_name.ilike(f'%{search}%'),
                Customer.customer_code.ilike(f'%{search}%'),
                Customer.contact_person.ilike(f'%{search}%'),
                Customer.email.ilike(f'%{search}%')
            )
        )
    
    if status == 'active':
        query = query.filter(Customer.is_active == True)
    elif status == 'inactive':
        query = query.filter(Customer.is_active == False)
    
    customers = query.order_by(Customer.company_name).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('customer/index.html', customers=customers, search=search, status=status)


@customer_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new customer."""
    form = CustomerForm()
    form.customer_code.data = generate_customer_code()
    
    if form.validate_on_submit():
        customer = Customer(
            customer_code=form.customer_code.data,
            company_name=form.company_name.data,
            contact_person=form.contact_person.data,
            email=form.email.data,
            phone=form.phone.data,
            mobile=form.mobile.data,
            gst_vat=form.gst_vat.data,
            country=form.country.data or 'India',
            currency=form.currency.data,
            payment_terms=form.payment_terms.data,
            incoterms=form.incoterms.data,
            shipping_terms=form.shipping_terms.data,
            billing_address=form.billing_address.data,
            shipping_address=form.shipping_address.data,
            delivery_address=form.delivery_address.data,
            notes=form.notes.data,
            is_active=form.is_active.data
        )
        
        db.session.add(customer)
        db.session.commit()
        
        # Handle file uploads
        files = request.files.getlist('attachments')
        for file in files:
            if file and file.filename:
                result = upload_file(file, 'customers', customer.id)
                if result:
                    attachment = Attachment(
                        entity_type='customer',
                        entity_id=customer.id,
                        file_name=result['file_name'],
                        original_name=result['original_name'],
                        file_type=result['file_type'],
                        file_size=result['file_size'],
                        file_path=result['file_path'],
                        uploaded_by=current_user.id
                    )
                    db.session.add(attachment)
        db.session.commit()
        
        flash(f'Customer "{customer.company_name}" created successfully.', 'success')
        return redirect(url_for('customer.view', id=customer.id))
    
    return render_template('customer/form.html', form=form, customer=None)


@customer_bp.route('/<int:id>')
@login_required
def view(id):
    """View customer details."""
    customer = Customer.query.get_or_404(id)
    
    # Get customer quotations
    quotations = Quotation.query.filter_by(customer_id=id).order_by(Quotation.quotation_date.desc()).limit(10).all()
    
    # Get attachments
    attachments = Attachment.query.filter_by(entity_type='customer', entity_id=id).all()
    
    return render_template('customer/view.html', customer=customer, quotations=quotations, attachments=attachments)


@customer_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit customer."""
    customer = Customer.query.get_or_404(id)
    form = CustomerForm(obj=customer)
    
    if form.validate_on_submit():
        # Track changes
        from app.utils import track_revision
        for field in ['company_name', 'contact_person', 'email', 'phone', 'gst_vat', 'country', 'currency', 'payment_terms']:
            old_val = getattr(customer, field)
            new_val = getattr(form, field).data
            if old_val != new_val:
                track_revision('customer', customer.id, field, old_val, new_val, current_user.id)
        
        customer.company_name = form.company_name.data
        customer.contact_person = form.contact_person.data
        customer.email = form.email.data
        customer.phone = form.phone.data
        customer.mobile = form.mobile.data
        customer.gst_vat = form.gst_vat.data
        customer.country = form.country.data
        customer.currency = form.currency.data
        customer.payment_terms = form.payment_terms.data
        customer.incoterms = form.incoterms.data
        customer.shipping_terms = form.shipping_terms.data
        customer.billing_address = form.billing_address.data
        customer.shipping_address = form.shipping_address.data
        customer.delivery_address = form.delivery_address.data
        customer.notes = form.notes.data
        customer.is_active = form.is_active.data
        
        db.session.commit()
        
        # Handle new file uploads
        files = request.files.getlist('attachments')
        for file in files:
            if file and file.filename:
                result = upload_file(file, 'customers', customer.id)
                if result:
                    attachment = Attachment(
                        entity_type='customer',
                        entity_id=customer.id,
                        file_name=result['file_name'],
                        original_name=result['original_name'],
                        file_type=result['file_type'],
                        file_size=result['file_size'],
                        file_path=result['file_path'],
                        uploaded_by=current_user.id
                    )
                    db.session.add(attachment)
        db.session.commit()
        
        flash(f'Customer "{customer.company_name}" updated successfully.', 'success')
        return redirect(url_for('customer.view', id=customer.id))
    
    attachments = Attachment.query.filter_by(entity_type='customer', entity_id=id).all()
    
    # Calculate customer statistics
    from sqlalchemy import func
    won_quotations = customer.quotations.filter_by(status='won')
    won_count = won_quotations.count()
    total_value = db.session.query(func.sum(Quotation.total_amount)).filter(
        Quotation.customer_id == id,
        Quotation.status == 'won'
    ).scalar() or 0
    
    return render_template('customer/form.html', form=form, customer=customer, 
                         attachments=attachments, won_count=won_count, total_value=total_value)


@customer_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete customer."""
    customer = Customer.query.get_or_404(id)
    
    # Check for existing quotations
    if customer.quotations.count() > 0:
        flash(f'Cannot delete customer with existing quotations.', 'danger')
        return redirect(url_for('customer.view', id=id))
    
    # Delete attachments
    attachments = Attachment.query.filter_by(entity_type='customer', entity_id=id).all()
    for attachment in attachments:
        delete_file(attachment.file_path)
        db.session.delete(attachment)
    
    db.session.delete(customer)
    db.session.commit()
    
    flash(f'Customer "{customer.company_name}" deleted.', 'success')
    return redirect(url_for('customer.index'))


@customer_bp.route('/<int:id>/attachment/<int:attach_id>/delete', methods=['POST'])
@login_required
def delete_attachment(id, attach_id):
    """Delete customer attachment."""
    attachment = Attachment.query.get_or_404(attach_id)
    delete_file(attachment.file_path)
    db.session.delete(attachment)
    db.session.commit()
    
    flash('Attachment deleted.', 'success')
    return redirect(url_for('customer.edit', id=id))


@customer_bp.route('/export')
@login_required
def export():
    """Export customers to Excel."""
    customers = Customer.query.order_by(Customer.company_name).all()
    output = export_customers_to_excel(customers)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'customers_export_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )


@customer_bp.route('/api/search')
@login_required
def api_search():
    """API endpoint to search customers."""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    customers = Customer.query.filter(
        Customer.is_active == True,
        or_(
            Customer.company_name.ilike(f'%{query}%'),
            Customer.customer_code.ilike(f'%{query}%')
        )
    ).limit(10).all()
    
    return jsonify([
        {'id': c.id, 'code': c.customer_code, 'name': c.company_name, 'currency': c.currency}
        for c in customers
    ])


from datetime import datetime
