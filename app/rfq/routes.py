from flask import render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from app.rfq import rfq_bp
from app.models import db, RFQ, RFQItem, Customer, Attachment
from app.forms import RFQForm, RFQItemForm
from app.utils import generate_rfq_number, upload_file, delete_file


@rfq_bp.route('/')
@login_required
def index():
    """RFQ list page."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    status = request.args.get('status', '')
    priority = request.args.get('priority', '')
    
    query = RFQ.query
    
    if status:
        query = query.filter(RFQ.status == status)
    if priority:
        query = query.filter(RFQ.priority == priority)
    
    rfqs = query.order_by(RFQ.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('rfq/index.html', rfqs=rfqs, status=status, priority=priority)


@rfq_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new RFQ."""
    form = RFQForm()
    form.rfq_number.data = generate_rfq_number()
    
    # Pre-select customer if provided
    customer_id = request.args.get('customer', type=int)
    if customer_id:
        form.customer_id.data = Customer.query.get(customer_id)
    
    if form.validate_on_submit():
        rfq = RFQ(
            rfq_number=form.rfq_number.data,
            customer_id=form.customer_id.data.id,
            subject=form.subject.data,
            description=form.description.data,
            status=form.status.data,
            priority=form.priority.data,
            due_date=form.due_date.data,
            notes=form.notes.data
        )
        
        db.session.add(rfq)
        db.session.commit()
        
        # Handle file uploads
        files = request.files.getlist('attachments')
        for file in files:
            if file and file.filename:
                result = upload_file(file, 'rfqs', rfq.id)
                if result:
                    attachment = Attachment(
                        entity_type='rfq',
                        entity_id=rfq.id,
                        file_name=result['file_name'],
                        original_name=result['original_name'],
                        file_type=result['file_type'],
                        file_size=result['file_size'],
                        file_path=result['file_path'],
                        uploaded_by=current_user.id
                    )
                    db.session.add(attachment)
        db.session.commit()
        
        flash(f'RFQ "{rfq.rfq_number}" created successfully.', 'success')
        return redirect(url_for('rfq.view', id=rfq.id))
    
    return render_template('rfq/form.html', form=form, rfq=None)


@rfq_bp.route('/<int:id>')
@login_required
def view(id):
    """View RFQ details."""
    rfq = RFQ.query.get_or_404(id)
    attachments = Attachment.query.filter_by(entity_type='rfq', entity_id=id).all()
    return render_template('rfq/view.html', rfq=rfq, attachments=attachments)


@rfq_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit RFQ."""
    rfq = RFQ.query.get_or_404(id)
    form = RFQForm(obj=rfq)
    
    if form.validate_on_submit():
        rfq.customer_id = form.customer_id.data.id
        rfq.subject = form.subject.data
        rfq.description = form.description.data
        rfq.status = form.status.data
        rfq.priority = form.priority.data
        rfq.due_date = form.due_date.data
        rfq.notes = form.notes.data
        
        db.session.commit()
        
        flash(f'RFQ "{rfq.rfq_number}" updated.', 'success')
        return redirect(url_for('rfq.view', id=rfq.id))
    
    attachments = Attachment.query.filter_by(entity_type='rfq', entity_id=id).all()
    return render_template('rfq/form.html', form=form, rfq=rfq, attachments=attachments)


@rfq_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete RFQ."""
    rfq = RFQ.query.get_or_404(id)
    db.session.delete(rfq)
    db.session.commit()
    flash('RFQ deleted.', 'success')
    return redirect(url_for('rfq.index'))


@rfq_bp.route('/<int:id>/item/add', methods=['POST'])
@login_required
def add_item(id):
    """Add item to RFQ."""
    rfq = RFQ.query.get_or_404(id)
    
    item = RFQItem(
        rfq_id=rfq.id,
        part_description=request.form.get('part_description'),
        drawing_number=request.form.get('drawing_number'),
        material=request.form.get('material'),
        quantity=request.form.get('quantity', type=int),
        target_price=request.form.get('target_price', type=float),
        sequence=rfq.items.count()
    )
    
    db.session.add(item)
    db.session.commit()
    
    flash('Item added.', 'success')
    return redirect(url_for('rfq.view', id=rfq.id))


@rfq_bp.route('/<int:id>/item/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_item(id, item_id):
    """Delete RFQ item."""
    item = RFQItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    flash('Item deleted.', 'success')
    return redirect(url_for('rfq.view', id=id))
