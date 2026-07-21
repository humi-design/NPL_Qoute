from flask import render_template, redirect, url_for, request, flash, jsonify, send_file
from flask_login import login_required, current_user
from sqlalchemy import or_
from datetime import datetime
from app.product import product_bp
from app.models import db, Product, Attachment, ProductCostHistory, ProcessTemplate
from app.forms import ProductForm, ProductSearchForm
from app.utils import generate_product_code, upload_file, delete_file, export_products_to_excel, track_revision


@product_bp.route('/')
@login_required
def index():
    """Product list page."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    search = request.args.get('search', '')
    status = request.args.get('status', '')
    material = request.args.get('material', '')
    standard = request.args.get('standard', '')
    
    query = Product.query
    
    if search:
        query = query.filter(
            or_(
                Product.internal_code.ilike(f'%{search}%'),
                Product.product_name.ilike(f'%{search}%'),
                Product.drawing_number.ilike(f'%{search}%'),
                Product.customer_part_number.ilike(f'%{search}%'),
                Product.description.ilike(f'%{search}%')
            )
        )
    
    if status:
        query = query.filter(Product.status == status)
    
    if material:
        query = query.filter(Product.material.ilike(f'%{material}%'))
    
    if standard:
        query = query.filter(Product.standard.ilike(f'%{standard}%'))
    
    products = query.order_by(Product.updated_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get unique materials for filter
    materials = db.session.query(Product.material).filter(
        Product.material.isnot(None),
        Product.material != ''
    ).distinct().order_by(Product.material).all()
    materials = [m[0] for m in materials]
    
    return render_template('product/index.html', 
                         products=products, 
                         search=search, 
                         status=status,
                         material=material,
                         standard=standard,
                         materials=materials)


@product_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new product."""
    form = ProductForm()
    form.internal_code.data = generate_product_code()
    
    # Check if coming from search/duplicate
    duplicate_from = request.args.get('duplicate')
    if duplicate_from:
        source = Product.query.get(int(duplicate_from))
        if source:
            form = ProductForm(obj=source)
            form.internal_code.data = generate_product_code()
            form.product_name.data = f"{source.product_name} (Copy)"
    
    if form.validate_on_submit():
        product = Product(
            internal_code=form.internal_code.data,
            product_name=form.product_name.data,
            customer_part_number=form.customer_part_number.data,
            drawing_number=form.drawing_number.data,
            revision=form.revision.data or 'A',
            material=form.material.data,
            weight=form.weight.data,
            standard=form.standard.data,
            standard_number=form.standard_number.data,
            thread=form.thread.data,
            dimensions=form.dimensions.data,
            description=form.description.data,
            manufacturing_template_id=form.manufacturing_template_id.data.id if form.manufacturing_template_id.data else None,
            status=form.status.data
        )
        
        db.session.add(product)
        db.session.commit()
        
        # Handle file uploads
        files = request.files.getlist('attachments')
        for file in files:
            if file and file.filename:
                result = upload_file(file, 'products', product.id)
                if result:
                    attachment = Attachment(
                        entity_type='product',
                        entity_id=product.id,
                        file_name=result['file_name'],
                        original_name=result['original_name'],
                        file_type=result['file_type'],
                        file_size=result['file_size'],
                        file_path=result['file_path'],
                        uploaded_by=current_user.id
                    )
                    db.session.add(attachment)
        db.session.commit()
        
        flash(f'Product "{product.product_name}" created successfully.', 'success')
        return redirect(url_for('product.view', id=product.id))
    
    templates = ProcessTemplate.query.filter_by(is_active=True).all()
    return render_template('product/form.html', form=form, product=None, templates=templates)


@product_bp.route('/<int:id>')
@login_required
def view(id):
    """View product details."""
    product = Product.query.get_or_404(id)
    
    # Get attachments
    attachments = Attachment.query.filter_by(entity_type='product', entity_id=id).all()
    
    # Get cost history
    cost_history = ProductCostHistory.query.filter_by(product_id=id).order_by(
        ProductCostHistory.recorded_at.desc()
    ).limit(10).all()
    
    # Get quotation count
    quotation_count = product.quotations.count()
    
    return render_template('product/view.html', 
                         product=product, 
                         attachments=attachments,
                         cost_history=cost_history,
                         quotation_count=quotation_count)


@product_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit product."""
    product = Product.query.get_or_404(id)
    form = ProductForm(obj=product)
    
    if form.validate_on_submit():
        # Track changes
        changes = []
        for field in ['product_name', 'customer_part_number', 'drawing_number', 'revision', 
                     'material', 'weight', 'standard', 'standard_number', 'thread', 'status']:
            old_val = getattr(product, field)
            new_val = getattr(form, field).data
            if str(old_val) != str(new_val):
                changes.append((field, old_val, new_val))
                track_revision('product', product.id, field, old_val, new_val, current_user.id)
        
        product.product_name = form.product_name.data
        product.customer_part_number = form.customer_part_number.data
        product.drawing_number = form.drawing_number.data
        product.revision = form.revision.data or 'A'
        product.material = form.material.data
        product.weight = form.weight.data
        product.standard = form.standard.data
        product.standard_number = form.standard_number.data
        product.thread = form.thread.data
        product.dimensions = form.dimensions.data
        product.description = form.description.data
        product.manufacturing_template_id = form.manufacturing_template_id.data.id if form.manufacturing_template_id.data else None
        product.status = form.status.data
        
        db.session.commit()
        
        # Handle new file uploads
        files = request.files.getlist('attachments')
        for file in files:
            if file and file.filename:
                result = upload_file(file, 'products', product.id)
                if result:
                    attachment = Attachment(
                        entity_type='product',
                        entity_id=product.id,
                        file_name=result['file_name'],
                        original_name=result['original_name'],
                        file_type=result['file_type'],
                        file_size=result['file_size'],
                        file_path=result['file_path'],
                        uploaded_by=current_user.id
                    )
                    db.session.add(attachment)
        db.session.commit()
        
        flash(f'Product "{product.product_name}" updated successfully.', 'success')
        return redirect(url_for('product.view', id=product.id))
    
    attachments = Attachment.query.filter_by(entity_type='product', entity_id=id).all()
    templates = ProcessTemplate.query.filter_by(is_active=True).all()
    return render_template('product/form.html', form=form, product=product, templates=templates, attachments=attachments)


@product_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete product."""
    product = Product.query.get_or_404(id)
    
    # Check for existing quotations
    if product.quotations.count() > 0:
        flash(f'Cannot delete product with existing quotations.', 'danger')
        return redirect(url_for('product.view', id=id))
    
    # Delete attachments
    attachments = Attachment.query.filter_by(entity_type='product', entity_id=id).all()
    for attachment in attachments:
        delete_file(attachment.file_path)
        db.session.delete(attachment)
    
    db.session.delete(product)
    db.session.commit()
    
    flash(f'Product "{product.product_name}" deleted.', 'success')
    return redirect(url_for('product.index'))


@product_bp.route('/<int:id>/duplicate', methods=['GET', 'POST'])
@login_required
def duplicate(id):
    """Duplicate product."""
    source = Product.query.get_or_404(id)
    return redirect(url_for('product.new', duplicate=source.id))


@product_bp.route('/<int:id>/attachment/<int:attach_id>/delete', methods=['POST'])
@login_required
def delete_attachment(id, attach_id):
    """Delete product attachment."""
    attachment = Attachment.query.get_or_404(attach_id)
    delete_file(attachment.file_path)
    db.session.delete(attachment)
    db.session.commit()
    
    flash('Attachment deleted.', 'success')
    return redirect(url_for('product.edit', id=id))


@product_bp.route('/export')
@login_required
def export():
    """Export products to Excel."""
    products = Product.query.order_by(Product.updated_at.desc()).all()
    output = export_products_to_excel(products)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'products_export_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )


@product_bp.route('/api/search')
@login_required
def api_search():
    """API endpoint to search products."""
    query = request.args.get('q', '')
    material = request.args.get('material', '')
    standard = request.args.get('standard', '')
    
    q = Product.query.filter(Product.status == 'active')
    
    if query:
        q = q.filter(
            or_(
                Product.internal_code.ilike(f'%{query}%'),
                Product.product_name.ilike(f'%{query}%'),
                Product.drawing_number.ilike(f'%{query}%'),
                Product.customer_part_number.ilike(f'%{query}%')
            )
        )
    
    if material:
        q = q.filter(Product.material.ilike(f'%{material}%'))
    
    if standard:
        q = q.filter(Product.standard.ilike(f'%{standard}%'))
    
    products = q.limit(20).all()
    
    return jsonify([
        {
            'id': p.id,
            'code': p.internal_code,
            'name': p.product_name,
            'material': p.material,
            'drawing': p.drawing_number,
            'revision': p.revision
        }
        for p in products
    ])


@product_bp.route('/api/similar')
@login_required
def api_similar():
    """Find similar products."""
    product_id = request.args.get('id')
    product = Product.query.get(product_id) if product_id else None
    
    if not product:
        return jsonify([])
    
    # Find products with similar attributes
    similar = Product.query.filter(
        Product.id != product.id,
        Product.status == 'active',
        or_(
            Product.material == product.material,
            Product.standard == product.standard,
            Product.thread == product.thread
        )
    ).limit(10).all()
    
    return jsonify([
        {
            'id': p.id,
            'code': p.internal_code,
            'name': p.product_name,
            'material': p.material,
            'similarity': sum([
                p.material == product.material,
                p.standard == product.standard,
                p.thread == product.thread
            ])
        }
        for p in similar
    ])


@product_bp.route('/<int:id>/load-template', methods=['POST'])
@login_required
def load_template(id):
    """Load manufacturing template into product."""
    product = Product.query.get_or_404(id)
    template_id = request.form.get('template_id')
    
    if template_id:
        template = ProcessTemplate.query.get(int(template_id))
        if template:
            product.manufacturing_template_id = template.id
            db.session.commit()
            return jsonify({'success': True, 'message': f'Loaded template: {template.template_name}'})
    
    return jsonify({'success': False, 'message': 'Template not found'})
