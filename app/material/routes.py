from flask import render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required
from app.material import material_bp
from app.models import db, Material, MaterialRate
from app.forms import MaterialForm, MaterialRateForm
from app.utils import generate_material_code


@material_bp.route('/')
@login_required
def index():
    """Material list page."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    search = request.args.get('search', '')
    
    query = Material.query
    if search:
        query = query.filter(
            Material.material_name.ilike(f'%{search}%') |
            Material.material_code.ilike(f'%{search}%')
        )
    
    materials = query.order_by(Material.material_name).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('material/index.html', materials=materials, search=search)


@material_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new material."""
    form = MaterialForm()
    form.material_code.data = generate_material_code()
    
    if form.validate_on_submit():
        material = Material(
            material_code=form.material_code.data,
            material_name=form.material_name.data,
            density=form.density.data,
            grade=form.grade.data,
            hsn_code=form.hsn_code.data,
            default_supplier=form.default_supplier.data,
            default_rate=form.default_rate.data,
            rate_unit=form.rate_unit.data,
            remarks=form.remarks.data,
            is_active=form.is_active.data
        )
        
        db.session.add(material)
        db.session.commit()
        
        # Add initial rate
        if form.default_rate.data:
            rate = MaterialRate(
                material_id=material.id,
                rate=form.default_rate.data,
                unit=form.rate_unit.data,
                supplier=form.default_supplier.data,
                effective_date=material.created_at.date()
            )
            db.session.add(rate)
            db.session.commit()
        
        flash(f'Material "{material.material_name}" created.', 'success')
        return redirect(url_for('material.view', id=material.id))
    
    return render_template('material/form.html', form=form, material=None)


@material_bp.route('/<int:id>')
@login_required
def view(id):
    """View material details."""
    material = Material.query.get_or_404(id)
    rates = MaterialRate.query.filter_by(material_id=id).order_by(MaterialRate.effective_date.desc()).all()
    return render_template('material/view.html', material=material, rates=rates)


@material_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit material."""
    material = Material.query.get_or_404(id)
    form = MaterialForm(obj=material)
    
    if form.validate_on_submit():
        material.material_name = form.material_name.data
        material.density = form.density.data
        material.grade = form.grade.data
        material.hsn_code = form.hsn_code.data
        material.default_supplier = form.default_supplier.data
        material.default_rate = form.default_rate.data
        material.rate_unit = form.rate_unit.data
        material.remarks = form.remarks.data
        material.is_active = form.is_active.data
        
        db.session.commit()
        flash(f'Material "{material.material_name}" updated.', 'success')
        return redirect(url_for('material.view', id=id))
    
    return render_template('material/form.html', form=form, material=material)


@material_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete material."""
    material = Material.query.get_or_404(id)
    db.session.delete(material)
    db.session.commit()
    flash('Material deleted.', 'success')
    return redirect(url_for('material.index'))


@material_bp.route('/<int:id>/rate', methods=['POST'])
@login_required
def add_rate(id):
    """Add rate to material."""
    material = Material.query.get_or_404(id)
    
    rate = MaterialRate(
        material_id=material.id,
        rate=request.form.get('rate', type=float),
        unit=request.form.get('unit'),
        supplier=request.form.get('supplier'),
        effective_date=request.form.get('effective_date')
    )
    
    # Deactivate old rates
    MaterialRate.query.filter_by(material_id=id, is_current=True).update({'is_current': False})
    
    db.session.add(rate)
    material.default_rate = rate.rate
    material.rate_unit = rate.unit
    db.session.commit()
    
    flash('Rate updated.', 'success')
    return redirect(url_for('material.view', id=id))
