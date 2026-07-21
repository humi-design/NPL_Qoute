from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required
from app.vendor import vendor_bp
from app.models import db, Vendor, VendorRate
from app.forms import VendorForm, VendorRateForm
from app.utils import generate_vendor_code


@vendor_bp.route('/')
@login_required
def index():
    """Vendor list page."""
    vendors = Vendor.query.order_by(Vendor.vendor_name).all()
    return render_template('vendor/index.html', vendors=vendors)


@vendor_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new vendor."""
    form = VendorForm()
    form.vendor_code.data = generate_vendor_code()
    
    if form.validate_on_submit():
        vendor = Vendor(
            vendor_code=form.vendor_code.data,
            vendor_name=form.vendor_name.data,
            contact_person=form.contact_person.data,
            email=form.email.data,
            phone=form.phone.data,
            mobile=form.mobile.data,
            gst=form.gst.data,
            address=form.address.data,
            lead_time=form.lead_time.data,
            default_currency=form.default_currency.data,
            payment_terms=form.payment_terms.data,
            notes=form.notes.data,
            is_active=form.is_active.data
        )
        
        db.session.add(vendor)
        db.session.commit()
        
        flash(f'Vendor "{vendor.vendor_name}" created.', 'success')
        return redirect(url_for('vendor.view', id=vendor.id))
    
    return render_template('vendor/form.html', form=form, vendor=None)


@vendor_bp.route('/<int:id>')
@login_required
def view(id):
    """View vendor details."""
    vendor = Vendor.query.get_or_404(id)
    rates = VendorRate.query.filter_by(vendor_id=id).order_by(VendorRate.effective_date.desc()).all()
    return render_template('vendor/view.html', vendor=vendor, rates=rates)


@vendor_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit vendor."""
    vendor = Vendor.query.get_or_404(id)
    form = VendorForm(obj=vendor)
    
    if form.validate_on_submit():
        vendor.vendor_name = form.vendor_name.data
        vendor.contact_person = form.contact_person.data
        vendor.email = form.email.data
        vendor.phone = form.phone.data
        vendor.mobile = form.mobile.data
        vendor.gst = form.gst.data
        vendor.address = form.address.data
        vendor.lead_time = form.lead_time.data
        vendor.default_currency = form.default_currency.data
        vendor.payment_terms = form.payment_terms.data
        vendor.notes = form.notes.data
        vendor.is_active = form.is_active.data
        
        db.session.commit()
        flash(f'Vendor "{vendor.vendor_name}" updated.', 'success')
        return redirect(url_for('vendor.view', id=id))
    
    return render_template('vendor/form.html', form=form, vendor=vendor)


@vendor_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete vendor."""
    vendor = Vendor.query.get_or_404(id)
    db.session.delete(vendor)
    db.session.commit()
    flash('Vendor deleted.', 'success')
    return redirect(url_for('vendor.index'))


@vendor_bp.route('/<int:id>/rate', methods=['POST'])
@login_required
def add_rate(id):
    """Add rate to vendor."""
    vendor = Vendor.query.get_or_404(id)
    
    rate = VendorRate(
        vendor_id=vendor.id,
        service_name=request.form.get('service_name'),
        rate=request.form.get('rate', type=float),
        unit=request.form.get('unit'),
        min_quantity=request.form.get('min_quantity', type=int),
        lead_time_days=request.form.get('lead_time_days', type=int),
        effective_date=request.form.get('effective_date'),
        notes=request.form.get('notes')
    )
    
    VendorRate.query.filter_by(vendor_id=id, is_current=True).update({'is_current': False})
    
    db.session.add(rate)
    db.session.commit()
    
    flash('Rate added.', 'success')
    return redirect(url_for('vendor.view', id=id))
