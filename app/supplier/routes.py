"""Supplier Master routes.

Suppliers are the procurement-side master (raw materials/consumables),
kept separate from Vendors (subcontracting/services). Supports configurable
Supplier Types, supplied materials/grades, deactivate-instead-of-delete and
Excel export. Follows the same conventions as the customer/vendor modules.
"""
from datetime import datetime

from flask import (render_template, redirect, url_for, request, flash,
                   jsonify, send_file)
from flask_login import login_required, current_user
from sqlalchemy import or_, case

from app.supplier import supplier_bp
from app.models import (db, Supplier, SupplierType, SupplierMaterial, Material,
                        Attachment, SupplierPurchaseOrder, SupplierOrderReceipt,
                        SupplierOrderReceiptLine, SupplierInvoice, SupplierPayment)
from app.forms import (SupplierForm, SupplierTypeForm, SupplierMaterialForm,
                       PaymentAllocationForm)
from app.utils import (generate_supplier_code, generate_supplier_type_code,
                       compute_supplier_summary, compute_supplier_rate_history,
                       compute_supplier_procurement_summary,
                       export_suppliers_to_excel, upload_file, delete_file,
                       track_revision)

# Whitelisted sort keys -> model column (prevents arbitrary column injection).
SUPPLIER_SORT_FIELDS = {
    'code': Supplier.supplier_code,
    'name': Supplier.company_name,
    'type': Supplier.supplier_type_id,
    'contact': Supplier.contact_person,
    'city': Supplier.city,
    'lead_time': Supplier.lead_time,
    'created': Supplier.created_at,
}


def _nulls_last(column):
    """Portable NULLS LAST ordering (MySQL has no NULLS LAST)."""
    return case((column.is_(None), 1), else_=0)


def _supplier_query_from_request():
    """Build the supplier list query from request args (search/filter/sort)."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    type_id = request.args.get('type', type=int)
    sort = request.args.get('sort', 'name')
    direction = request.args.get('dir', 'asc')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)

    query = Supplier.query

    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Supplier.company_name.ilike(like),
            Supplier.supplier_code.ilike(like),
            Supplier.contact_person.ilike(like),
            Supplier.email.ilike(like),
            Supplier.city.ilike(like),
            Supplier.gst_vat.ilike(like),
        ))

    if status == 'active':
        query = query.filter(Supplier.is_active.is_(True))
    elif status == 'inactive':
        query = query.filter(Supplier.is_active.is_(False))

    if type_id:
        query = query.filter(Supplier.supplier_type_id == type_id)

    column = SUPPLIER_SORT_FIELDS.get(sort, Supplier.company_name)
    order = column.desc() if direction == 'desc' else column.asc()
    query = query.order_by(_nulls_last(column), order)

    if per_page not in (10, 25, 50, 100):
        per_page = 25

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    filters = {
        'search': search, 'status': status, 'type': type_id or '',
        'sort': sort, 'dir': direction, 'per_page': per_page,
    }
    return pagination, filters


def _populate_type_choices(form, include_id=None):
    """Fill supplier type choices, keeping any already-linked (inactive) type."""
    query = SupplierType.query
    if include_id:
        query = query.filter(or_(SupplierType.is_active.is_(True),
                                 SupplierType.id == include_id))
    else:
        query = query.filter(SupplierType.is_active.is_(True))
    types = query.order_by(SupplierType.sort_order, SupplierType.type_name).all()
    form.supplier_type_id.choices = [('', 'Select Type')] + [
        (str(t.id), t.type_name) for t in types
    ]


# ============================================================
# Supplier Types (configurable master, admin-managed)
# ============================================================

@supplier_bp.route('/types')
@login_required
def types():
    """List supplier types."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to manage supplier types.', 'danger')
        return redirect(url_for('supplier.index'))

    all_types = SupplierType.query.order_by(SupplierType.sort_order,
                                            SupplierType.type_name).all()
    counts = {t.id: t.suppliers.count() for t in all_types}
    return render_template('supplier/types.html', types=all_types, counts=counts)


@supplier_bp.route('/types/new', methods=['GET', 'POST'])
@login_required
def new_type():
    """Create a supplier type."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to manage supplier types.', 'danger')
        return redirect(url_for('supplier.index'))

    form = SupplierTypeForm()
    if request.method == 'GET':
        form.type_code.data = generate_supplier_type_code()

    if form.validate_on_submit():
        if SupplierType.query.filter_by(type_code=form.type_code.data).first():
            flash('Type code already exists.', 'danger')
            return render_template('supplier/type_form.html', form=form, supplier_type=None)

        st = SupplierType(
            type_code=form.type_code.data,
            type_name=form.type_name.data,
            description=form.description.data,
            sort_order=form.sort_order.data or 0,
            is_active=form.is_active.data,
        )
        db.session.add(st)
        db.session.commit()
        flash(f'Supplier type "{st.type_name}" created.', 'success')
        return redirect(url_for('supplier.types'))

    return render_template('supplier/type_form.html', form=form, supplier_type=None)


@supplier_bp.route('/types/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_type(id):
    """Edit a supplier type."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to manage supplier types.', 'danger')
        return redirect(url_for('supplier.index'))

    st = SupplierType.query.get_or_404(id)
    form = SupplierTypeForm(obj=st)

    if form.validate_on_submit():
        clash = SupplierType.query.filter(SupplierType.type_code == form.type_code.data,
                                          SupplierType.id != st.id).first()
        if clash:
            flash('Type code already exists.', 'danger')
            return render_template('supplier/type_form.html', form=form, supplier_type=st)

        st.type_code = form.type_code.data
        st.type_name = form.type_name.data
        st.description = form.description.data
        st.sort_order = form.sort_order.data or 0
        st.is_active = form.is_active.data
        db.session.commit()
        flash(f'Supplier type "{st.type_name}" updated.', 'success')
        return redirect(url_for('supplier.types'))

    return render_template('supplier/type_form.html', form=form, supplier_type=st)


@supplier_bp.route('/types/<int:id>/delete', methods=['POST'])
@login_required
def delete_type(id):
    """Delete a supplier type (blocked while suppliers reference it)."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to manage supplier types.', 'danger')
        return redirect(url_for('supplier.index'))

    st = SupplierType.query.get_or_404(id)
    if st.suppliers.count() > 0:
        flash('Cannot delete a supplier type that is in use. Deactivate it instead.', 'danger')
        return redirect(url_for('supplier.types'))

    db.session.delete(st)
    db.session.commit()
    flash('Supplier type deleted.', 'success')
    return redirect(url_for('supplier.types'))


# ============================================================
# Suppliers
# ============================================================

@supplier_bp.route('/')
@login_required
def index():
    """Supplier grid with search, filters, sorting and pagination."""
    pagination, filters = _supplier_query_from_request()
    summaries = {s.id: compute_supplier_summary(s) for s in pagination.items}
    types_list = SupplierType.query.order_by(SupplierType.sort_order,
                                             SupplierType.type_name).all()
    return render_template('supplier/index.html', suppliers=pagination,
                           summaries=summaries, types=types_list, **filters)


@supplier_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create a supplier."""
    form = SupplierForm()
    _populate_type_choices(form)
    if request.method == 'GET':
        form.supplier_code.data = generate_supplier_code()
        preselected = request.args.get('type', type=int)
        if preselected:
            form.supplier_type_id.data = str(preselected)

    if form.validate_on_submit():
        supplier = Supplier(
            supplier_code=form.supplier_code.data,
            company_name=form.company_name.data,
            supplier_type_id=int(form.supplier_type_id.data) if form.supplier_type_id.data else None,
            contact_person=form.contact_person.data,
            email=form.email.data,
            phone=form.phone.data,
            mobile=form.mobile.data,
            website=form.website.data,
            address_line1=form.address_line1.data,
            address_line2=form.address_line2.data,
            city=form.city.data,
            state=form.state.data,
            postal_code=form.postal_code.data,
            country=form.country.data or 'India',
            gst_vat=form.gst_vat.data,
            pan=form.pan.data,
            cin=form.cin.data,
            tax_id=form.tax_id.data,
            default_currency=form.default_currency.data,
            payment_terms=form.payment_terms.data,
            lead_time=form.lead_time.data,
            document_references=form.document_references.data,
            notes=form.notes.data,
            is_active=form.is_active.data,
        )
        db.session.add(supplier)
        db.session.commit()

        _save_attachments(supplier.id)
        flash(f'Supplier "{supplier.company_name}" created.', 'success')
        return redirect(url_for('supplier.view', id=supplier.id))

    return render_template('supplier/form.html', form=form, supplier=None, attachments=[])


@supplier_bp.route('/<int:id>')
@login_required
def view(id):
    """View a supplier with supplied materials and attachments."""
    supplier = Supplier.query.get_or_404(id)
    materials = supplier.supplied_materials.all()
    attachments = Attachment.query.filter_by(entity_type='supplier', entity_id=id).all()
    summary = compute_supplier_summary(supplier)
    rate_history = compute_supplier_rate_history(supplier)
    procurement = compute_supplier_procurement_summary(supplier)

    # Materials in the master whose default supplier is this supplier
    linked_materials = Material.query.filter_by(supplier_id=id).order_by(
        Material.material_name).all()

    # Purchase orders, receipts (with batch/heat), invoices and payments
    purchase_orders = SupplierPurchaseOrder.query.filter_by(supplier_id=id).order_by(
        SupplierPurchaseOrder.id.desc()).all()
    receipts = SupplierOrderReceipt.query.filter_by(supplier_id=id).order_by(
        SupplierOrderReceipt.receipt_date.desc()).all()
    batch_lines = (SupplierOrderReceiptLine.query
                   .join(SupplierOrderReceipt)
                   .filter(SupplierOrderReceipt.supplier_id == id)
                   .order_by(SupplierOrderReceiptLine.id.desc()).all())
    invoices = SupplierInvoice.query.filter_by(supplier_id=id).order_by(
        SupplierInvoice.invoice_date.desc()).all()
    payments = SupplierPayment.query.filter_by(supplier_id=id).order_by(
        SupplierPayment.payment_date.desc()).all()

    material_form = SupplierMaterialForm()
    material_form.material_id.choices = [('', 'Select Material')] + [
        (str(m.id), f'{m.material_code} - {m.material_name}')
        for m in Material.query.filter_by(is_active=True).order_by(Material.material_name).all()
    ]
    allocation_form = PaymentAllocationForm()
    allocation_form.invoice_id.choices = [('', 'Select Invoice')] + [
        (str(i.id), f'{i.invoice_number} — outstanding {i.outstanding_amount:g}')
        for i in invoices if i.status != 'cancelled'
    ]

    from app.utils import _may_see_financials
    return render_template('supplier/view.html', supplier=supplier,
                           can_view_financials=_may_see_financials(current_user),
                           materials=materials, attachments=attachments,
                           summary=summary, material_form=material_form,
                           rate_history=rate_history,
                           linked_materials=linked_materials,
                           procurement=procurement,
                           purchase_orders=purchase_orders,
                           receipts=receipts, batch_lines=batch_lines,
                           invoices=invoices, payments=payments,
                           allocation_form=allocation_form)


@supplier_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit a supplier."""
    supplier = Supplier.query.get_or_404(id)
    form = SupplierForm(obj=supplier)
    _populate_type_choices(form, include_id=supplier.supplier_type_id)

    if form.validate_on_submit():
        tracked = ['company_name', 'contact_person', 'email', 'phone', 'gst_vat',
                   'payment_terms', 'lead_time', 'is_active']
        for field in tracked:
            old_val = getattr(supplier, field)
            new_val = getattr(form, field).data
            if old_val != new_val:
                track_revision('supplier', supplier.id, field, old_val, new_val,
                               current_user.id)

        supplier.company_name = form.company_name.data
        supplier.supplier_type_id = int(form.supplier_type_id.data) if form.supplier_type_id.data else None
        supplier.contact_person = form.contact_person.data
        supplier.email = form.email.data
        supplier.phone = form.phone.data
        supplier.mobile = form.mobile.data
        supplier.website = form.website.data
        supplier.address_line1 = form.address_line1.data
        supplier.address_line2 = form.address_line2.data
        supplier.city = form.city.data
        supplier.state = form.state.data
        supplier.postal_code = form.postal_code.data
        supplier.country = form.country.data
        supplier.gst_vat = form.gst_vat.data
        supplier.pan = form.pan.data
        supplier.cin = form.cin.data
        supplier.tax_id = form.tax_id.data
        supplier.default_currency = form.default_currency.data
        supplier.payment_terms = form.payment_terms.data
        supplier.lead_time = form.lead_time.data
        supplier.document_references = form.document_references.data
        supplier.notes = form.notes.data
        supplier.is_active = form.is_active.data
        db.session.commit()

        _save_attachments(supplier.id)
        flash(f'Supplier "{supplier.company_name}" updated.', 'success')
        return redirect(url_for('supplier.view', id=supplier.id))

    attachments = Attachment.query.filter_by(entity_type='supplier', entity_id=id).all()
    return render_template('supplier/form.html', form=form, supplier=supplier,
                           attachments=attachments)


@supplier_bp.route('/<int:id>/deactivate', methods=['POST'])
@login_required
def deactivate(id):
    """Deactivate a supplier (non-destructive alternative to delete)."""
    supplier = Supplier.query.get_or_404(id)
    supplier.is_active = False
    db.session.commit()
    flash(f'Supplier "{supplier.company_name}" deactivated.', 'success')
    return redirect(url_for('supplier.view', id=id))


@supplier_bp.route('/<int:id>/activate', methods=['POST'])
@login_required
def activate(id):
    """Re-activate a supplier."""
    supplier = Supplier.query.get_or_404(id)
    supplier.is_active = True
    db.session.commit()
    flash(f'Supplier "{supplier.company_name}" activated.', 'success')
    return redirect(url_for('supplier.view', id=id))


@supplier_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete a supplier only when it has no linked materials or documents.

    Suppliers linked to transactions/materials are protected; use deactivate.
    """
    supplier = Supplier.query.get_or_404(id)

    if supplier.supplied_materials.count() > 0:
        flash('Cannot delete a supplier with linked materials. Deactivate it instead.', 'danger')
        return redirect(url_for('supplier.view', id=id))

    if Material.query.filter_by(supplier_id=id).count() > 0:
        flash('Cannot delete a supplier referenced by materials. Deactivate it instead.', 'danger')
        return redirect(url_for('supplier.view', id=id))

    _delete_attachments('supplier', id)
    db.session.delete(supplier)
    db.session.commit()
    flash(f'Supplier "{supplier.company_name}" deleted.', 'success')
    return redirect(url_for('supplier.index'))


# ============================================================
# Supplied materials
# ============================================================

@supplier_bp.route('/<int:id>/material', methods=['POST'])
@login_required
def add_material(id):
    """Add a supplied material to a supplier."""
    supplier = Supplier.query.get_or_404(id)
    form = SupplierMaterialForm()
    form.material_id.choices = [('', 'Select Material')] + [
        (str(m.id), f'{m.material_code} - {m.material_name}')
        for m in Material.query.filter_by(is_active=True).order_by(Material.material_name).all()
    ]

    if form.validate_on_submit():
        material_id = int(form.material_id.data) if form.material_id.data else None
        if material_id and SupplierMaterial.query.filter_by(
                supplier_id=id, material_id=material_id).first():
            flash('That material is already linked to this supplier.', 'warning')
            return redirect(url_for('supplier.view', id=id))

        link = SupplierMaterial(
            supplier_id=id,
            material_id=material_id,
            grade=form.grade.data,
            rate=form.rate.data,
            rate_unit=form.rate_unit.data,
            lead_time=form.lead_time.data,
            notes=form.notes.data,
            is_preferred=form.is_preferred.data,
        )
        db.session.add(link)
        db.session.commit()
        flash('Supplied material added.', 'success')
    else:
        flash('Please select a valid material.', 'danger')

    return redirect(url_for('supplier.view', id=id))


@supplier_bp.route('/<int:id>/material/<int:link_id>/delete', methods=['POST'])
@login_required
def delete_material(id, link_id):
    """Remove a supplied-material link."""
    link = SupplierMaterial.query.get_or_404(link_id)
    if link.supplier_id != id:
        flash('Invalid material link.', 'danger')
        return redirect(url_for('supplier.view', id=id))

    db.session.delete(link)
    db.session.commit()
    flash('Supplied material removed.', 'success')
    return redirect(url_for('supplier.view', id=id))


# ============================================================
# Export + API
# ============================================================

@supplier_bp.route('/export')
@login_required
def export():
    """Export the filtered supplier list to Excel."""
    pagination, filters = _supplier_query_from_request()
    output = export_suppliers_to_excel(pagination.items)
    filename = f'suppliers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True, download_name=filename)


@supplier_bp.route('/api/search')
@login_required
def api_search():
    """Typeahead search for suppliers (used by other modules)."""
    term = request.args.get('q', '').strip()
    if len(term) < 2:
        return jsonify([])

    like = f'%{term}%'
    results = Supplier.query.filter(
        Supplier.is_active.is_(True),
        or_(Supplier.company_name.ilike(like), Supplier.supplier_code.ilike(like))
    ).order_by(Supplier.company_name).limit(20).all()

    return jsonify([
        {'id': s.id, 'code': s.supplier_code, 'name': s.company_name,
         'type': s.supplier_type.type_name if s.supplier_type else None,
         'currency': s.default_currency, 'lead_time': s.lead_time}
        for s in results
    ])


# ============================================================
# Attachment helpers
# ============================================================

def _save_attachments(supplier_id):
    """Persist any uploaded files against the supplier record."""
    files = request.files.getlist('attachments')
    for file in files:
        if file and file.filename:
            result = upload_file(file, 'suppliers', supplier_id)
            if result:
                db.session.add(Attachment(
                    entity_type='supplier',
                    entity_id=supplier_id,
                    file_name=result['file_name'],
                    original_name=result['original_name'],
                    file_type=result['file_type'],
                    file_size=result['file_size'],
                    file_path=result['file_path'],
                    uploaded_by=current_user.id,
                ))
    db.session.commit()


def _delete_attachments(entity_type, entity_id):
    """Delete attachment rows and their files for an entity."""
    for attachment in Attachment.query.filter_by(entity_type=entity_type,
                                                 entity_id=entity_id).all():
        delete_file(attachment.file_path)
        db.session.delete(attachment)
