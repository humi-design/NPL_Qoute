from datetime import datetime, date

from flask import (
    render_template, redirect, url_for, request, flash, jsonify, send_file
)
from flask_login import login_required, current_user
from sqlalchemy import or_, case

from app.vendor import vendor_bp
from app.models import (
    db, Vendor, VendorRate, VendorWorkOrder, VendorQualityIssue,
    Process, Product, QuotationOperation,
    VendorInvoice, VendorPayment,
)
from app.forms import (VendorForm, VendorWorkOrderForm, VendorQualityIssueForm,
                       PaymentAllocationForm)
from app.utils import (
    generate_vendor_code, generate_work_order_number, compute_vendor_summary,
    compute_vendor_financial_summary, referential_delete_block,
    export_vendors_to_excel, export_vendor_work_orders_to_excel
)

# Allowed sort columns. Whitelisting keeps user-supplied sort params away from
# the SQL layer.
VENDOR_SORT_FIELDS = {
    'code': Vendor.vendor_code,
    'name': Vendor.vendor_name,
    'contact': Vendor.contact_person,
    'lead_time': Vendor.lead_time,
    'created': Vendor.created_at,
}

WORK_ORDER_SORT_FIELDS = {
    'number': VendorWorkOrder.work_order_number,
    'ordered': VendorWorkOrder.quantity_ordered,
    'received': VendorWorkOrder.quantity_received,
    'order_date': VendorWorkOrder.order_date,
    'due_date': VendorWorkOrder.due_date,
    'status': VendorWorkOrder.status,
    'priority': VendorWorkOrder.priority,
}

WORK_VIEWS = ('current', 'past', 'all')

# Status rule used by the Current / Past / All views (single source of truth).
CURRENT_STATUSES = VendorWorkOrder.OPEN_STATUSES
PAST_STATUSES = VendorWorkOrder.COMPLETE_STATUSES


def _apply_vendor_work_view(query, view):
    """Apply the Current / Past / All status rule to a work order query."""
    if view == 'current':
        return query.filter(VendorWorkOrder.status.in_(CURRENT_STATUSES))
    if view == 'past':
        return query.filter(VendorWorkOrder.status.in_(PAST_STATUSES))
    return query


def _nulls_last(column):
    """Portable NULLS LAST ordering (MySQL has no NULLS LAST syntax)."""
    return case((column.is_(None), 1), else_=0)


def _overdue_filter():
    """SQL condition matching open work orders whose due date has passed."""
    return db.and_(
        VendorWorkOrder.status.in_(CURRENT_STATUSES),
        VendorWorkOrder.due_date.isnot(None),
        VendorWorkOrder.due_date < date.today()
    )


def _vendor_query_from_request():
    """Build a Vendor query from the shared index/export request filters."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    work = request.args.get('work', '')

    query = Vendor.query

    if search:
        query = query.filter(
            or_(
                Vendor.vendor_name.ilike(f'%{search}%'),
                Vendor.vendor_code.ilike(f'%{search}%'),
                Vendor.contact_person.ilike(f'%{search}%'),
                Vendor.email.ilike(f'%{search}%'),
                Vendor.gst.ilike(f'%{search}%')
            )
        )

    if status == 'active':
        query = query.filter(Vendor.is_active == True)
    elif status == 'inactive':
        query = query.filter(Vendor.is_active == False)

    if work == 'current':
        query = query.filter(
            Vendor.work_orders.any(VendorWorkOrder.status.in_(CURRENT_STATUSES))
        )
    elif work == 'past':
        query = query.filter(
            Vendor.work_orders.any(VendorWorkOrder.status.in_(PAST_STATUSES))
        )
    elif work == 'overdue':
        query = query.filter(Vendor.work_orders.any(_overdue_filter()))

    return query


# ============================================================
# Vendor register
# ============================================================

@vendor_bp.route('/')
@login_required
def index():
    """Vendor list page with search, filters, sorting and pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    if per_page not in (10, 25, 50, 100):
        per_page = 25

    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    work = request.args.get('work', '')
    sort = request.args.get('sort', 'name')
    direction = request.args.get('dir', 'asc')

    query = _vendor_query_from_request()

    sort_col = VENDOR_SORT_FIELDS.get(sort, Vendor.vendor_name)
    if direction == 'desc':
        sort_col = sort_col.desc()

    vendors = query.order_by(sort_col).paginate(page=page, per_page=per_page, error_out=False)

    # Work summary for the vendors on this page only (avoids N+1 across all rows).
    v_summaries = {v.id: compute_vendor_summary(v) for v in vendors.items}

    return render_template(
        'vendor/index.html',
        vendors=vendors,
        v_summaries=v_summaries,
        search=search,
        status=status,
        work=work,
        sort=sort,
        dir=direction,
        per_page=per_page,
    )


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
    """View vendor details with Current / Past / All work views."""
    vendor = Vendor.query.get_or_404(id)
    rates = VendorRate.query.filter_by(vendor_id=id).order_by(
        VendorRate.effective_date.desc()
    ).all()

    work_view = request.args.get('work', 'current')
    if work_view not in WORK_VIEWS:
        work_view = 'current'

    work_query = VendorWorkOrder.query.filter_by(vendor_id=id)
    work_query = _apply_vendor_work_view(work_query, work_view)
    work_orders = work_query.order_by(
        _nulls_last(VendorWorkOrder.due_date),
        VendorWorkOrder.due_date.asc(),
        VendorWorkOrder.id.desc()
    ).all()

    # Manufacturing processes routed to this vendor (existing Process records)
    processes = vendor.processes.order_by(Process.process_name).all()

    # Quotation operations outsourced to this vendor
    quotation_operations = QuotationOperation.query.filter_by(vendor_id=id).order_by(
        QuotationOperation.id.desc()
    ).limit(50).all()

    quality_issues = VendorQualityIssue.query.filter_by(vendor_id=id).order_by(
        VendorQualityIssue.issue_date.desc()
    ).all()

    summary = compute_vendor_summary(vendor)
    financials = compute_vendor_financial_summary(vendor)
    invoices = VendorInvoice.query.filter_by(vendor_id=id).order_by(
        VendorInvoice.invoice_date.desc()).all()
    payments = VendorPayment.query.filter_by(vendor_id=id).order_by(
        VendorPayment.payment_date.desc()).all()
    allocation_form = PaymentAllocationForm()
    allocation_form.invoice_id.choices = [('', 'Select Invoice')] + [
        (str(i.id), f'{i.invoice_number} — outstanding {i.outstanding_amount:g}')
        for i in invoices if i.status != 'cancelled'
    ]

    from app.utils import _may_see_financials
    return render_template(
        'vendor/view.html',
        vendor=vendor,
        can_view_financials=_may_see_financials(current_user),
        rates=rates,
        work_orders=work_orders,
        work_view=work_view,
        processes=processes,
        quotation_operations=quotation_operations,
        quality_issues=quality_issues,
        summary=summary,
        financials=financials,
        invoices=invoices,
        payments=payments,
        allocation_form=allocation_form,
        now_date=date.today().isoformat(),
    )


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

    # Work orders cascade-delete with the vendor, but the app's existing policy
    # is to refuse while they exist (see tests/test_vendors.py), so keep that
    # check explicit.
    if vendor.work_orders.count() > 0:
        flash('Cannot delete vendor with existing work orders.', 'danger')
        return redirect(url_for('vendor.view', id=id))

    # Refuse while anything else still references the vendor (invoices,
    # payments, quotation/order operations, stock locations, cost entries).
    block = referential_delete_block(vendor)
    if block:
        flash(block, 'danger')
        return redirect(url_for('vendor.view', id=id))

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


@vendor_bp.route('/export')
@login_required
def export():
    """Export vendors (respecting current filters) with work summary to Excel."""
    vendors = _vendor_query_from_request().order_by(Vendor.vendor_name).all()
    output = export_vendors_to_excel(vendors)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'vendors_export_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )


@vendor_bp.route('/api/search')
@login_required
def api_search():
    """API endpoint to search vendors (used by autocomplete widgets)."""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])

    vendors = Vendor.query.filter(
        Vendor.is_active == True,
        or_(
            Vendor.vendor_name.ilike(f'%{query}%'),
            Vendor.vendor_code.ilike(f'%{query}%')
        )
    ).limit(10).all()

    return jsonify([
        {'id': v.id, 'code': v.vendor_code, 'name': v.vendor_name,
         'currency': v.default_currency}
        for v in vendors
    ])


# ============================================================
# Work order register (all vendors) with Current / Past / All
# ============================================================

@vendor_bp.route('/work-orders')
@login_required
def work_orders():
    """Global vendor work order register with search, filters and pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    if per_page not in (10, 25, 50, 100):
        per_page = 25

    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    work_view = request.args.get('work', 'all')
    if work_view not in WORK_VIEWS:
        work_view = 'all'
    vendor_id = request.args.get('vendor_id', type=int)
    overdue = request.args.get('overdue', '')
    sort = request.args.get('sort', 'due_date')
    direction = request.args.get('dir', 'asc')

    query = VendorWorkOrder.query
    query = _apply_vendor_work_view(query, work_view)

    if status:
        query = query.filter(VendorWorkOrder.status == status)

    if vendor_id:
        query = query.filter(VendorWorkOrder.vendor_id == vendor_id)

    if overdue == '1':
        query = query.filter(_overdue_filter())

    if search:
        query = query.filter(
            or_(
                VendorWorkOrder.work_order_number.ilike(f'%{search}%'),
                VendorWorkOrder.title.ilike(f'%{search}%'),
                VendorWorkOrder.part_description.ilike(f'%{search}%'),
                VendorWorkOrder.drawing_number.ilike(f'%{search}%'),
                VendorWorkOrder.service_name.ilike(f'%{search}%'),
                VendorWorkOrder.vendor.has(Vendor.vendor_name.ilike(f'%{search}%'))
            )
        )

    sort_col = WORK_ORDER_SORT_FIELDS.get(sort, VendorWorkOrder.due_date)
    if sort == 'due_date':
        order_by = (_nulls_last(VendorWorkOrder.due_date),
                    sort_col.desc() if direction == 'desc' else sort_col.asc())
    else:
        order_by = (sort_col.desc() if direction == 'desc' else sort_col.asc(),)

    work_orders_page = query.order_by(*order_by, VendorWorkOrder.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    vendors = Vendor.query.order_by(Vendor.vendor_name).all()

    return render_template(
        'vendor/work_orders.html',
        work_orders=work_orders_page,
        vendors=vendors,
        search=search,
        status=status,
        work_view=work_view,
        vendor_id=vendor_id,
        overdue=overdue,
        sort=sort,
        dir=direction,
        per_page=per_page,
    )


def _populate_work_order_choices(form):
    """Populate vendor/process/product select choices on the work order form."""
    form.vendor_id.choices = [('', 'Select Vendor')] + [
        (str(v.id), f'{v.vendor_code} - {v.vendor_name}')
        for v in Vendor.query.filter_by(is_active=True).order_by(Vendor.vendor_name).all()
    ]

    form.process_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.process_code} - {p.process_name}')
        for p in Process.query.filter_by(process_type='vendor').order_by(Process.process_name).all()
    ]

    form.product_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.internal_code} - {p.product_name}')
        for p in Product.query.order_by(Product.product_name).limit(500).all()
    ]


def _apply_work_order_form(form, work_order):
    """Copy submitted form data onto a work order instance.

    Returns False when the submitted vendor does not exist, so callers can
    reject tampered/invalid input instead of raising a ValueError.
    """
    try:
        vendor_id = int(form.vendor_id.data)
    except (TypeError, ValueError):
        return False
    if not Vendor.query.get(vendor_id):
        return False

    work_order.vendor_id = vendor_id
    work_order.process_id = int(form.process_id.data) if form.process_id.data else None
    work_order.product_id = int(form.product_id.data) if form.product_id.data else None
    work_order.title = form.title.data
    work_order.part_description = form.part_description.data
    work_order.drawing_number = form.drawing_number.data
    work_order.service_name = form.service_name.data
    work_order.quantity_ordered = form.quantity_ordered.data or 0
    work_order.quantity_received = form.quantity_received.data or 0
    work_order.quantity_rejected = form.quantity_rejected.data or 0
    work_order.unit = form.unit.data
    work_order.rate = form.rate.data or 0
    work_order.currency = form.currency.data
    work_order.order_date = form.order_date.data
    work_order.due_date = form.due_date.data
    work_order.status = form.status.data
    work_order.priority = form.priority.data
    work_order.notes = form.notes.data

    # Keep the completed date consistent with the status.
    if work_order.status == 'completed':
        work_order.completed_date = form.completed_date.data or date.today()
    else:
        work_order.completed_date = form.completed_date.data

    return True


@vendor_bp.route('/work-orders/new', methods=['GET', 'POST'])
@login_required
def new_work_order():
    """Create a new vendor work order."""
    form = VendorWorkOrderForm()
    _populate_work_order_choices(form)

    preselected_vendor = request.args.get('vendor_id', type=int)

    if request.method == 'GET':
        form.work_order_number.data = generate_work_order_number()
        form.order_date.data = date.today()
        if preselected_vendor:
            form.vendor_id.data = str(preselected_vendor)

    if form.validate_on_submit():
        work_order = VendorWorkOrder(
            work_order_number=form.work_order_number.data,
            created_by=current_user.id
        )
        if not _apply_work_order_form(form, work_order):
            flash('Please select a valid vendor.', 'danger')
            return render_template('vendor/work_order_form.html', form=form, work_order=None)

        db.session.add(work_order)
        db.session.commit()

        flash(f'Work order "{work_order.work_order_number}" created.', 'success')
        return redirect(url_for('vendor.view', id=work_order.vendor_id, work='all'))

    return render_template('vendor/work_order_form.html', form=form, work_order=None)


@vendor_bp.route('/work-orders/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_work_order(id):
    """Edit a vendor work order."""
    work_order = VendorWorkOrder.query.get_or_404(id)
    form = VendorWorkOrderForm(obj=work_order)
    _populate_work_order_choices(form)

    if form.validate_on_submit():
        if not _apply_work_order_form(form, work_order):
            flash('Please select a valid vendor.', 'danger')
            return render_template('vendor/work_order_form.html', form=form, work_order=work_order)

        db.session.commit()

        flash(f'Work order "{work_order.work_order_number}" updated.', 'success')
        return redirect(url_for('vendor.view', id=work_order.vendor_id, work='all'))

    return render_template('vendor/work_order_form.html', form=form, work_order=work_order)


@vendor_bp.route('/work-orders/<int:id>/delete', methods=['POST'])
@login_required
def delete_work_order(id):
    """Delete a vendor work order."""
    work_order = VendorWorkOrder.query.get_or_404(id)
    vendor_id = work_order.vendor_id

    # Detach quality issues rather than deleting them with the work order.
    VendorQualityIssue.query.filter_by(work_order_id=id).update({'work_order_id': None})

    db.session.delete(work_order)
    db.session.commit()
    flash('Work order deleted.', 'success')
    return redirect(url_for('vendor.view', id=vendor_id, work='all'))


@vendor_bp.route('/work-orders/<int:id>/status/<status>', methods=['POST'])
@login_required
def change_work_order_status(id, status):
    """Change work order status (pending/in_progress/completed/cancelled)."""
    work_order = VendorWorkOrder.query.get_or_404(id)

    if status not in ('pending', 'in_progress', 'completed', 'cancelled'):
        flash('Invalid status.', 'danger')
        return redirect(url_for('vendor.view', id=work_order.vendor_id, work='all'))

    work_order.status = status
    if status == 'completed':
        work_order.completed_date = date.today()
    elif status == 'cancelled':
        work_order.completed_date = None

    db.session.commit()
    flash(f'Work order {work_order.work_order_number} marked {status.replace("_", " ")}.', 'success')
    return redirect(request.referrer or url_for('vendor.view', id=work_order.vendor_id, work='all'))


@vendor_bp.route('/work-orders/<int:id>/receive', methods=['POST'])
@login_required
def receive_work_order(id):
    """Record a goods-receipt quantity against a work order."""
    work_order = VendorWorkOrder.query.get_or_404(id)

    qty = request.form.get('quantity', type=int) or 0
    rejected = request.form.get('rejected', type=int) or 0
    if qty < 0 or rejected < 0:
        flash('Quantity cannot be negative.', 'danger')
        return redirect(url_for('vendor.view', id=work_order.vendor_id, work='all'))

    work_order.quantity_received = (work_order.quantity_received or 0) + qty
    work_order.quantity_rejected = (work_order.quantity_rejected or 0) + rejected

    # Auto-complete when the full ordered quantity has been received.
    if (work_order.quantity_ordered or 0) > 0 and \
            work_order.quantity_received >= work_order.quantity_ordered and \
            work_order.status in CURRENT_STATUSES:
        work_order.status = 'completed'
        work_order.completed_date = date.today()
    elif work_order.status == 'pending':
        work_order.status = 'in_progress'

    db.session.commit()
    flash(f'Receipt recorded for {work_order.work_order_number}.', 'success')
    return redirect(request.referrer or url_for('vendor.view', id=work_order.vendor_id, work='all'))


@vendor_bp.route('/work-orders/export')
@login_required
def export_work_orders():
    """Export vendor work orders to Excel."""
    query = VendorWorkOrder.query
    work_view = request.args.get('work', 'all')
    query = _apply_vendor_work_view(query, work_view if work_view in WORK_VIEWS else 'all')
    vendor_id = request.args.get('vendor_id', type=int)
    if vendor_id:
        query = query.filter(VendorWorkOrder.vendor_id == vendor_id)

    work_orders_list = query.order_by(
        _nulls_last(VendorWorkOrder.due_date), VendorWorkOrder.due_date.asc()
    ).all()
    output = export_vendor_work_orders_to_excel(work_orders_list)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'vendor_work_orders_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )


# ============================================================
# Quality issues
# ============================================================

def _populate_quality_choices(form, vendor):
    """Populate the vendor and work order selects on the quality issue form."""
    form.vendor_id.choices = [(str(vendor.id), f'{vendor.vendor_code} - {vendor.vendor_name}')]
    form.work_order_id.choices = [('', 'Not linked')] + [
        (str(w.id), f'{w.work_order_number} - {w.title or w.service_name or "Work"}')
        for w in vendor.work_orders.order_by(VendorWorkOrder.id.desc()).all()
    ]


@vendor_bp.route('/<int:id>/quality/new', methods=['GET', 'POST'])
@login_required
def new_quality_issue(id):
    """Log a quality issue against a vendor."""
    vendor = Vendor.query.get_or_404(id)
    form = VendorQualityIssueForm()
    _populate_quality_choices(form, vendor)

    if request.method == 'GET':
        form.issue_date.data = date.today()
        form.vendor_id.data = str(vendor.id)

    if form.validate_on_submit():
        issue = VendorQualityIssue(
            vendor_id=vendor.id,
            work_order_id=int(form.work_order_id.data) if form.work_order_id.data else None,
            issue_date=form.issue_date.data,
            issue_type=form.issue_type.data,
            severity=form.severity.data,
            description=form.description.data,
            quantity_affected=form.quantity_affected.data or 0,
            status=form.status.data,
            resolution=form.resolution.data,
            resolved_date=form.resolved_date.data,
            reported_by=current_user.id
        )
        db.session.add(issue)
        db.session.commit()

        flash('Quality issue logged.', 'success')
        return redirect(url_for('vendor.view', id=vendor.id))

    return render_template('vendor/quality_form.html', form=form, vendor=vendor, issue=None)


@vendor_bp.route('/quality/<int:issue_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_quality_issue(issue_id):
    """Edit a vendor quality issue."""
    issue = VendorQualityIssue.query.get_or_404(issue_id)
    vendor = issue.vendor
    form = VendorQualityIssueForm(obj=issue)
    _populate_quality_choices(form, vendor)

    if form.validate_on_submit():
        issue.work_order_id = int(form.work_order_id.data) if form.work_order_id.data else None
        issue.issue_date = form.issue_date.data
        issue.issue_type = form.issue_type.data
        issue.severity = form.severity.data
        issue.description = form.description.data
        issue.quantity_affected = form.quantity_affected.data or 0
        issue.status = form.status.data
        issue.resolution = form.resolution.data
        issue.resolved_date = form.resolved_date.data

        if issue.status in ('resolved', 'closed') and not issue.resolved_date:
            issue.resolved_date = date.today()

        db.session.commit()
        flash('Quality issue updated.', 'success')
        return redirect(url_for('vendor.view', id=vendor.id))

    return render_template('vendor/quality_form.html', form=form, vendor=vendor, issue=issue)


@vendor_bp.route('/quality/<int:issue_id>/delete', methods=['POST'])
@login_required
def delete_quality_issue(issue_id):
    """Delete a vendor quality issue."""
    issue = VendorQualityIssue.query.get_or_404(issue_id)
    vendor_id = issue.vendor_id
    db.session.delete(issue)
    db.session.commit()
    flash('Quality issue deleted.', 'success')
    return redirect(url_for('vendor.view', id=vendor_id))
