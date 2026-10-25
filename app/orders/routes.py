"""Orders & manufacturing tracking routes.

Provides:
  * Order register with Current / Past / All views, search, filters, sort,
    pagination and status summaries.
  * Raw material tracker & allocation (lot-level balances, movement ledger,
    over-allocation and duplicate-movement guards).
  * Vendor process tracking (per-order route of operations, issue / return /
    accept / reject / rework / scrap with partial deliveries).
  * Quality inspection (incoming / in-process / final).
  * Final inspection & packaging, warehouse receipt and order completion.
  * A traceable order timeline from material procurement to dispatch.

Integrity rules enforced here:
  * A lot can never be over-allocated or over-issued (guarded against the
    derived available / remaining balances).
  * The same movement cannot be posted twice (DB unique constraint on
    type + reference) - duplicates are caught and reported, not silently added.
  * An operation cannot return / accept / reject / scrap more than it issued.
"""
from datetime import date, datetime

from flask import (render_template, redirect, url_for, request, flash, send_file)
from flask_login import login_required, current_user
from sqlalchemy import or_, case
from sqlalchemy.exc import IntegrityError

from app.orders import orders_bp
from app.models import (
    db, Order, OrderOperation, OperationTransaction, MaterialLot,
    MaterialMovement, Inspection, PackagingRecord,
    Customer, Product, Material, Supplier, Vendor, Process, Machine,
    SupplierPurchaseOrder, SupplierOrderReceipt, SupplierOrderReceiptLine,
    Quotation, QuotationItem, Dispatch, DispatchLine, Attachment,
)
from app.forms import (
    OrderForm, OrderOperationForm, OperationTransactionForm, MaterialLotForm,
    MaterialMovementForm, InspectionForm, PackagingRecordForm,
    DispatchForm, DispatchLineForm,
)
from app.utils import (
    notify_roles,
    generate_order_number, generate_inspection_number, generate_lot_number,
    compute_order_summary, compute_material_requirement, build_order_timeline,
    export_orders_to_excel, export_material_lots_to_excel,
    export_vendor_process_to_excel, compute_order_dispatch_summary,
    packaging_available, generate_dispatch_number, export_dispatches_to_excel,
)

# Whitelisted sort columns keep user-supplied sort params away from the SQL layer.
ORDER_SORT_FIELDS = {
    'number': Order.order_number,
    'customer': Order.customer_id,
    'drawing': Order.drawing_number,
    'qty': Order.quantity_ordered,
    'order_date': Order.order_date,
    'due_date': Order.due_date,
    'status': Order.status,
    'priority': Order.priority,
}
LOT_SORT_FIELDS = {
    'lot': MaterialLot.lot_number,
    'material': MaterialLot.material_id,
    'received': MaterialLot.quantity_received,
    'date': MaterialLot.receipt_date,
}
PROCESS_SORT_FIELDS = {
    'order': OrderOperation.order_id,
    'sequence': OrderOperation.sequence,
    'operation': OrderOperation.operation_name,
    'vendor': OrderOperation.vendor_id,
    'planned': OrderOperation.quantity_planned,
    'issued': OrderOperation.sequence,
    'due_date': OrderOperation.due_date,
    'status': OrderOperation.status,
}
PER_PAGE_CHOICES = (10, 25, 50, 100)

# Current / Past / All rule (single source of truth).
CURRENT_STATUSES = Order.OPEN_STATUSES
PAST_STATUSES = Order.COMPLETE_STATUSES

# Movement types that consume stock (checked against the available balance).
CONSUMING_MOVEMENTS = ('allocation', 'issue', 'scrap', 'shortage')


def _nulls_last(column):
    """Portable NULLS LAST ordering (MySQL has no NULLS LAST)."""
    return case((column.is_(None), 1), else_=0)


def _can_write():
    return current_user.has_permission('write')


def _require_write():
    if not _can_write():
        flash('You do not have permission to modify orders.', 'danger')
        return False
    return True


def _apply_order_view(query, view):
    """Apply the Current / Past / All status rule to an order query."""
    if view == 'current':
        return query.filter(Order.status.in_(CURRENT_STATUSES))
    if view == 'past':
        return query.filter(Order.status.in_(PAST_STATUSES))
    return query


def _paginate(query, default_sort, sort_fields, default_dir='desc'):
    """Apply whitelisted sort + pagination from request args (app convention)."""
    sort = request.args.get('sort', default_sort)
    direction = request.args.get('dir', default_dir)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)

    column = sort_fields.get(sort, sort_fields[default_sort])
    order = column.desc() if direction == 'desc' else column.asc()
    query = query.order_by(_nulls_last(column), order)

    if per_page not in PER_PAGE_CHOICES:
        per_page = 25

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return pagination, {'sort': sort, 'dir': direction, 'per_page': per_page}


# ============================================================
# Order register
# ============================================================

@orders_bp.route('/')
@login_required
def index():
    """Order register with Current / Past / All views, filters and summaries."""
    view = request.args.get('view', 'current')
    if view not in ('current', 'past', 'all'):
        view = 'current'

    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    customer_id = request.args.get('customer_id', type=int)
    overdue = request.args.get('overdue', '')

    query = Order.query
    query = _apply_order_view(query, view)

    if status:
        query = query.filter(Order.status == status)
    if customer_id:
        query = query.filter(Order.customer_id == customer_id)
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Order.order_number.ilike(like),
            Order.part_description.ilike(like),
            Order.drawing_number.ilike(like),
            Order.material_grade.ilike(like),
            Order.customer.has(Customer.company_name.ilike(like)),
        ))

    # Status summary tiles reflect the current view + search, before the
    # overdue filter narrows the list further.
    summary_rows = query.with_entities(Order.status).all()
    status_summary = {}
    for (s,) in summary_rows:
        status_summary[s] = status_summary.get(s, 0) + 1

    if overdue == '1':
        today = date.today()
        query = query.filter(Order.due_date.isnot(None), Order.due_date < today,
                             Order.status.in_(CURRENT_STATUSES))

    pagination, sortview = _paginate(query, 'order_date', ORDER_SORT_FIELDS)
    customers = Customer.query.order_by(Customer.company_name).all()

    return render_template('orders/index.html', orders=pagination, view=view,
                           search=search, status=status, customer_id=customer_id,
                           overdue=overdue, customers=customers,
                           status_summary=status_summary,
                           open_statuses=Order.OPEN_STATUSES, **sortview)


@orders_bp.route('/export')
@login_required
def export_orders():
    """Export the current order view to Excel."""
    view = request.args.get('view', 'all')
    if view not in ('current', 'past', 'all'):
        view = 'all'
    query = _apply_order_view(Order.query, view)
    customer_id = request.args.get('customer_id', type=int)
    if customer_id:
        query = query.filter(Order.customer_id == customer_id)
    orders_list = query.order_by(_nulls_last(Order.due_date),
                                 Order.due_date.asc()).all()
    output = export_orders_to_excel(orders_list)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'orders_{datetime.now().strftime("%Y%m%d")}.xlsx')


def _populate_order_choices(form, order=None):
    """Populate customer / quotation / product selects on the order form."""
    form.customer_id.choices = [('', 'Select Customer')] + [
        (str(c.id), f'{c.customer_code} - {c.company_name}')
        for c in Customer.query.order_by(Customer.company_name).all()
    ]
    form.quotation_id.choices = [('', 'Not linked')] + [
        (str(q.id), f'{q.quotation_number} ({q.status})')
        for q in Quotation.query.order_by(Quotation.id.desc()).limit(300).all()
    ]
    form.product_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.internal_code} - {p.product_name}')
        for p in Product.query.order_by(Product.product_name).limit(500).all()
    ]


@orders_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_order():
    """Create a new order (optionally prefilled from a won quotation)."""
    if not _require_write():
        return redirect(url_for('orders.index'))
    form = OrderForm()
    _populate_order_choices(form)

    # Prefill from a quotation so a won quote can be promoted to an order.
    quotation = None
    if request.method == 'GET':
        qid = request.args.get('quotation_id', type=int)
        if qid:
            quotation = Quotation.query.get(qid)
        form.order_date.data = date.today()
        if quotation:
            form.quotation_id.data = str(quotation.id)
            form.customer_id.data = str(quotation.customer_id)
            item = quotation.items.first()
            if item:
                form.part_description.data = item.part_description
                form.drawing_number.data = item.drawing_number
                form.material_grade.data = item.material
                form.quantity_ordered.data = (item.quantity_1 or 0) or None
                form.product_id.data = str(item.product_id) if item.product_id else ''

    if form.validate_on_submit():
        customer = Customer.query.get(form.customer_id.data)
        if not customer:
            flash('Select a valid customer.', 'danger')
            return render_template('orders/form.html', form=form, order=None)
        order = Order(
            order_number=generate_order_number(),
            customer_id=customer.id,
            quotation_id=int(form.quotation_id.data) if form.quotation_id.data else None,
            product_id=int(form.product_id.data) if form.product_id.data else None,
            part_description=form.part_description.data,
            drawing_number=form.drawing_number.data,
            revision=form.revision.data,
            specifications=form.specifications.data,
            material_grade=form.material_grade.data,
            standard=form.standard.data,
            quantity_ordered=form.quantity_ordered.data or 0,
            unit=form.unit.data,
            rate=form.rate.data or 0,
            currency=form.currency.data,
            order_date=form.order_date.data,
            due_date=form.due_date.data,
            planned_start=form.planned_start.data,
            actual_start=form.actual_start.data,
            actual_completion=form.actual_completion.data,
            status=form.status.data,
            priority=form.priority.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(order)
        db.session.commit()
        notify_roles(
            ['admin', 'production'],
            f'New order {order.order_number}',
            f'{order.quantity_ordered} {order.unit} for '
            f'{order.customer.company_name if order.customer else "-"}'
            + (f', due {order.due_date}' if order.due_date else ''),
            'order_created', 'order', order.id)
        db.session.commit()
        flash(f'Order {order.order_number} created.', 'success')
        return redirect(url_for('orders.view_order', id=order.id))

    return render_template('orders/form.html', form=form, order=None)


@orders_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_order(id):
    """Edit an order's header, planned values and status."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    form = OrderForm(obj=order)
    _populate_order_choices(form, order)
    if request.method == 'GET':
        form.customer_id.data = str(order.customer_id)
        form.quotation_id.data = str(order.quotation_id) if order.quotation_id else ''
        form.product_id.data = str(order.product_id) if order.product_id else ''

    if form.validate_on_submit():
        order.customer_id = int(form.customer_id.data)
        order.quotation_id = int(form.quotation_id.data) if form.quotation_id.data else None
        order.product_id = int(form.product_id.data) if form.product_id.data else None
        order.part_description = form.part_description.data
        order.drawing_number = form.drawing_number.data
        order.revision = form.revision.data
        order.specifications = form.specifications.data
        order.material_grade = form.material_grade.data
        order.standard = form.standard.data
        order.quantity_ordered = form.quantity_ordered.data or 0
        order.unit = form.unit.data
        order.rate = form.rate.data or 0
        order.currency = form.currency.data
        order.order_date = form.order_date.data
        order.due_date = form.due_date.data
        order.planned_start = form.planned_start.data
        order.actual_start = form.actual_start.data
        order.actual_completion = form.actual_completion.data
        order.status = form.status.data
        order.priority = form.priority.data
        order.notes = form.notes.data
        db.session.commit()
        flash(f'Order {order.order_number} updated.', 'success')
        return redirect(url_for('orders.view_order', id=id))

    return render_template('orders/form.html', form=form, order=order)


@orders_bp.route('/<int:id>')
@login_required
def view_order(id):
    """Order detail: summary, route, lots, inspections, packaging, timeline."""
    order = Order.query.get_or_404(id)
    summary = compute_order_summary(order)
    requirement = compute_material_requirement(order)
    timeline = build_order_timeline(order)
    op_form = OrderOperationForm()
    _populate_operation_choices(op_form)
    dispatch_summary = compute_order_dispatch_summary(order)
    documents = Attachment.query.filter_by(
        entity_type='order', entity_id=order.id).all()
    return render_template('orders/view.html', order=order, summary=summary,
                           requirement=requirement, timeline=timeline,
                           op_form=op_form, dispatch_summary=dispatch_summary,
                           documents=documents)


@orders_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete_order(id):
    """Delete an order only when it has no operations, lots or inspections."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    if order.operations.count() or order.lots.count() \
            or order.inspections.count() or order.packaging.count():
        flash('Cannot delete an order with a route, lots, inspections or '
              'packaging. Cancel it instead.', 'danger')
        return redirect(url_for('orders.view_order', id=id))
    db.session.delete(order)
    db.session.commit()
    flash('Order deleted.', 'success')
    return redirect(url_for('orders.index'))


@orders_bp.route('/<int:id>/status/<status>', methods=['POST'])
@login_required
def set_order_status(id, status):
    """Move an order between statuses (e.g. put on hold, cancel, complete)."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    if status not in ('draft', 'confirmed', 'in_production', 'completed',
                      'on_hold', 'cancelled'):
        flash('Unknown status.', 'danger')
        return redirect(url_for('orders.view_order', id=id))
    order.status = status
    if status == 'in_production' and not order.actual_start:
        order.actual_start = date.today()
    if status == 'completed' and not order.actual_completion:
        order.actual_completion = date.today()
    db.session.commit()
    flash(f'Order status changed to {status.replace("_", " ")}.', 'success')
    return redirect(url_for('orders.view_order', id=id))


# ============================================================
# Order operations (the order's manufacturing route)
# ============================================================

def _populate_operation_choices(form):
    form.process_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.process_code} - {p.process_name}')
        for p in Process.query.order_by(Process.process_name).all()
    ]
    form.vendor_id.choices = [('', 'Not linked')] + [
        (str(v.id), f'{v.vendor_code} - {v.vendor_name}')
        for v in Vendor.query.filter_by(is_active=True).order_by(Vendor.vendor_name).all()
    ]
    form.machine_id.choices = [('', 'Not linked')] + [
        (str(m.id), f'{m.machine_code} - {m.machine_name}')
        for m in Machine.query.order_by(Machine.machine_name).all()
    ]


@orders_bp.route('/<int:id>/operation/add', methods=['POST'])
@login_required
def add_operation(id):
    """Add a step to the order's route (route is per-order, not a template)."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    form = OrderOperationForm()
    _populate_operation_choices(form)
    if not form.validate_on_submit():
        flash('Check the operation values.', 'danger')
        return redirect(url_for('orders.view_order', id=id))

    is_vendor = bool(form.is_vendor.data)
    vendor_id = int(form.vendor_id.data) if form.vendor_id.data else None
    if is_vendor and not vendor_id:
        flash('Select a vendor for a subcontracted operation.', 'danger')
        return redirect(url_for('orders.view_order', id=id))

    op = OrderOperation(
        order_id=order.id,
        sequence=form.sequence.data,
        process_id=int(form.process_id.data) if form.process_id.data else None,
        operation_name=form.operation_name.data,
        is_vendor=is_vendor,
        vendor_id=vendor_id if is_vendor else None,
        machine_id=int(form.machine_id.data) if form.machine_id.data else None,
        quantity_planned=form.quantity_planned.data or order.quantity_ordered or 0,
        planned_issue_date=form.planned_issue_date.data,
        due_date=form.due_date.data,
        rate=form.rate.data or 0,
        actual_start=form.actual_start.data,
        actual_end=form.actual_end.data,
        status=form.status.data,
        is_final=bool(form.is_final.data),
        notes=form.notes.data,
    )
    db.session.add(op)
    db.session.commit()
    flash('Operation added to the route.', 'success')
    return redirect(url_for('orders.view_order', id=id))


@orders_bp.route('/operation/<int:op_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_operation(op_id):
    """Edit a route operation (planned vs actual, vendor, status)."""
    op = OrderOperation.query.get_or_404(op_id)
    order_id = op.order_id
    if not _require_write():
        return redirect(url_for('orders.view_order', id=order_id))
    form = OrderOperationForm(obj=op)
    _populate_operation_choices(form)
    if request.method == 'GET':
        form.process_id.data = str(op.process_id) if op.process_id else ''
        form.vendor_id.data = str(op.vendor_id) if op.vendor_id else ''
        form.machine_id.data = str(op.machine_id) if op.machine_id else ''
        form.is_vendor.data = op.is_vendor

    if form.validate_on_submit():
        is_vendor = bool(form.is_vendor.data)
        vendor_id = int(form.vendor_id.data) if form.vendor_id.data else None
        if is_vendor and not vendor_id:
            flash('Select a vendor for a subcontracted operation.', 'danger')
            return render_template('orders/operation_form.html', form=form, op=op)
        op.sequence = form.sequence.data
        op.process_id = int(form.process_id.data) if form.process_id.data else None
        op.operation_name = form.operation_name.data
        op.is_vendor = is_vendor
        op.vendor_id = vendor_id if is_vendor else None
        op.machine_id = int(form.machine_id.data) if form.machine_id.data else None
        op.quantity_planned = form.quantity_planned.data or 0
        op.planned_issue_date = form.planned_issue_date.data
        op.due_date = form.due_date.data
        op.rate = form.rate.data or 0
        op.actual_start = form.actual_start.data
        op.actual_end = form.actual_end.data
        op.status = form.status.data
        op.is_final = bool(form.is_final.data)
        op.notes = form.notes.data
        db.session.commit()
        flash('Operation updated.', 'success')
        return redirect(url_for('orders.view_order', id=order_id))

    return render_template('orders/operation_form.html', form=form, op=op)


@orders_bp.route('/operation/<int:op_id>/delete', methods=['POST'])
@login_required
def delete_operation(op_id):
    """Delete a route operation only when nothing has been posted against it."""
    op = OrderOperation.query.get_or_404(op_id)
    order_id = op.order_id
    if not _require_write():
        return redirect(url_for('orders.view_order', id=order_id))
    if op.transactions.count():
        flash('Cannot delete an operation that already has movements. '
              'Cancel it instead.', 'danger')
        return redirect(url_for('orders.view_order', id=order_id))
    db.session.delete(op)
    db.session.commit()
    flash('Operation removed.', 'success')
    return redirect(url_for('orders.view_order', id=order_id))


@orders_bp.route('/operation/<int:op_id>/transaction', methods=['POST'])
@login_required
def add_transaction(op_id):
    """Post a movement (issue / return / accept / reject / rework / scrap).

    Enforces that the operation never returns / accepts / rejects / scraps more
    than it issued, and that the same movement reference is not posted twice.
    """
    op = OrderOperation.query.get_or_404(op_id)
    order_id = op.order_id
    if not _require_write():
        return redirect(url_for('orders.view_order', id=order_id))
    form = OperationTransactionForm()
    if not form.validate_on_submit():
        flash('Check the movement values.', 'danger')
        return redirect(url_for('orders.view_order', id=order_id))

    txn_type = form.transaction_type.data
    qty = form.quantity.data or 0
    reference = (form.reference.data or '').strip()

    # Duplicate guard (mirrors the DB unique constraint, but with a clear message).
    if reference and OperationTransaction.query.filter_by(
            order_operation_id=op.id, transaction_type=txn_type,
            reference=reference).first():
        flash(f'That {txn_type} reference has already been posted.', 'danger')
        return redirect(url_for('orders.view_order', id=order_id))

    if txn_type in ('return', 'accept', 'reject', 'rework', 'scrap'):
        # These movements cannot exceed what was issued and not yet accounted for.
        outstanding = op.issued_quantity - op.returned_quantity \
            - op.rejected_quantity - op.scrapped_quantity
        if qty > outstanding:
            flash(f'Only {max(outstanding, 0)} unit(s) are outstanding for this '
                  f'operation.', 'danger')
            return redirect(url_for('orders.view_order', id=order_id))

    txn = OperationTransaction(
        order_operation_id=op.id,
        transaction_type=txn_type,
        transaction_date=form.transaction_date.data,
        quantity=qty,
        reference=reference or None,
        notes=form.notes.data,
        created_by=current_user.id,
    )
    db.session.add(txn)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        flash('That movement has already been posted.', 'danger')
        return redirect(url_for('orders.view_order', id=order_id))

    _refresh_operation_status(op)
    _refresh_order_status(op.order)
    db.session.commit()
    flash('Movement recorded.', 'success')
    return redirect(url_for('orders.view_order', id=order_id))


def _refresh_operation_status(op):
    """Advance an operation's status from its posted movements."""
    if op.status == 'cancelled':
        return
    if op.issued_quantity > 0 and not op.actual_start:
        op.actual_start = date.today()
        if op.status == 'pending':
            op.status = 'in_progress'
    planned = op.quantity_planned or 0
    if planned and op.accepted_quantity >= planned:
        op.status = 'completed'
        if not op.actual_end:
            op.actual_end = date.today()


def _refresh_order_status(order):
    """Move an order into production / completed based on its operations."""
    if order.status in ('cancelled', 'on_hold'):
        return
    ops = order.operations.all()
    if not ops:
        return
    if any(o.issued_quantity > 0 or o.status != 'pending' for o in ops):
        if order.status in ('draft', 'confirmed'):
            order.status = 'in_production'
        if not order.actual_start:
            order.actual_start = date.today()


# ============================================================
# Raw material tracker & allocation
# ============================================================

@orders_bp.route('/materials')
@login_required
def materials():
    """Raw material tracker: lot balances, allocation and remaining stock."""
    search = request.args.get('search', '').strip()
    material_id = request.args.get('material_id', type=int)
    supplier_id = request.args.get('supplier_id', type=int)
    low_stock = request.args.get('low_stock', '')

    query = MaterialLot.query
    if material_id:
        query = query.filter(MaterialLot.material_id == material_id)
    if supplier_id:
        query = query.filter(MaterialLot.supplier_id == supplier_id)
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            MaterialLot.lot_number.ilike(like),
            MaterialLot.heat_number.ilike(like),
            MaterialLot.batch_number.ilike(like),
            MaterialLot.certificate_ref.ilike(like),
            MaterialLot.grade.ilike(like),
            MaterialLot.material.has(Material.material_name.ilike(like)),
        ))

    pagination, sortview = _paginate(query, 'date', LOT_SORT_FIELDS)
    lots = pagination.items
    if low_stock == '1':
        lots = [l for l in lots if l.quantity_remaining <= 0]

    materials_list = Material.query.order_by(Material.material_name).all()
    suppliers = Supplier.query.order_by(Supplier.company_name).all()
    totals = {
        'received': sum(l.quantity_received for l in pagination.items),
        'allocated': sum(l.quantity_allocated for l in pagination.items),
        'issued': sum(l.quantity_issued for l in pagination.items),
        'available': sum(l.quantity_available for l in pagination.items),
        'remaining': sum(l.quantity_remaining for l in pagination.items),
        'shortage': sum(l.quantity_shortage for l in pagination.items),
    }
    return render_template('orders/materials.html', lots=lots, pagination=pagination,
                           materials=materials_list, suppliers=suppliers,
                           search=search, material_id=material_id,
                           supplier_id=supplier_id, low_stock=low_stock,
                           totals=totals, **sortview)


@orders_bp.route('/materials/export')
@login_required
def export_materials():
    """Export the material lot register to Excel."""
    lots_list = MaterialLot.query.order_by(MaterialLot.id.desc()).all()
    output = export_material_lots_to_excel(lots_list)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'material_lots_{datetime.now().strftime("%Y%m%d")}.xlsx')


def _populate_lot_choices(form, lot=None):
    form.material_id.choices = [('', 'Not linked')] + [
        (str(m.id), f'{m.material_code} - {m.material_name}')
        for m in Material.query.order_by(Material.material_name).all()
    ]
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number)
        for o in Order.query.order_by(Order.id.desc()).limit(300).all()
    ]
    form.supplier_id.choices = [('', 'Not linked')] + [
        (str(s.id), f'{s.supplier_code} - {s.company_name}')
        for s in Supplier.query.order_by(Supplier.company_name).all()
    ]
    form.po_id.choices = [('', 'Not linked')] + [
        (str(p.id), p.po_number)
        for p in SupplierPurchaseOrder.query.order_by(SupplierPurchaseOrder.id.desc()).limit(300).all()
    ]
    form.receipt_id.choices = [('', 'Not linked')] + [
        (str(r.id), r.receipt_number)
        for r in SupplierOrderReceipt.query.order_by(SupplierOrderReceipt.id.desc()).limit(300).all()
    ]


@orders_bp.route('/materials/new', methods=['GET', 'POST'])
@login_required
def new_lot():
    """Record a received material lot (heat/lot/certificate + quantity)."""
    if not _require_write():
        return redirect(url_for('orders.materials'))
    form = MaterialLotForm()
    _populate_lot_choices(form)
    if request.method == 'GET':
        form.lot_number.data = generate_lot_number()
        form.receipt_date.data = date.today()

    if form.validate_on_submit():
        lot = MaterialLot(
            lot_number=form.lot_number.data,
            material_id=int(form.material_id.data) if form.material_id.data else None,
            order_id=int(form.order_id.data) if form.order_id.data else None,
            supplier_id=int(form.supplier_id.data) if form.supplier_id.data else None,
            po_id=int(form.po_id.data) if form.po_id.data else None,
            receipt_id=int(form.receipt_id.data) if form.receipt_id.data else None,
            grade=form.grade.data,
            specification=form.specification.data,
            heat_number=form.heat_number.data,
            batch_number=form.batch_number.data,
            certificate_ref=form.certificate_ref.data,
            quantity_received=form.quantity_received.data or 0,
            unit=form.unit.data,
            receipt_date=form.receipt_date.data,
            created_by=current_user.id,
        )
        db.session.add(lot)
        db.session.flush()
        # The received quantity is the opening movement in the ledger.
        db.session.add(MaterialMovement(
            lot_id=lot.id, movement_type='receipt',
            movement_date=form.receipt_date.data or date.today(),
            quantity=lot.quantity_received, order_id=lot.order_id,
            reference=lot.lot_number, created_by=current_user.id))
        db.session.commit()
        flash(f'Material lot {lot.lot_number} recorded.', 'success')
        return redirect(url_for('orders.materials'))

    return render_template('orders/lot_form.html', form=form, lot=None)


@orders_bp.route('/materials/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_lot(id):
    """Edit a material lot's descriptive fields (not its received quantity)."""
    lot = MaterialLot.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.materials'))
    form = MaterialLotForm(obj=lot)
    _populate_lot_choices(form, lot)
    if request.method == 'GET':
        form.material_id.data = str(lot.material_id) if lot.material_id else ''
        form.order_id.data = str(lot.order_id) if lot.order_id else ''
        form.supplier_id.data = str(lot.supplier_id) if lot.supplier_id else ''
        form.po_id.data = str(lot.po_id) if lot.po_id else ''
        form.receipt_id.data = str(lot.receipt_id) if lot.receipt_id else ''

    if form.validate_on_submit():
        lot.material_id = int(form.material_id.data) if form.material_id.data else None
        lot.order_id = int(form.order_id.data) if form.order_id.data else None
        lot.supplier_id = int(form.supplier_id.data) if form.supplier_id.data else None
        lot.po_id = int(form.po_id.data) if form.po_id.data else None
        lot.receipt_id = int(form.receipt_id.data) if form.receipt_id.data else None
        lot.grade = form.grade.data
        lot.specification = form.specification.data
        lot.heat_number = form.heat_number.data
        lot.batch_number = form.batch_number.data
        lot.certificate_ref = form.certificate_ref.data
        lot.unit = form.unit.data
        lot.receipt_date = form.receipt_date.data
        db.session.commit()
        flash('Material lot updated.', 'success')
        return redirect(url_for('orders.materials'))

    return render_template('orders/lot_form.html', form=form, lot=lot)


@orders_bp.route('/materials/<int:id>/delete', methods=['POST'])
@login_required
def delete_lot(id):
    """Delete a lot only when no movements have been posted against it."""
    lot = MaterialLot.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.materials'))
    if lot.movements.count() > 1:  # the opening receipt row is expected
        flash('Cannot delete a lot that has stock movements. Adjust it instead.',
              'danger')
        return redirect(url_for('orders.materials'))
    db.session.delete(lot)
    db.session.commit()
    flash('Material lot deleted.', 'success')
    return redirect(url_for('orders.materials'))


@orders_bp.route('/materials/<int:id>/movement', methods=['POST'])
@login_required
def add_movement(id):
    """Post a stock movement for a lot (allocate / issue / return / scrap...).

    Prevents over-allocation and over-issue, and rejects duplicate postings.
    """
    lot = MaterialLot.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.materials'))
    form = MaterialMovementForm()
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number)
        for o in Order.query.order_by(Order.id.desc()).limit(300).all()
    ]
    form.order_operation_id.choices = [('', 'Not linked')] + [
        (str(o.id), f'{o.order.order_number} #{o.sequence} {o.operation_name}')
        for o in OrderOperation.query.order_by(OrderOperation.id.desc()).limit(300).all()
    ]
    if not form.validate_on_submit():
        flash('Check the movement values.', 'danger')
        return redirect(request.referrer or url_for('orders.materials'))

    mtype = form.movement_type.data
    qty = form.quantity.data or 0
    reference = (form.reference.data or '').strip()

    if reference and MaterialMovement.query.filter_by(
            lot_id=lot.id, movement_type=mtype, reference=reference).first():
        flash(f'That {mtype} reference has already been posted for this lot.',
              'danger')
        return redirect(request.referrer or url_for('orders.materials'))

    if mtype == 'allocation' and qty > lot.quantity_remaining:
        flash(f'Only {lot.quantity_remaining:g} {lot.unit} are free to allocate.',
              'danger')
        return redirect(request.referrer or url_for('orders.materials'))
    if mtype == 'issue' and qty > lot.quantity_available:
        flash(f'Only {lot.quantity_available:g} {lot.unit} are available to issue.',
              'danger')
        return redirect(request.referrer or url_for('orders.materials'))
    if mtype == 'scrap' and qty > lot.quantity_available:
        flash(f'Cannot scrap more than the {lot.quantity_available:g} {lot.unit} '
              'available.', 'danger')
        return redirect(request.referrer or url_for('orders.materials'))

    movement = MaterialMovement(
        lot_id=lot.id,
        movement_type=mtype,
        movement_date=form.movement_date.data,
        quantity=qty,
        order_id=int(form.order_id.data) if form.order_id.data else None,
        order_operation_id=int(form.order_operation_id.data) if form.order_operation_id.data else None,
        reference=reference or None,
        notes=form.notes.data,
        created_by=current_user.id,
    )
    db.session.add(movement)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        flash('That movement has already been posted.', 'danger')
        return redirect(request.referrer or url_for('orders.materials'))
    db.session.commit()
    flash('Stock movement recorded.', 'success')
    return redirect(request.referrer or url_for('orders.materials'))


# ============================================================
# Vendor process tracking
# ============================================================

@orders_bp.route('/vendor-processes')
@login_required
def vendor_processes():
    """Track each order through its manufacturing operations (incl. vendors)."""
    search = request.args.get('search', '').strip()
    vendor_id = request.args.get('vendor_id', type=int)
    status = request.args.get('status', '')
    only_vendor = request.args.get('only_vendor', '')
    overdue = request.args.get('overdue', '')

    query = OrderOperation.query.join(Order, OrderOperation.order_id == Order.id)
    if vendor_id:
        query = query.filter(OrderOperation.vendor_id == vendor_id)
    if only_vendor == '1':
        query = query.filter(OrderOperation.is_vendor.is_(True))
    if status:
        query = query.filter(OrderOperation.status == status)
    if overdue == '1':
        query = query.filter(OrderOperation.due_date.isnot(None),
                             OrderOperation.due_date < date.today(),
                             OrderOperation.status.in_(OrderOperation.OPEN_STATUSES))
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Order.order_number.ilike(like),
            OrderOperation.operation_name.ilike(like),
            Order.drawing_number.ilike(like),
            OrderOperation.vendor.has(Vendor.vendor_name.ilike(like)),
        ))

    pagination, sortview = _paginate(query, 'sequence', PROCESS_SORT_FIELDS, default_dir='asc')
    operations = pagination.items
    vendors = Vendor.query.order_by(Vendor.vendor_name).all()
    totals = {
        'planned': sum(o.quantity_planned for o in operations),
        'issued': sum(o.issued_quantity for o in operations),
        'returned': sum(o.returned_quantity for o in operations),
        'accepted': sum(o.accepted_quantity for o in operations),
        'rejected': sum(o.rejected_quantity for o in operations),
        'rework': sum(o.rework_quantity for o in operations),
        'pending': sum(o.pending_quantity for o in operations),
    }
    return render_template('orders/vendor_processes.html', operations=operations,
                           pagination=pagination, vendors=vendors, search=search,
                           vendor_id=vendor_id, status=status,
                           only_vendor=only_vendor, overdue=overdue,
                           totals=totals, **sortview)


@orders_bp.route('/vendor-processes/export')
@login_required
def export_vendor_processes():
    """Export vendor/subcontract operations to Excel."""
    operations = OrderOperation.query.filter_by(is_vendor=True).order_by(
        OrderOperation.order_id, OrderOperation.sequence).all()
    output = export_vendor_process_to_excel(operations)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'vendor_processes_{datetime.now().strftime("%Y%m%d")}.xlsx')


# ============================================================
# Quality inspection
# ============================================================

@orders_bp.route('/inspections')
@login_required
def inspections():
    """List incoming, in-process and final inspections with filters."""
    inspection_type = request.args.get('type', '')
    result = request.args.get('result', '')
    order_id = request.args.get('order_id', type=int)
    search = request.args.get('search', '').strip()

    query = Inspection.query
    if inspection_type:
        query = query.filter(Inspection.inspection_type == inspection_type)
    if result:
        query = query.filter(Inspection.result == result)
    if order_id:
        query = query.filter(Inspection.order_id == order_id)
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Inspection.inspection_number.ilike(like),
            Inspection.acceptance_criteria.ilike(like),
            Inspection.order.has(Order.order_number.ilike(like)),
        ))

    sort_fields = {
        'number': Inspection.inspection_number,
        'date': Inspection.inspection_date,
        'type': Inspection.inspection_type,
        'result': Inspection.result,
    }
    pagination, sortview = _paginate(query, 'date', sort_fields)
    orders = Order.query.order_by(Order.id.desc()).limit(300).all()
    summary = {}
    for r in ('pending', 'passed', 'failed', 'partial'):
        summary[r] = Inspection.query.filter_by(result=r).count()
    return render_template('orders/inspections.html', inspections=pagination.items,
                           pagination=pagination, inspection_type=inspection_type,
                           result=result, order_id=order_id, search=search,
                           orders=orders, summary=summary, **sortview)


def _populate_inspection_choices(form, order):
    form.order_operation_id.choices = [('', 'Not linked')] + [
        (str(o.id), f'#{o.sequence} {o.operation_name}')
        for o in order.operations.all()
    ]
    form.lot_id.choices = [('', 'Not linked')] + [
        (str(l.id), f'{l.lot_number} ({l.material.material_name if l.material else ""})')
        for l in order.lots.all()
    ]


@orders_bp.route('/<int:id>/inspection/new', methods=['GET', 'POST'])
@login_required
def new_inspection(id):
    """Record an inspection (incoming / in-process / final) for an order."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    form = InspectionForm()
    _populate_inspection_choices(form, order)
    if request.method == 'GET':
        form.inspection_date.data = date.today()

    if form.validate_on_submit():
        if (form.quantity_accepted.data or 0) + (form.quantity_rejected.data or 0) \
                > (form.quantity_inspected.data or 0):
            flash('Accepted + rejected cannot exceed the quantity inspected.',
                  'danger')
            return render_template('orders/inspection_form.html', form=form, order=order)

        inspection = Inspection(
            inspection_number=generate_inspection_number(),
            order_id=order.id,
            order_operation_id=int(form.order_operation_id.data) if form.order_operation_id.data else None,
            lot_id=int(form.lot_id.data) if form.lot_id.data else None,
            inspection_type=form.inspection_type.data,
            inspection_date=form.inspection_date.data,
            acceptance_criteria=form.acceptance_criteria.data,
            tolerance=form.tolerance.data,
            measurements=form.measurements.data,
            sample_size=form.sample_size.data or 0,
            quantity_inspected=form.quantity_inspected.data or 0,
            quantity_accepted=form.quantity_accepted.data or 0,
            quantity_rejected=form.quantity_rejected.data or 0,
            quantity_rework=form.quantity_rework.data or 0,
            result=form.result.data,
            notes=form.notes.data,
            inspected_by=current_user.id,
        )
        db.session.add(inspection)
        db.session.commit()
        flash(f'Inspection {inspection.inspection_number} recorded.', 'success')
        return redirect(url_for('orders.view_order', id=id))

    return render_template('orders/inspection_form.html', form=form, order=order)


@orders_bp.route('/inspection/<int:insp_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_inspection(insp_id):
    """Edit an inspection, preserving its technical fields."""
    inspection = Inspection.query.get_or_404(insp_id)
    order = inspection.order
    if not _require_write():
        return redirect(url_for('orders.view_order', id=order.id))
    form = InspectionForm(obj=inspection)
    _populate_inspection_choices(form, order)
    if request.method == 'GET':
        form.order_operation_id.data = (str(inspection.order_operation_id)
                                        if inspection.order_operation_id else '')
        form.lot_id.data = str(inspection.lot_id) if inspection.lot_id else ''

    if form.validate_on_submit():
        inspection.order_operation_id = int(form.order_operation_id.data) if form.order_operation_id.data else None
        inspection.lot_id = int(form.lot_id.data) if form.lot_id.data else None
        inspection.inspection_type = form.inspection_type.data
        inspection.inspection_date = form.inspection_date.data
        inspection.acceptance_criteria = form.acceptance_criteria.data
        inspection.tolerance = form.tolerance.data
        inspection.measurements = form.measurements.data
        inspection.sample_size = form.sample_size.data or 0
        inspection.quantity_inspected = form.quantity_inspected.data or 0
        inspection.quantity_accepted = form.quantity_accepted.data or 0
        inspection.quantity_rejected = form.quantity_rejected.data or 0
        inspection.quantity_rework = form.quantity_rework.data or 0
        inspection.result = form.result.data
        inspection.notes = form.notes.data
        db.session.commit()
        flash('Inspection updated.', 'success')
        return redirect(url_for('orders.view_order', id=order.id))

    return render_template('orders/inspection_form.html', form=form, order=order)


# ============================================================
# Final inspection & packaging / completion
# ============================================================

@orders_bp.route('/packaging')
@login_required
def packaging():
    """Final inspection & packaging register with warehouse receipt status."""
    status = request.args.get('status', '')
    order_id = request.args.get('order_id', type=int)
    query = PackagingRecord.query
    if status:
        query = query.filter(PackagingRecord.status == status)
    if order_id:
        query = query.filter(PackagingRecord.order_id == order_id)

    sort_fields = {
        'date': PackagingRecord.packaging_date,
        'packed': PackagingRecord.quantity_packed,
        'accepted': PackagingRecord.quantity_accepted,
        'status': PackagingRecord.status,
    }
    pagination, sortview = _paginate(query, 'date', sort_fields)
    orders = Order.query.order_by(Order.id.desc()).limit(300).all()
    summary = {
        'packed': PackagingRecord.query.filter_by(status='packed').count(),
        'warehoused': PackagingRecord.query.filter_by(status='warehoused').count(),
        'dispatched': PackagingRecord.query.filter_by(status='dispatched').count(),
    }
    return render_template('orders/packaging.html', records=pagination.items,
                           pagination=pagination, orders=orders, status=status,
                           order_id=order_id, summary=summary, **sortview)


@orders_bp.route('/<int:id>/packaging/new', methods=['GET', 'POST'])
@login_required
def new_packaging(id):
    """Record packaging + warehouse receipt and complete the order if full."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))
    form = PackagingRecordForm()
    if request.method == 'GET':
        form.packaging_date.data = date.today()
        form.quantity_packed.data = order.quantity_produced or None

    if form.validate_on_submit():
        record = PackagingRecord(
            order_id=order.id,
            packaging_date=form.packaging_date.data,
            packaging_type=form.packaging_type.data,
            quantity_packed=form.quantity_packed.data or 0,
            quantity_accepted=form.quantity_accepted.data or 0,
            quantity_rejected=form.quantity_rejected.data or 0,
            label_ref=form.label_ref.data,
            warehouse_receipt_number=form.warehouse_receipt_number.data,
            warehouse_receipt_date=form.warehouse_receipt_date.data,
            location=form.location.data,
            status=form.status.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(record)
        db.session.flush()
        _maybe_complete_order(order)
        db.session.commit()
        flash('Packaging record added.', 'success')
        return redirect(url_for('orders.view_order', id=id))

    return render_template('orders/packaging_form.html', form=form, order=order)


@orders_bp.route('/packaging/<int:rec_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_packaging(rec_id):
    """Edit a packaging record and re-evaluate order completion."""
    record = PackagingRecord.query.get_or_404(rec_id)
    order = record.order
    if not _require_write():
        return redirect(url_for('orders.view_order', id=order.id))
    form = PackagingRecordForm(obj=record)
    if form.validate_on_submit():
        record.packaging_date = form.packaging_date.data
        record.packaging_type = form.packaging_type.data
        record.quantity_packed = form.quantity_packed.data or 0
        record.quantity_accepted = form.quantity_accepted.data or 0
        record.quantity_rejected = form.quantity_rejected.data or 0
        record.label_ref = form.label_ref.data
        record.warehouse_receipt_number = form.warehouse_receipt_number.data
        record.warehouse_receipt_date = form.warehouse_receipt_date.data
        record.location = form.location.data
        record.status = form.status.data
        record.notes = form.notes.data
        _maybe_complete_order(order)
        db.session.commit()
        flash('Packaging record updated.', 'success')
        return redirect(url_for('orders.view_order', id=order.id))

    return render_template('orders/packaging_form.html', form=form, order=order)


def _maybe_complete_order(order):
    """Complete an order once the full ordered quantity is packed and warehoused."""
    ordered = order.quantity_ordered or 0
    if not ordered:
        return
    warehoused = sum(p.quantity_accepted for p in order.packaging.all()
                     if p.status in ('warehoused', 'dispatched'))
    if warehoused >= ordered and order.status not in ('cancelled', 'completed'):
        order.status = 'completed'
        order.actual_completion = date.today()


# ============================================================
# Dispatch: shipments of finished goods to customers
# ============================================================
# A dispatch consumes quantity from packing records (the finished-goods
# ledger). Nothing is double-stored: ``packaging_available`` is accepted minus
# already-dispatched, so a line can never ship more than the warehouse holds.
# Partial dispatches are normal; the order's completion rule is untouched, so a
# part-shipment never marks the order complete.

DISPATCH_SORT_FIELDS = {
    'number': Dispatch.dispatch_number,
    'date': Dispatch.dispatch_date,
    'customer': Dispatch.customer_id,
    'status': Dispatch.status,
}


def _packaging_choices(order):
    """Packing records on an order that still have quantity left to ship."""
    choices = []
    for rec in order.packaging.all():
        avail = packaging_available(rec)
        if avail > 0 and rec.status in ('warehoused', 'dispatched'):
            choices.append((rec.id, f'{rec.packaging_date} - '
                                    f'{rec.quantity_accepted} accepted, '
                                    f'{avail} available'))
    return choices


@orders_bp.route('/dispatches')
@login_required
def dispatches():
    """Dispatch register with status summary and filters."""
    status = request.args.get('status', '')
    delivery = request.args.get('delivery', '')
    order_id = request.args.get('order_id', type=int)

    query = Dispatch.query
    if status:
        query = query.filter(Dispatch.status == status)
    if delivery:
        query = query.filter(Dispatch.delivery_status == delivery)
    if order_id:
        query = query.filter(Dispatch.order_id == order_id)

    pagination, sortview = _paginate(query, 'date', DISPATCH_SORT_FIELDS,
                                     default_dir='desc')
    summary = {
        'prepared': Dispatch.query.filter_by(status='prepared').count(),
        'in_transit': Dispatch.query.filter_by(status='in_transit').count(),
        'delivered': Dispatch.query.filter_by(delivery_status='delivered').count(),
        'pending_delivery': Dispatch.query.filter(
            Dispatch.delivery_status != 'delivered',
            Dispatch.status.notin_(['cancelled', 'returned'])).count(),
    }
    orders = Order.query.order_by(Order.id.desc()).limit(300).all()
    return render_template('orders/dispatches.html', dispatches=pagination.items,
                           pagination=pagination, orders=orders, status=status,
                           delivery=delivery, order_id=order_id, summary=summary,
                           **sortview)


@orders_bp.route('/dispatches/export')
@login_required
def export_dispatches():
    """Export the dispatch register to XLSX."""
    query = Dispatch.query
    status = request.args.get('status', '')
    if status:
        query = query.filter(Dispatch.status == status)
    dispatches = query.order_by(Dispatch.dispatch_date.desc()).all()
    output = export_dispatches_to_excel(dispatches)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'dispatches_{date.today()}.xlsx')


@orders_bp.route('/<int:id>/dispatch/new', methods=['GET', 'POST'])
@login_required
def new_dispatch(id):
    """Create a dispatch (full or partial) against an order."""
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_order', id=id))

    form = DispatchForm()
    form.order_id.choices = [(order.id, order.order_number)]
    form.order_id.data = order.id
    if request.method == 'GET':
        form.dispatch_date.data = date.today()
        if order.customer:
            form.destination.data = order.customer.delivery_address

    if form.validate_on_submit():
        dispatch = Dispatch(
            dispatch_number=generate_dispatch_number(),
            order_id=order.id,
            customer_id=order.customer_id,
            dispatch_date=form.dispatch_date.data,
            mode=form.mode.data,
            transporter=form.transporter.data,
            vehicle_number=form.vehicle_number.data,
            shipment_reference=form.shipment_reference.data,
            eway_bill_number=form.eway_bill_number.data,
            invoice_number=form.invoice_number.data,
            invoice_date=form.invoice_date.data,
            packing_list_ref=form.packing_list_ref.data,
            destination=form.destination.data,
            status=form.status.data,
            delivery_status=form.delivery_status.data,
            delivered_date=form.delivered_date.data,
            pod_reference=form.pod_reference.data,
            receiver_name=form.receiver_name.data,
            remarks=form.remarks.data,
            created_by=current_user.id,
        )
        db.session.add(dispatch)
        db.session.commit()
        notify_roles(
            ['admin', 'sales'],
            f'Dispatch {dispatch.dispatch_number} created',
            f'Order {order.order_number}'
            + (f' for {order.customer.company_name}' if order.customer else ''),
            'dispatch_created', 'order', order.id)
        db.session.commit()
        flash('Dispatch created. Add the packed quantities to ship.', 'success')
        return redirect(url_for('orders.view_dispatch', id=dispatch.id))

    return render_template('orders/dispatch_form.html', form=form, order=order,
                           dispatch=None)


@orders_bp.route('/dispatch/<int:id>')
@login_required
def view_dispatch(id):
    """Dispatch detail with its lines and the warehouse stock it draws from."""
    dispatch = Dispatch.query.get_or_404(id)
    order = dispatch.order
    line_form = DispatchLineForm()
    line_form.packaging_record_id.choices = _packaging_choices(order)
    avail = {rec.id: packaging_available(rec) for rec in order.packaging.all()}
    summary = compute_order_dispatch_summary(order)
    documents = Attachment.query.filter_by(
        entity_type='dispatch', entity_id=dispatch.id).all()
    return render_template('orders/dispatch_view.html', dispatch=dispatch,
                           order=order, line_form=line_form, avail=avail,
                           summary=summary, documents=documents)


@orders_bp.route('/dispatch/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_dispatch(id):
    """Edit dispatch header details / delivery status."""
    dispatch = Dispatch.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('orders.view_dispatch', id=id))
    form = DispatchForm(obj=dispatch)
    form.order_id.choices = [(dispatch.order_id, dispatch.order.order_number)]
    if form.validate_on_submit():
        dispatch.dispatch_date = form.dispatch_date.data
        dispatch.mode = form.mode.data
        dispatch.transporter = form.transporter.data
        dispatch.vehicle_number = form.vehicle_number.data
        dispatch.shipment_reference = form.shipment_reference.data
        dispatch.eway_bill_number = form.eway_bill_number.data
        dispatch.invoice_number = form.invoice_number.data
        dispatch.invoice_date = form.invoice_date.data
        dispatch.packing_list_ref = form.packing_list_ref.data
        dispatch.destination = form.destination.data
        dispatch.status = form.status.data
        dispatch.delivery_status = form.delivery_status.data
        dispatch.delivered_date = form.delivered_date.data
        dispatch.pod_reference = form.pod_reference.data
        dispatch.receiver_name = form.receiver_name.data
        dispatch.remarks = form.remarks.data
        db.session.commit()
        flash('Dispatch updated.', 'success')
        return redirect(url_for('orders.view_dispatch', id=id))
    return render_template('orders/dispatch_form.html', form=form,
                           order=dispatch.order, dispatch=dispatch)


@orders_bp.route('/dispatch/<int:id>/line/add', methods=['POST'])
@login_required
def add_dispatch_line(id):
    """Ship a quantity from a packing record; never beyond what is available."""
    dispatch = Dispatch.query.get_or_404(id)
    order = dispatch.order
    if not _require_write():
        return redirect(url_for('orders.view_dispatch', id=id))

    form = DispatchLineForm()
    form.packaging_record_id.choices = _packaging_choices(order)
    if form.validate_on_submit():
        rec = PackagingRecord.query.get(form.packaging_record_id.data)
        if not rec or rec.order_id != order.id:
            flash('That packing record does not belong to this order.', 'danger')
            return redirect(url_for('orders.view_dispatch', id=id))

        available = packaging_available(rec)
        qty = form.quantity.data or 0
        if qty > available:
            flash(f'Only {available} pcs are available on that packing record.',
                  'danger')
            return redirect(url_for('orders.view_dispatch', id=id))

        db.session.add(DispatchLine(
            dispatch_id=dispatch.id, packaging_record_id=rec.id,
            product_id=order.product_id,
            quantity=qty, quantity_rejected=form.quantity_rejected.data or 0,
            label_ref=form.label_ref.data or rec.label_ref,
            notes=form.notes.data))
        db.session.flush()
        # A packing record whose full accepted quantity has now shipped is
        # marked dispatched; any remainder stays available for a later dispatch.
        if packaging_available(rec) <= 0 and rec.status == 'warehoused':
            rec.status = 'dispatched'
        db.session.commit()
        flash('Dispatch line added.', 'success')
    else:
        flash('Could not add the dispatch line. Check the quantity.', 'danger')
    return redirect(url_for('orders.view_dispatch', id=id))


@orders_bp.route('/dispatch/line/<int:line_id>/delete', methods=['POST'])
@login_required
def delete_dispatch_line(line_id):
    """Remove a dispatch line and return the quantity to available stock."""
    line = DispatchLine.query.get_or_404(line_id)
    dispatch = line.dispatch
    if not _require_write():
        return redirect(url_for('orders.view_dispatch', id=dispatch.id))
    rec = line.packaging_record
    db.session.delete(line)
    db.session.flush()
    if rec and rec.status == 'dispatched' and packaging_available(rec) > 0:
        rec.status = 'warehoused'
    db.session.commit()
    flash('Dispatch line removed.', 'success')
    return redirect(url_for('orders.view_dispatch', id=dispatch.id))


@orders_bp.route('/dispatch/<int:id>/delete', methods=['POST'])
@login_required
def delete_dispatch(id):
    """Delete a dispatch and release its quantities back to stock."""
    dispatch = Dispatch.query.get_or_404(id)
    order_id = dispatch.order_id
    if not _require_write():
        return redirect(url_for('orders.view_dispatch', id=id))
    for line in dispatch.lines.all():
        rec = line.packaging_record
        db.session.delete(line)
        db.session.flush()
        if rec and rec.status == 'dispatched' and packaging_available(rec) > 0:
            rec.status = 'warehoused'
    db.session.delete(dispatch)
    db.session.commit()
    flash('Dispatch deleted.', 'success')
    return redirect(url_for('orders.view_order', id=order_id))
