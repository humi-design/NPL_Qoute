"""Costing routes: estimated vs actual cost per order, and missing inputs.

Estimates come from the existing quotation chain (QuotationItem /
ProductCostHistory). Actuals come only from recorded OrderCostEntry rows, so
nothing is invented. A UNIQUE(source_type, source_id, category) constraint on
OrderCostEntry means the same document can never be counted twice.
"""
from datetime import date

from flask import render_template, redirect, url_for, request, flash, send_file
from flask_login import login_required, current_user
from sqlalchemy import or_, case
from sqlalchemy.exc import IntegrityError

from app.costing import costing_bp
from app.models import (
    db, Order, OrderOperation, OrderCostEntry, Vendor, Supplier, Material,
)
from app.forms import OrderCostEntryForm
from app.utils import (
    COST_CATEGORIES, compute_order_cost, convert_currency,
    export_order_costs_to_excel,
)

PER_PAGE_CHOICES = (10, 25, 50, 100)

ORDER_SORT_FIELDS = {
    'number': Order.order_number,
    'customer': Order.customer_id,
    'qty': Order.quantity_ordered,
    'order_date': Order.order_date,
    'status': Order.status,
}


def _nulls_last(column):
    return case((column.is_(None), 1), else_=0)


def _can_write():
    return current_user.has_permission('write')


def _require_write():
    if not _can_write():
        flash('You do not have permission to modify cost records.', 'danger')
        return False
    return True


def _paginate(query, default_sort, sort_fields, default_dir='desc'):
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


def _refresh_operation_actuals(order):
    """Roll recorded cost entries up onto the order's operations.

    Sets OrderOperation.rate to the actual per-piece cost where a cost entry is
    linked to that operation and the quantity is known, so the route shows
    actual vs planned without storing a second figure.
    """
    for op in order.operations.all():
        entries = [e for e in op.cost_entries.all()]
        if not entries:
            continue
        total = sum(e.effective_amount for e in entries)
        if op.quantity_planned:
            op.rate = round(total / op.quantity_planned, 4)


@costing_bp.route('/')
@login_required
def index():
    """Order costing list: estimated vs actual with variance."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    missing_only = request.args.get('missing', '') == '1'

    query = Order.query
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Order.order_number.ilike(like),
            Order.part_description.ilike(like),
            Order.drawing_number.ilike(like)))
    if status:
        query = query.filter(Order.status == status)

    pagination, args = _paginate(query, 'order_date', ORDER_SORT_FIELDS)

    rows = []
    for o in pagination.items:
        cost = compute_order_cost(o)
        if missing_only and not cost['missing']:
            continue
        rows.append({'order': o, 'cost': cost})

    # Plant-level totals over the current page for the summary tiles.
    totals = {
        'estimate': sum(r['cost']['estimate_total'] for r in rows),
        'actual': sum(r['cost']['actual_total'] for r in rows),
        'missing_inputs': sum(len(r['cost']['missing']) for r in rows),
        'without_estimate': sum(1 for r in rows if not r['cost']['has_estimate']),
        'without_actual': sum(1 for r in rows if not r['cost']['has_actual']),
    }
    totals['variance'] = totals['actual'] - totals['estimate']

    return render_template('costing/index.html', rows=rows, pagination=pagination,
                           args=args, search=search, status=status,
                           missing_only=missing_only, totals=totals,
                           categories=COST_CATEGORIES)


@costing_bp.route('/export')
@login_required
def export_costs():
    query = Order.query
    search = request.args.get('search', '').strip()
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            Order.order_number.ilike(like),
            Order.part_description.ilike(like)))
    rows = [{'order': o, 'cost': compute_order_cost(o)} for o in query.all()]
    output = export_order_costs_to_excel(rows)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='order_costs.xlsx')


@costing_bp.route('/order/<int:id>')
@login_required
def order_cost(id):
    """Estimated vs actual cost breakdown for one order."""
    order = Order.query.get_or_404(id)
    cost = compute_order_cost(order)
    entries = order.cost_entries.order_by(OrderCostEntry.entry_date.desc()).all()
    # Actual cost per operation, from the recorded entries.
    op_costs = {}
    for op in order.operations.all():
        op_entries = op.cost_entries.all()
        op_costs[op.id] = {
            'total': sum(e.effective_amount for e in op_entries),
            'count': len(op_entries),
        }
    return render_template('costing/order_cost.html', order=order, cost=cost,
                           entries=entries, op_costs=op_costs,
                           categories=COST_CATEGORIES)


@costing_bp.route('/order/<int:id>/entry/new', methods=['GET', 'POST'])
@login_required
def new_cost_entry(id):
    order = Order.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('costing.order_cost', id=order.id))
    form = OrderCostEntryForm()
    form.order_operation_id.choices = [('', 'Not linked')] + [
        (str(o.id), f'#{o.sequence} {o.operation_name}')
        for o in order.operations.order_by(OrderOperation.sequence).all()]
    form.vendor_id.choices = [('', 'Not linked')] + [
        (str(v.id), v.vendor_name) for v in Vendor.query.order_by(Vendor.vendor_name).all()]
    form.supplier_id.choices = [('', 'Not linked')] + [
        (str(s.id), s.company_name) for s in Supplier.query.order_by(
            Supplier.company_name).all()]

    if form.validate_on_submit():
        currency = (form.currency.data or order.currency or 'INR').strip().upper()
        amount = form.amount.data or 0
        amount_base = convert_currency(amount, currency, order.currency or 'INR')

        entry = OrderCostEntry(
            order_id=order.id,
            category=form.category.data,
            amount=amount,
            currency=currency,
            amount_base=amount_base,
            entry_date=form.entry_date.data,
            description=form.description.data,
            source_type='manual',
            source_id=None,
            order_operation_id=int(form.order_operation_id.data)
            if form.order_operation_id.data else None,
            vendor_id=int(form.vendor_id.data) if form.vendor_id.data else None,
            supplier_id=int(form.supplier_id.data) if form.supplier_id.data else None,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(entry)
        try:
            db.session.flush()
        except IntegrityError:
            db.session.rollback()
            flash('That cost entry already exists.', 'danger')
            return render_template('costing/entry_form.html', form=form, order=order)
        _refresh_operation_actuals(order)
        db.session.commit()
        flash('Cost entry recorded.', 'success')
        return redirect(url_for('costing.order_cost', id=order.id))

    return render_template('costing/entry_form.html', form=form, order=order)


@costing_bp.route('/entry/<int:id>/delete', methods=['POST'])
@login_required
def delete_cost_entry(id):
    entry = OrderCostEntry.query.get_or_404(id)
    order_id = entry.order_id
    if not _require_write():
        return redirect(url_for('costing.order_cost', id=order_id))
    if entry.source_type != 'manual':
        flash('This cost was generated from another document and cannot be '
              'deleted here — reverse the source document instead.', 'danger')
        return redirect(url_for('costing.order_cost', id=order_id))
    db.session.delete(entry)
    db.session.commit()
    flash('Cost entry deleted.', 'success')
    return redirect(url_for('costing.order_cost', id=order_id))


@costing_bp.route('/missing')
@login_required
def missing_inputs():
    """Report orders whose cost picture is incomplete, without inventing values."""
    rows = []
    for o in Order.query.all():
        cost = compute_order_cost(o)
        if cost['missing'] or not cost['has_estimate'] or not cost['has_actual']:
            rows.append({'order': o, 'cost': cost})
    return render_template('costing/missing.html', rows=rows,
                           categories=COST_CATEGORIES)
