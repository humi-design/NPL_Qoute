"""Inventory routes: stock position, ledger, locations, transfers, adjustments.

This module does NOT introduce a second stock system. It reads the ledgers that
already own each balance (MaterialLot/MaterialMovement for raw material and
scrap, OperationTransaction for WIP and material at subcontractors,
PackagingRecord for finished goods) and adds only what was missing: named
locations, transfers between them, and authorised adjustments/reversals.

Integrity rules enforced here:
  * Stock cannot go negative and a lot cannot be over-allocated unless the
    policy in SystemSetting explicitly allows it AND an admin authorises it.
  * History is never edited: a reversal posts a compensating adjustment that
    references the original movement, so the audit trail stays intact.
  * The same document is never counted twice.
"""
from datetime import date

from flask import render_template, redirect, url_for, request, flash, send_file
from flask_login import login_required, current_user
from sqlalchemy import or_, case
from sqlalchemy.exc import IntegrityError

from app.inventory import inventory_bp
from app.models import (
    db, MaterialLot, MaterialMovement, StockLocation, StockTransfer,
    StockAdjustment, Order, OrderOperation, Vendor, Material, Supplier,
    SupplierOrderReceiptLine, PackagingRecord,
)
from app.forms import StockLocationForm, StockTransferForm, StockAdjustmentForm
from app.utils import (
    INVENTORY_CATEGORIES, lot_position, compute_inventory_position,
    build_stock_ledger, get_stock_policy, generate_transfer_number,
    generate_adjustment_number, export_stock_position_to_excel,
    export_stock_ledger_to_excel,
)

PER_PAGE_CHOICES = (10, 25, 50, 100)

MOVEMENT_SORT_FIELDS = {
    'date': MaterialMovement.movement_date,
    'type': MaterialMovement.movement_type,
    'quantity': MaterialMovement.quantity,
    'lot': MaterialMovement.lot_id,
}
LOCATION_SORT_FIELDS = {
    'code': StockLocation.code,
    'name': StockLocation.name,
    'kind': StockLocation.kind,
}
TRANSFER_SORT_FIELDS = {
    'number': StockTransfer.transfer_number,
    'date': StockTransfer.transfer_date,
    'quantity': StockTransfer.quantity,
    'status': StockTransfer.status,
}
ADJUSTMENT_SORT_FIELDS = {
    'number': StockAdjustment.adjustment_number,
    'date': StockAdjustment.adjustment_date,
    'type': StockAdjustment.adjustment_type,
    'quantity': StockAdjustment.quantity,
}


def _nulls_last(column):
    """Portable NULLS LAST ordering (MySQL has no NULLS LAST)."""
    return case((column.is_(None), 1), else_=0)


def _can_write():
    return current_user.has_permission('write')


def _require_write(message='You do not have permission to modify inventory.'):
    if not _can_write():
        flash(message, 'danger')
        return False
    return True


def _is_admin():
    return current_user.has_permission('admin')


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


def _location_choices(include_blank=True):
    choices = [('', 'Select Location')] if include_blank else []
    choices += [(str(l.id), f'{l.code} — {l.name}')
                for l in StockLocation.query.filter_by(is_active=True).order_by(
                    StockLocation.code).all()]
    return choices


# ============================================================
# Stock position & ledger
# ============================================================

@inventory_bp.route('/')
@login_required
def index():
    """Stock position by category with a per-lot breakdown."""
    totals = compute_inventory_position()
    policy = get_stock_policy()

    rows = []
    for key, label in INVENTORY_CATEGORIES:
        rows.append({'key': key, 'label': label, 'quantity': totals.get(key, 0)})

    search = request.args.get('search', '').strip()
    lot_query = MaterialLot.query
    if search:
        like = f'%{search}%'
        lot_query = lot_query.filter(or_(
            MaterialLot.lot_number.ilike(like),
            MaterialLot.heat_number.ilike(like),
            MaterialLot.batch_number.ilike(like),
            MaterialLot.grade.ilike(like)))
    lots = lot_query.order_by(MaterialLot.id.desc()).limit(200).all()
    lot_rows = [lot_position(l) for l in lots]

    return render_template('inventory/index.html', rows=rows, totals=totals,
                           lot_rows=lot_rows, search=search, policy=policy)


@inventory_bp.route('/export')
@login_required
def export_position():
    totals = compute_inventory_position()
    rows = [{'label': label, 'quantity': totals.get(key, 0)}
            for key, label in INVENTORY_CATEGORIES]
    output = export_stock_position_to_excel(rows)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='stock_position.xlsx')


@inventory_bp.route('/ledger')
@login_required
def ledger():
    """Unified, traceable stock ledger across every ledger table."""
    lot_id = request.args.get('lot_id', type=int)
    order_id = request.args.get('order_id', type=int)
    kind = request.args.get('kind', '').strip()

    lot = MaterialLot.query.get(lot_id) if lot_id else None
    order = Order.query.get(order_id) if order_id else None

    events = build_stock_ledger(lot=lot, order=order)
    if kind:
        events = [e for e in events if e['kind'] == kind]

    kinds = sorted({e['kind'] for e in build_stock_ledger(lot=lot, order=order)})

    return render_template('inventory/ledger.html', events=events, lot=lot,
                           order=order, kind=kind, kinds=kinds,
                           lots=MaterialLot.query.order_by(
                               MaterialLot.id.desc()).limit(300).all(),
                           orders=Order.query.order_by(Order.id.desc()).limit(300).all())


@inventory_bp.route('/ledger/export')
@login_required
def export_ledger():
    lot_id = request.args.get('lot_id', type=int)
    order_id = request.args.get('order_id', type=int)
    kind = request.args.get('kind', '').strip()
    lot = MaterialLot.query.get(lot_id) if lot_id else None
    order = Order.query.get(order_id) if order_id else None
    events = build_stock_ledger(lot=lot, order=order)
    if kind:
        events = [e for e in events if e['kind'] == kind]
    output = export_stock_ledger_to_excel(events)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='stock_ledger.xlsx')


@inventory_bp.route('/lots')
@login_required
def lots():
    """Material lots with their full derived position."""
    search = request.args.get('search', '').strip()
    material_id = request.args.get('material_id', type=int)
    low_stock = request.args.get('low_stock', '')

    query = MaterialLot.query
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            MaterialLot.lot_number.ilike(like),
            MaterialLot.heat_number.ilike(like),
            MaterialLot.batch_number.ilike(like),
            MaterialLot.certificate_ref.ilike(like)))
    if material_id:
        query = query.filter(MaterialLot.material_id == material_id)

    pagination, args = _paginate(query, 'date', {
        'lot': MaterialLot.lot_number,
        'material': MaterialLot.material_id,
        'received': MaterialLot.quantity_received,
        'date': MaterialLot.receipt_date,
    })

    lot_rows = [lot_position(l) for l in pagination.items]
    if low_stock == '1':
        lot_rows = [r for r in lot_rows if r['remaining'] <= 0]

    return render_template('inventory/lots.html', lot_rows=lot_rows,
                           pagination=pagination, args=args, search=search,
                           material_id=material_id, low_stock=low_stock,
                           materials=Material.query.order_by(
                               Material.material_name).all())


@inventory_bp.route('/lots/<int:id>')
@login_required
def lot_detail(id):
    """One lot: position, movements, transfers, adjustments and traceability."""
    lot = MaterialLot.query.get_or_404(id)
    pos = lot_position(lot)
    movements = lot.movements.order_by(MaterialMovement.movement_date.desc()).all()
    transfers = lot.transfers.order_by(StockTransfer.transfer_date.desc()).all()
    adjustments = lot.adjustments.order_by(StockAdjustment.adjustment_date.desc()).all()
    policy = get_stock_policy()
    return render_template('inventory/lot_detail.html', lot=lot, pos=pos,
                           movements=movements, transfers=transfers,
                           adjustments=adjustments, policy=policy)


@inventory_bp.route('/movements/<int:id>/reverse', methods=['POST'])
@login_required
def reverse_movement(id):
    """Reverse a material movement with a compensating, auditable adjustment.

    History is never edited. A 'reversal' adjustment adds back what a consuming
    movement took (allocation / issue / scrap / shortage); a 'decrease' removes
    what an adding movement brought in (return / adjustment). The original
    movement is referenced so the pair is traceable.
    """
    movement = MaterialMovement.query.get_or_404(id)
    lot = movement.lot
    if not _require_write():
        return redirect(url_for('inventory.lot_detail', id=lot.id))

    if movement.movement_type == 'receipt':
        flash('A goods receipt cannot be reversed here — it is tied to the '
              'procurement receipt. Post an adjustment instead.', 'danger')
        return redirect(url_for('inventory.lot_detail', id=lot.id))

    existing = StockAdjustment.query.filter_by(reverses_movement_id=movement.id).first()
    if existing:
        flash(f'That movement was already reversed by {existing.adjustment_number}.',
              'danger')
        return redirect(url_for('inventory.lot_detail', id=lot.id))

    # Consuming movements give stock back; adding movements take it away.
    adds_back = movement.movement_type in ('allocation', 'issue', 'scrap', 'shortage')
    adj_type = 'reversal' if adds_back else 'decrease'

    policy = get_stock_policy()
    if not adds_back and not policy['allow_negative_stock']:
        # Removing stock could take the lot negative; block unless authorised.
        available = lot.quantity_available
        if (movement.quantity or 0) > available and not _is_admin():
            flash('Reversing this movement would make the lot negative. '
                  'An administrator must authorise it.', 'danger')
            return redirect(url_for('inventory.lot_detail', id=lot.id))

    adjustment = StockAdjustment(
        adjustment_number=generate_adjustment_number(),
        adjustment_date=date.today(),
        material_lot_id=lot.id,
        location_id=None,
        order_id=movement.order_id,
        adjustment_type=adj_type,
        quantity=movement.quantity or 0,
        unit=lot.unit,
        reverses_movement_id=movement.id,
        reason=f'Reversal of {movement.movement_type} {movement.reference or ""}'.strip(),
        authorized_by=current_user.id if _is_admin() else None,
        created_by=current_user.id,
    )
    db.session.add(adjustment)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        flash('That movement could not be reversed.', 'danger')
        return redirect(url_for('inventory.lot_detail', id=lot.id))
    db.session.commit()
    flash(f'Movement reversed by adjustment {adjustment.adjustment_number}.', 'success')
    return redirect(url_for('inventory.lot_detail', id=lot.id))


# ============================================================
# Stock locations
# ============================================================

@inventory_bp.route('/locations')
@login_required
def locations():
    query = StockLocation.query
    kind = request.args.get('kind', '')
    if kind:
        query = query.filter(StockLocation.kind == kind)
    pagination, args = _paginate(query, 'code', LOCATION_SORT_FIELDS, default_dir='asc')
    return render_template('inventory/locations.html', pagination=pagination,
                           args=args, kind=kind,
                           kinds=StockLocation.LOCATION_KINDS)


@inventory_bp.route('/locations/new', methods=['GET', 'POST'])
@login_required
def new_location():
    if not _require_write():
        return redirect(url_for('inventory.locations'))
    form = StockLocationForm()
    form.vendor_id.choices = [('', 'Not linked')] + [
        (str(v.id), v.vendor_name) for v in Vendor.query.order_by(Vendor.vendor_name).all()]
    if form.validate_on_submit():
        if StockLocation.query.filter_by(code=form.code.data.strip()).first():
            flash('That location code already exists.', 'danger')
        else:
            loc = StockLocation(
                code=form.code.data.strip(), name=form.name.data.strip(),
                kind=form.kind.data,
                vendor_id=int(form.vendor_id.data) if form.vendor_id.data else None,
                address=form.address.data, is_active=bool(form.is_active.data),
                notes=form.notes.data)
            db.session.add(loc)
            db.session.commit()
            flash('Stock location created.', 'success')
            return redirect(url_for('inventory.locations'))
    return render_template('inventory/location_form.html', form=form, location=None)


@inventory_bp.route('/locations/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_location(id):
    location = StockLocation.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('inventory.locations'))
    form = StockLocationForm(obj=location)
    form.vendor_id.choices = [('', 'Not linked')] + [
        (str(v.id), v.vendor_name) for v in Vendor.query.order_by(Vendor.vendor_name).all()]
    if form.validate_on_submit():
        dup = StockLocation.query.filter_by(code=form.code.data.strip()).first()
        if dup and dup.id != location.id:
            flash('That location code already exists.', 'danger')
        else:
            location.code = form.code.data.strip()
            location.name = form.name.data.strip()
            location.kind = form.kind.data
            location.vendor_id = int(form.vendor_id.data) if form.vendor_id.data else None
            location.address = form.address.data
            location.is_active = bool(form.is_active.data)
            location.notes = form.notes.data
            db.session.commit()
            flash('Stock location updated.', 'success')
            return redirect(url_for('inventory.locations'))
    return render_template('inventory/location_form.html', form=form, location=location)


@inventory_bp.route('/locations/<int:id>/delete', methods=['POST'])
@login_required
def delete_location(id):
    location = StockLocation.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('inventory.locations'))
    if location.transfers_out.count() or location.transfers_in.count() \
            or location.adjustments.count():
        flash('This location has stock history — deactivate it instead of deleting.',
              'danger')
        return redirect(url_for('inventory.locations'))
    db.session.delete(location)
    db.session.commit()
    flash('Stock location deleted.', 'success')
    return redirect(url_for('inventory.locations'))


# ============================================================
# Stock transfers
# ============================================================

@inventory_bp.route('/transfers')
@login_required
def transfers():
    query = StockTransfer.query
    status = request.args.get('status', '')
    location_id = request.args.get('location_id', type=int)
    if status:
        query = query.filter(StockTransfer.status == status)
    if location_id:
        query = query.filter(or_(StockTransfer.from_location_id == location_id,
                                 StockTransfer.to_location_id == location_id))
    pagination, args = _paginate(query, 'date', TRANSFER_SORT_FIELDS)
    return render_template('inventory/transfers.html', pagination=pagination,
                           args=args, status=status, location_id=location_id,
                           locations=StockLocation.query.order_by(
                               StockLocation.code).all())


@inventory_bp.route('/transfers/new', methods=['GET', 'POST'])
@login_required
def new_transfer():
    if not _require_write():
        return redirect(url_for('inventory.transfers'))
    form = StockTransferForm()
    form.from_location_id.choices = _location_choices(include_blank=False)
    form.to_location_id.choices = _location_choices(include_blank=False)
    form.material_lot_id.choices = [('', 'Not linked')] + [
        (str(l.id), f'{l.lot_number} — {l.material.material_name if l.material else ""}')
        for l in MaterialLot.query.order_by(MaterialLot.id.desc()).limit(300).all()]
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number) for o in Order.query.order_by(
            Order.id.desc()).limit(300).all()]

    if form.validate_on_submit():
        lot = MaterialLot.query.get(int(form.material_lot_id.data)) \
            if form.material_lot_id.data else None
        qty = form.quantity.data or 0

        # A transfer out of a lot cannot exceed what is available there, unless
        # the negative-stock policy has been explicitly enabled.
        policy = get_stock_policy()
        if lot is not None and form.status.data == 'received':
            available = lot.quantity_available
            if qty > available and not policy['allow_negative_stock']:
                flash(f'Only {available:g} {lot.unit} are available in lot '
                      f'{lot.lot_number}. Adjust the quantity or post a stock '
                      'adjustment first.', 'danger')
                return render_template('inventory/transfer_form.html', form=form)

        transfer = StockTransfer(
            transfer_number=generate_transfer_number(),
            from_location_id=int(form.from_location_id.data),
            to_location_id=int(form.to_location_id.data),
            transfer_date=form.transfer_date.data,
            quantity=qty,
            unit=form.unit.data or (lot.unit if lot else 'pcs'),
            material_lot_id=lot.id if lot else None,
            order_id=int(form.order_id.data) if form.order_id.data else None,
            status=form.status.data,
            reference=form.reference.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(transfer)
        db.session.commit()
        flash(f'Transfer {transfer.transfer_number} recorded.', 'success')
        return redirect(url_for('inventory.transfers'))

    return render_template('inventory/transfer_form.html', form=form)


@inventory_bp.route('/transfers/<int:id>/status/<status>', methods=['POST'])
@login_required
def set_transfer_status(id, status):
    transfer = StockTransfer.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('inventory.transfers'))
    if status not in ('pending', 'in_transit', 'received', 'cancelled'):
        flash('Unknown transfer status.', 'danger')
        return redirect(url_for('inventory.transfers'))
    transfer.status = status
    db.session.commit()
    flash(f'Transfer {transfer.transfer_number} marked {status.replace("_", " ")}.',
          'success')
    return redirect(request.referrer or url_for('inventory.transfers'))


@inventory_bp.route('/transfers/<int:id>/delete', methods=['POST'])
@login_required
def delete_transfer(id):
    transfer = StockTransfer.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('inventory.transfers'))
    db.session.delete(transfer)
    db.session.commit()
    flash('Transfer deleted.', 'success')
    return redirect(url_for('inventory.transfers'))


# ============================================================
# Stock adjustments
# ============================================================

@inventory_bp.route('/adjustments')
@login_required
def adjustments():
    query = StockAdjustment.query
    adj_type = request.args.get('type', '')
    if adj_type:
        query = query.filter(StockAdjustment.adjustment_type == adj_type)
    pagination, args = _paginate(query, 'date', ADJUSTMENT_SORT_FIELDS)
    return render_template('inventory/adjustments.html', pagination=pagination,
                           args=args, adj_type=adj_type,
                           types=StockAdjustment.ADJUSTMENT_TYPES)


@inventory_bp.route('/adjustments/new', methods=['GET', 'POST'])
@login_required
def new_adjustment():
    if not _require_write():
        return redirect(url_for('inventory.adjustments'))
    form = StockAdjustmentForm()
    form.material_lot_id.choices = [
        (str(l.id), f'{l.lot_number} — {l.material.material_name if l.material else ""}')
        for l in MaterialLot.query.order_by(MaterialLot.id.desc()).limit(300).all()]
    form.location_id.choices = [('', 'Not linked')] + _location_choices(include_blank=False)
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number) for o in Order.query.order_by(
            Order.id.desc()).limit(300).all()]

    if form.validate_on_submit():
        lot = MaterialLot.query.get(int(form.material_lot_id.data)) \
            if form.material_lot_id.data else None
        if lot is None:
            flash('Select a material lot to adjust.', 'danger')
            return render_template('inventory/adjustment_form.html', form=form)

        qty = form.quantity.data or 0
        removes = form.adjustment_type.data in ('decrease', 'write_off', 'correction')
        policy = get_stock_policy()

        # A reducing adjustment that would take the lot negative needs an admin.
        if removes and not policy['allow_negative_stock']:
            available = lot.quantity_available
            if qty > available and not _is_admin():
                flash(f'Only {available:g} {lot.unit} are available in lot '
                      f'{lot.lot_number}. An administrator must authorise an '
                      'adjustment beyond that.', 'danger')
                return render_template('inventory/adjustment_form.html', form=form)

        adjustment = StockAdjustment(
            adjustment_number=generate_adjustment_number(),
            adjustment_date=form.adjustment_date.data,
            material_lot_id=lot.id,
            location_id=int(form.location_id.data) if form.location_id.data else None,
            order_id=int(form.order_id.data) if form.order_id.data else None,
            adjustment_type=form.adjustment_type.data,
            quantity=qty,
            unit=form.unit.data or lot.unit,
            reason=form.reason.data,
            reference=form.reference.data,
            authorized_by=current_user.id if (removes and qty > lot.quantity_available)
            else None,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(adjustment)
        db.session.commit()
        flash(f'Adjustment {adjustment.adjustment_number} recorded.', 'success')
        return redirect(url_for('inventory.adjustments'))

    return render_template('inventory/adjustment_form.html', form=form)


@inventory_bp.route('/adjustments/<int:id>/delete', methods=['POST'])
@login_required
def delete_adjustment(id):
    adjustment = StockAdjustment.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('inventory.adjustments'))
    db.session.delete(adjustment)
    db.session.commit()
    flash('Adjustment deleted.', 'success')
    return redirect(url_for('inventory.adjustments'))
