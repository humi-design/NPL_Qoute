"""Procurement & payables routes.

Purchase orders, goods receipts (with batch/heat/lot numbers and inspection
results) and supplier/vendor invoices and payments.

Financial integrity rules enforced here:
  * A receipt line's rejected quantity can never exceed the received quantity.
  * A payment allocation can never exceed the payment's unallocated balance.
  * A payment can be allocated to a given invoice only once (DB unique
    constraint), so invoices and payments are never double counted.
"""
from datetime import date, datetime

from flask import (render_template, redirect, url_for, request, flash, send_file)
from flask_login import login_required, current_user
from sqlalchemy import or_, case

from app.procurement import procurement_bp
from app.models import (db, Supplier, Vendor, Material, SupplierPurchaseOrder,
                        SupplierPurchaseOrderLine, SupplierOrderReceipt,
                        SupplierOrderReceiptLine, SupplierInvoice, SupplierPayment,
                        SupplierPaymentAllocation, VendorInvoice, VendorPayment,
                        VendorPaymentAllocation, VendorWorkOrder, Order,
                        SupplierRFQ, SupplierQuote)
from app.forms import (SupplierPurchaseOrderForm, SupplierPurchaseOrderLineForm,
                       SupplierOrderReceiptForm, SupplierReceiptLineForm,
                       SupplierInvoiceForm, SupplierPaymentForm,
                       PaymentAllocationForm, VendorInvoiceForm, VendorPaymentForm,
                       SupplierRFQForm, SupplierQuoteForm)
from app.utils import (notify_roles,
                       generate_po_number, generate_receipt_number,
                       generate_payment_number,
                       generate_supplier_rfq_number, compare_supplier_quotes,
                       export_purchase_orders_to_excel,
                       export_supplier_invoices_to_excel,
                       export_supplier_payments_to_excel,
                       export_quote_comparison_to_excel)

# Whitelisted sort columns keep user-supplied sort params away from the SQL layer.
PO_SORT_FIELDS = {
    'number': SupplierPurchaseOrder.po_number,
    'supplier': Supplier.company_name,
    'ordered': SupplierPurchaseOrder.order_date,
    'expected': SupplierPurchaseOrder.expected_date,
    'status': SupplierPurchaseOrder.status,
}
INVOICE_SORT_FIELDS = {
    'number': SupplierInvoice.invoice_number,
    'supplier': Supplier.company_name,
    'date': SupplierInvoice.invoice_date,
    'due': SupplierInvoice.due_date,
    'total': SupplierInvoice.total_amount,
    'status': SupplierInvoice.status,
}
PAYMENT_SORT_FIELDS = {
    'number': SupplierPayment.payment_number,
    'supplier': Supplier.company_name,
    'date': SupplierPayment.payment_date,
    'amount': SupplierPayment.amount,
    'method': SupplierPayment.method,
}
PER_PAGE_CHOICES = (10, 25, 50, 100)


def _nulls_last(column):
    """Portable NULLS LAST ordering (MySQL has no NULLS LAST)."""
    return case((column.is_(None), 1), else_=0)


def _paginate(query, default_sort, sort_fields, default_dir='desc'):
    """Apply whitelisted sort + pagination from request args (app convention).

    The caller is responsible for any joins the sort columns need (the
    supplier column is joined unconditionally by the list queries below).
    """
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


def _can_write():
    return current_user.has_permission('write')


def _require_write():
    if not _can_write():
        flash('You do not have permission to modify procurement records.', 'danger')
        return False
    return True


def _is_admin():
    return current_user.has_permission('admin')


def _material_choices():
    return [('', 'Select Material')] + [
        (str(m.id), f'{m.material_code} - {m.material_name}')
        for m in Material.query.order_by(Material.material_name).all()
    ]


# ============================================================
# Purchase orders
# ============================================================

@procurement_bp.route('/')
@login_required
def index():
    """Procurement dashboard: recent POs, payables snapshot."""
    from app.utils import compute_supplier_procurement_summary
    recent_pos = SupplierPurchaseOrder.query.order_by(
        SupplierPurchaseOrder.id.desc()).limit(10).all()
    open_pos = [p for p in SupplierPurchaseOrder.query.all() if p.is_open]
    overdue_pos = [p for p in open_pos if p.is_overdue]
    return render_template('procurement/index.html', recent_pos=recent_pos,
                           open_pos=open_pos, overdue_pos=overdue_pos)


@procurement_bp.route('/purchase-orders')
@login_required
def purchase_orders():
    """List purchase orders with search / status filter, sort and pagination."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')

    query = SupplierPurchaseOrder.query.join(Supplier)
    if status:
        query = query.filter(SupplierPurchaseOrder.status == status)
    if search:
        like = f'%{search}%'
        query = query.filter(or_(
            SupplierPurchaseOrder.po_number.ilike(like),
            Supplier.company_name.ilike(like),
        ))
    pagination, view = _paginate(query, 'number', PO_SORT_FIELDS)
    return render_template('procurement/purchase_orders.html', pos=pagination,
                           search=search, status=status, **view)


@procurement_bp.route('/purchase-orders/export')
@login_required
def export_purchase_orders():
    """Export the filtered purchase-order list to Excel."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')
    query = SupplierPurchaseOrder.query
    if status:
        query = query.filter(SupplierPurchaseOrder.status == status)
    if search:
        like = f'%{search}%'
        query = query.join(Supplier).filter(or_(
            SupplierPurchaseOrder.po_number.ilike(like),
            Supplier.company_name.ilike(like),
        ))
    pos = query.order_by(SupplierPurchaseOrder.id.desc()).all()
    output = export_purchase_orders_to_excel(pos)
    return send_file(output, download_name=f'purchase_orders_{date.today()}.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@procurement_bp.route('/purchase-orders/new', methods=['GET', 'POST'])
@login_required
def new_purchase_order():
    """Create a purchase order for a supplier."""
    if not _require_write():
        return redirect(url_for('procurement.purchase_orders'))

    supplier_id = request.args.get('supplier_id', type=int) or request.form.get(
        'supplier_id', type=int)
    suppliers = Supplier.query.filter_by(is_active=True).order_by(
        Supplier.company_name).all()
    if not suppliers:
        flash('Create a supplier before raising a purchase order.', 'warning')
        return redirect(url_for('supplier.index'))

    form = SupplierPurchaseOrderForm()
    if form.validate_on_submit():
        if not supplier_id:
            flash('Select a supplier for this purchase order.', 'danger')
            return render_template('procurement/po_form.html', form=form, po=None,
                                   suppliers=suppliers, supplier_id=None)
        po = SupplierPurchaseOrder(
            po_number=generate_po_number(),
            supplier_id=supplier_id,
            order_date=form.order_date.data,
            expected_date=form.expected_date.data,
            status=form.status.data,
            currency=form.currency.data,
            payment_terms=form.payment_terms.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(po)
        db.session.commit()
        notify_roles(
            ['admin', 'accounts'],
            f'Purchase order {po.po_number} raised',
            f'On {po.supplier.company_name if po.supplier else "-"}'
            + (f', expected {po.expected_date}' if po.expected_date else ''),
            'purchase_order_created', 'purchase_order', po.id)
        db.session.commit()
        flash(f'Purchase order {po.po_number} created.', 'success')
        return redirect(url_for('procurement.view_purchase_order', id=po.id))

    if supplier_id and not form.order_date.data:
        form.order_date.data = date.today()
    return render_template('procurement/po_form.html', form=form, po=None,
                           suppliers=suppliers, supplier_id=supplier_id)


@procurement_bp.route('/purchase-orders/<int:id>')
@login_required
def view_purchase_order(id):
    """View a PO with its lines and receipts."""
    po = SupplierPurchaseOrder.query.get_or_404(id)
    line_form = SupplierPurchaseOrderLineForm()
    line_form.material_id.choices = _material_choices()
    receipt_form = SupplierOrderReceiptForm()
    return render_template('procurement/po_view.html', po=po,
                           line_form=line_form, receipt_form=receipt_form,
                           today=date.today().isoformat())


@procurement_bp.route('/purchase-orders/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_purchase_order(id):
    """Edit PO header fields."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po = SupplierPurchaseOrder.query.get_or_404(id)
    form = SupplierPurchaseOrderForm(obj=po)
    if form.validate_on_submit():
        po.order_date = form.order_date.data
        po.expected_date = form.expected_date.data
        po.status = form.status.data
        po.currency = form.currency.data
        po.payment_terms = form.payment_terms.data
        po.notes = form.notes.data
        db.session.commit()
        flash('Purchase order updated.', 'success')
        return redirect(url_for('procurement.view_purchase_order', id=id))
    return render_template('procurement/po_form.html', form=form, po=po,
                           suppliers=Supplier.query.order_by(Supplier.company_name).all(),
                           supplier_id=po.supplier_id)


@procurement_bp.route('/purchase-orders/<int:id>/line', methods=['POST'])
@login_required
def add_po_line(id):
    """Add a material line to a PO."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po = SupplierPurchaseOrder.query.get_or_404(id)
    form = SupplierPurchaseOrderLineForm()
    form.material_id.choices = _material_choices()
    if form.validate_on_submit():
        line = SupplierPurchaseOrderLine(
            po_id=po.id,
            material_id=int(form.material_id.data) if form.material_id.data else None,
            description=form.description.data,
            grade=form.grade.data,
            quantity_ordered=form.quantity_ordered.data or 0,
            unit=form.unit.data,
            rate=form.rate.data or 0,
            rate_unit=form.rate_unit.data,
            notes=form.notes.data,
        )
        db.session.add(line)
        db.session.commit()
        flash('Line added.', 'success')
    else:
        flash('Could not add line — check the values.', 'danger')
    return redirect(url_for('procurement.view_purchase_order', id=id))


@procurement_bp.route('/purchase-orders/<int:id>/line/<int:line_id>/delete', methods=['POST'])
@login_required
def delete_po_line(id, line_id):
    """Remove a PO line (only when nothing has been received against it)."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    line = SupplierPurchaseOrderLine.query.get_or_404(line_id)
    if line.received_quantity > 0:
        flash('Cannot remove a line that already has receipts.', 'danger')
        return redirect(url_for('procurement.view_purchase_order', id=id))
    db.session.delete(line)
    db.session.commit()
    flash('Line removed.', 'success')
    return redirect(url_for('procurement.view_purchase_order', id=id))


@procurement_bp.route('/purchase-orders/<int:id>/status/<status>', methods=['POST'])
@login_required
def set_po_status(id, status):
    """Change PO status (cancelled / closed / sent / etc.)."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po = SupplierPurchaseOrder.query.get_or_404(id)
    allowed = ('draft', 'sent', 'partially_received', 'received', 'cancelled', 'closed')
    if status not in allowed:
        flash('Invalid status.', 'danger')
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po.status = status
    db.session.commit()
    flash(f'Purchase order marked {status.replace("_", " ")}.', 'success')
    return redirect(url_for('procurement.view_purchase_order', id=id))


@procurement_bp.route('/purchase-orders/<int:id>/delete', methods=['POST'])
@login_required
def delete_purchase_order(id):
    """Delete a PO only when it has no receipts and no invoices."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po = SupplierPurchaseOrder.query.get_or_404(id)
    if po.receipts.count() > 0 or po.invoices:
        flash('Cannot delete a PO with receipts or invoices. Cancel it instead.', 'danger')
        return redirect(url_for('procurement.view_purchase_order', id=id))
    db.session.delete(po)
    db.session.commit()
    flash('Purchase order deleted.', 'success')
    return redirect(url_for('procurement.purchase_orders'))


@procurement_bp.route('/purchase-orders/<int:id>/receipt', methods=['POST'])
@login_required
def add_receipt(id):
    """Record a goods receipt (GRN) against a PO, with batch/heat/lot details."""
    if not _require_write():
        return redirect(url_for('procurement.view_purchase_order', id=id))
    po = SupplierPurchaseOrder.query.get_or_404(id)
    form = SupplierOrderReceiptForm()
    if not form.validate_on_submit():
        flash('Receipt date is required.', 'danger')
        return redirect(url_for('procurement.view_purchase_order', id=id))

    receipt = SupplierOrderReceipt(
        receipt_number=generate_receipt_number(),
        po_id=po.id,
        supplier_id=po.supplier_id,
        receipt_date=form.receipt_date.data,
        supplier_document=form.supplier_document.data,
        notes=form.notes.data,
        received_by=current_user.id,
    )
    db.session.add(receipt)
    db.session.commit()
    flash(f'Goods receipt {receipt.receipt_number} created. Add line quantities below.',
          'success')
    return redirect(url_for('procurement.view_receipt', id=receipt.id))


@procurement_bp.route('/receipts/<int:id>')
@login_required
def view_receipt(id):
    """View a receipt and add/see its lines."""
    receipt = SupplierOrderReceipt.query.get_or_404(id)
    line_form = SupplierReceiptLineForm()
    line_form.po_line_id.choices = [('', 'Select Line')] + [
        (str(l.id), f'{l.material.material_name if l.material else l.description} '
                    f'({l.pending_quantity:g} pending)')
        for l in receipt.purchase_order.lines.all()
    ]
    return render_template('procurement/receipt_view.html', receipt=receipt,
                           line_form=line_form)


@procurement_bp.route('/receipts/<int:id>/line', methods=['POST'])
@login_required
def add_receipt_line(id):
    """Add a receipt line: quantities + batch/heat/lot + inspection result."""
    if not _require_write():
        return redirect(url_for('procurement.view_receipt', id=id))
    receipt = SupplierOrderReceipt.query.get_or_404(id)
    form = SupplierReceiptLineForm()
    form.po_line_id.choices = [('', 'Select Line')] + [
        (str(l.id), f'{l.material.material_name if l.material else l.description}')
        for l in receipt.purchase_order.lines.all()
    ]
    if not form.validate_on_submit():
        flash('Check the receipt line values.', 'danger')
        return redirect(url_for('procurement.view_receipt', id=id))

    received = form.quantity_received.data or 0
    rejected = form.quantity_rejected.data or 0
    if rejected > received:
        flash('Rejected quantity cannot exceed received quantity.', 'danger')
        return redirect(url_for('procurement.view_receipt', id=id))

    po_line = None
    if form.po_line_id.data:
        po_line = SupplierPurchaseOrderLine.query.get(int(form.po_line_id.data))

    line = SupplierOrderReceiptLine(
        receipt_id=receipt.id,
        po_line_id=po_line.id if po_line else None,
        material_id=po_line.material_id if po_line else None,
        quantity_received=received,
        quantity_rejected=rejected,
        batch_number=form.batch_number.data,
        heat_number=form.heat_number.data,
        lot_number=form.lot_number.data,
        certificate_ref=form.certificate_ref.data,
        inspection_status=form.inspection_status.data,
        inspection_notes=form.inspection_notes.data,
    )
    db.session.add(line)
    db.session.flush()
    _refresh_po_status(receipt.purchase_order)
    db.session.commit()
    flash('Receipt line added.', 'success')
    return redirect(url_for('procurement.view_receipt', id=id))


@procurement_bp.route('/receipt-lines/<int:line_id>/delete', methods=['POST'])
@login_required
def delete_receipt_line(line_id):
    """Delete a receipt line and refresh the PO status."""
    line = SupplierOrderReceiptLine.query.get_or_404(line_id)
    receipt_id = line.receipt_id
    if not _require_write():
        return redirect(url_for('procurement.view_receipt', id=receipt_id))
    po = line.receipt.purchase_order
    db.session.delete(line)
    db.session.flush()
    _refresh_po_status(po)
    db.session.commit()
    flash('Receipt line removed.', 'success')
    return redirect(url_for('procurement.view_receipt', id=receipt_id))


def _refresh_po_status(po):
    """Move a PO between sent / partially_received / received based on receipts.

    Only auto-advances statuses that reflect receipt progress; a manually
    cancelled or closed PO is left untouched.
    """
    if po.status in ('cancelled', 'closed', 'draft'):
        return
    received = po.received_quantity
    ordered = po.ordered_quantity
    if received <= 0:
        return
    po.status = 'received' if (ordered and received >= ordered) else 'partially_received'


# ============================================================
# Supplier invoices & payments
# ============================================================

@procurement_bp.route('/supplier-invoices')
@login_required
def supplier_invoices():
    """List supplier invoices with outstanding amounts, sort and pagination."""
    status = request.args.get('status', '')
    query = SupplierInvoice.query.join(Supplier)
    if status:
        query = query.filter(SupplierInvoice.status == status)
    pagination, view = _paginate(query, 'date', INVOICE_SORT_FIELDS)
    return render_template('procurement/supplier_invoices.html',
                           invoices=pagination, status=status, **view)


@procurement_bp.route('/supplier-invoices/export')
@login_required
def export_supplier_invoices():
    """Export the filtered supplier-invoice list to Excel."""
    status = request.args.get('status', '')
    query = SupplierInvoice.query
    if status:
        query = query.filter(SupplierInvoice.status == status)
    invoices = query.order_by(SupplierInvoice.id.desc()).all()
    output = export_supplier_invoices_to_excel(invoices)
    return send_file(output, download_name=f'supplier_invoices_{date.today()}.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@procurement_bp.route('/suppliers/<int:supplier_id>/invoice/new', methods=['GET', 'POST'])
@login_required
def new_supplier_invoice(supplier_id):
    """Record a supplier invoice."""
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    supplier = Supplier.query.get_or_404(supplier_id)
    form = SupplierInvoiceForm()
    form.po_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.po_number} ({p.status})')
        for p in SupplierPurchaseOrder.query.filter_by(supplier_id=supplier_id).all()
    ]
    if form.validate_on_submit():
        invoice = SupplierInvoice(
            invoice_number=form.invoice_number.data,
            supplier_id=supplier_id,
            po_id=int(form.po_id.data) if form.po_id.data else None,
            invoice_date=form.invoice_date.data,
            due_date=form.due_date.data,
            subtotal=form.subtotal.data or 0,
            tax_amount=form.tax_amount.data or 0,
            total_amount=form.total_amount.data or 0,
            currency=form.currency.data,
            status=form.status.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(invoice)
        db.session.commit()
        flash(f'Invoice {invoice.invoice_number} recorded.', 'success')
        return redirect(url_for('supplier.view', id=supplier_id))
    return render_template('procurement/invoice_form.html', form=form,
                           supplier=supplier, invoice=None)


@procurement_bp.route('/supplier-invoices/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_supplier_invoice(id):
    """Edit a supplier invoice (total cannot drop below the amount already paid)."""
    invoice = SupplierInvoice.query.get_or_404(id)
    supplier_id = invoice.supplier_id
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    supplier = Supplier.query.get_or_404(supplier_id)
    form = SupplierInvoiceForm(obj=invoice)
    form.po_id.choices = [('', 'Not linked')] + [
        (str(p.id), f'{p.po_number} ({p.status})')
        for p in SupplierPurchaseOrder.query.filter_by(supplier_id=supplier_id).all()
    ]
    # SelectField uses string choices; normalise the integer FK for preselection.
    form.po_id.data = str(invoice.po_id) if invoice.po_id else ''
    if form.validate_on_submit():
        new_total = form.total_amount.data or 0
        if new_total < invoice.paid_amount:
            flash('Total cannot be less than the amount already paid '
                  f'({invoice.paid_amount:g}).', 'danger')
            return render_template('procurement/invoice_form.html', form=form,
                                   supplier=supplier, invoice=invoice)
        invoice.invoice_number = form.invoice_number.data
        invoice.po_id = int(form.po_id.data) if form.po_id.data else None
        invoice.invoice_date = form.invoice_date.data
        invoice.due_date = form.due_date.data
        invoice.subtotal = form.subtotal.data or 0
        invoice.tax_amount = form.tax_amount.data or 0
        invoice.total_amount = new_total
        invoice.currency = form.currency.data
        invoice.status = form.status.data
        invoice.notes = form.notes.data
        db.session.commit()
        flash(f'Invoice {invoice.invoice_number} updated.', 'success')
        return redirect(url_for('supplier.view', id=supplier_id))
    return render_template('procurement/invoice_form.html', form=form,
                           supplier=supplier, invoice=invoice)


@procurement_bp.route('/supplier-invoices/<int:id>/delete', methods=['POST'])
@login_required
def delete_supplier_invoice(id):
    """Delete an invoice only when nothing has been paid against it."""
    invoice = SupplierInvoice.query.get_or_404(id)
    supplier_id = invoice.supplier_id
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    if invoice.allocations.count() > 0:
        flash('Cannot delete an invoice that has payments allocated to it.', 'danger')
        return redirect(url_for('supplier.view', id=supplier_id))
    db.session.delete(invoice)
    db.session.commit()
    flash('Invoice deleted.', 'success')
    return redirect(url_for('supplier.view', id=supplier_id))


@procurement_bp.route('/supplier-payments')
@login_required
def supplier_payments():
    """List supplier payments with unallocated (advance) balances, sort, pagination."""
    pagination, view = _paginate(SupplierPayment.query.join(Supplier), 'date',
                                 PAYMENT_SORT_FIELDS)
    return render_template('procurement/supplier_payments.html',
                           payments=pagination, **view)


@procurement_bp.route('/supplier-payments/export')
@login_required
def export_supplier_payments():
    """Export the supplier-payment list to Excel."""
    payments = SupplierPayment.query.order_by(SupplierPayment.id.desc()).all()
    output = export_supplier_payments_to_excel(payments)
    return send_file(output, download_name=f'supplier_payments_{date.today()}.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@procurement_bp.route('/suppliers/<int:supplier_id>/payment/new', methods=['GET', 'POST'])
@login_required
def new_supplier_payment(supplier_id):
    """Record a payment to a supplier."""
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    supplier = Supplier.query.get_or_404(supplier_id)
    form = SupplierPaymentForm()
    if form.validate_on_submit():
        payment = SupplierPayment(
            payment_number=generate_payment_number(),
            supplier_id=supplier_id,
            payment_date=form.payment_date.data,
            amount=form.amount.data or 0,
            currency=form.currency.data,
            method=form.method.data or None,
            reference=form.reference.data,
            is_advance=form.is_advance.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(payment)
        db.session.commit()
        flash(f'Payment {payment.payment_number} recorded.', 'success')
        return redirect(url_for('supplier.view', id=supplier_id))
    return render_template('procurement/payment_form.html', form=form,
                           supplier=supplier, payment=None)


@procurement_bp.route('/supplier-payments/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_supplier_payment(id):
    """Edit a supplier payment (amount cannot drop below what is allocated)."""
    payment = SupplierPayment.query.get_or_404(id)
    supplier_id = payment.supplier_id
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    supplier = Supplier.query.get_or_404(supplier_id)
    form = SupplierPaymentForm(obj=payment)
    form.method.data = payment.method or ''
    if form.validate_on_submit():
        new_amount = form.amount.data or 0
        if new_amount < payment.allocated_amount:
            flash('Amount cannot be less than the amount already allocated '
                  f'({payment.allocated_amount:g}).', 'danger')
            return render_template('procurement/payment_form.html', form=form,
                                   supplier=supplier, payment=payment)
        payment.payment_date = form.payment_date.data
        payment.amount = new_amount
        payment.currency = form.currency.data
        payment.method = form.method.data or None
        payment.reference = form.reference.data
        payment.is_advance = form.is_advance.data
        payment.notes = form.notes.data
        db.session.commit()
        flash(f'Payment {payment.payment_number} updated.', 'success')
        return redirect(url_for('supplier.view', id=supplier_id))
    return render_template('procurement/payment_form.html', form=form,
                           supplier=supplier, payment=payment)


@procurement_bp.route('/supplier-payments/<int:id>/allocate', methods=['POST'])
@login_required
def allocate_supplier_payment(id):
    """Apply a payment to one of the supplier's outstanding invoices."""
    payment = SupplierPayment.query.get_or_404(id)
    supplier_id = payment.supplier_id
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    form = PaymentAllocationForm()
    form.invoice_id.choices = [('', 'Select Invoice')] + [
        (str(i.id), f'{i.invoice_number} — outstanding {i.outstanding_amount:g}')
        for i in SupplierInvoice.query.filter_by(supplier_id=supplier_id)
        .filter(SupplierInvoice.status != 'cancelled').all()
    ]
    if not form.validate_on_submit() or not form.invoice_id.data:
        flash('Select an invoice and amount.', 'danger')
        return redirect(url_for('supplier.view', id=supplier_id))

    invoice = SupplierInvoice.query.get(int(form.invoice_id.data))
    amount = form.amount.data or 0

    if SupplierPaymentAllocation.query.filter_by(
            payment_id=payment.id, invoice_id=invoice.id).first():
        flash('This payment is already allocated to that invoice.', 'danger')
        return redirect(url_for('supplier.view', id=supplier_id))
    if amount > payment.unallocated_amount + 1e-6:
        flash(f'Amount exceeds the payment balance of {payment.unallocated_amount:g}.',
              'danger')
        return redirect(url_for('supplier.view', id=supplier_id))

    db.session.add(SupplierPaymentAllocation(
        payment_id=payment.id, invoice_id=invoice.id, amount=amount))
    db.session.commit()
    flash('Payment allocated.', 'success')
    return redirect(url_for('supplier.view', id=supplier_id))


@procurement_bp.route('/supplier-payments/<int:id>/delete', methods=['POST'])
@login_required
def delete_supplier_payment(id):
    """Delete a payment (its allocations cascade)."""
    payment = SupplierPayment.query.get_or_404(id)
    supplier_id = payment.supplier_id
    if not _require_write():
        return redirect(url_for('supplier.view', id=supplier_id))
    db.session.delete(payment)
    db.session.commit()
    flash('Payment deleted.', 'success')
    return redirect(url_for('supplier.view', id=supplier_id))


# ============================================================
# Vendor (job-work) invoices & payments
# ============================================================

@procurement_bp.route('/vendors/<int:vendor_id>/invoice/new', methods=['GET', 'POST'])
@login_required
def new_vendor_invoice(vendor_id):
    """Record a job-work invoice from a vendor."""
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    vendor = Vendor.query.get_or_404(vendor_id)
    form = VendorInvoiceForm()
    form.work_order_id.choices = [('', 'Not linked')] + [
        (str(w.id), f'{w.work_order_number} — {w.service_name or w.title or ""}')
        for w in VendorWorkOrder.query.filter_by(vendor_id=vendor_id).all()
    ]
    if form.validate_on_submit():
        invoice = VendorInvoice(
            invoice_number=form.invoice_number.data,
            vendor_id=vendor_id,
            work_order_id=int(form.work_order_id.data) if form.work_order_id.data else None,
            invoice_date=form.invoice_date.data,
            due_date=form.due_date.data,
            subtotal=form.subtotal.data or 0,
            tax_amount=form.tax_amount.data or 0,
            total_amount=form.total_amount.data or 0,
            currency=form.currency.data,
            status=form.status.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(invoice)
        db.session.commit()
        flash(f'Invoice {invoice.invoice_number} recorded.', 'success')
        return redirect(url_for('vendor.view', id=vendor_id))
    return render_template('procurement/invoice_form.html', form=form,
                           vendor=vendor, invoice=None, is_vendor=True)


@procurement_bp.route('/vendor-invoices/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_vendor_invoice(id):
    """Edit a vendor invoice (total cannot drop below the amount already paid)."""
    invoice = VendorInvoice.query.get_or_404(id)
    vendor_id = invoice.vendor_id
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    vendor = Vendor.query.get_or_404(vendor_id)
    form = VendorInvoiceForm(obj=invoice)
    form.work_order_id.choices = [('', 'Not linked')] + [
        (str(w.id), f'{w.work_order_number} — {w.service_name or w.title or ""}')
        for w in VendorWorkOrder.query.filter_by(vendor_id=vendor_id).all()
    ]
    form.work_order_id.data = (str(invoice.work_order_id)
                               if invoice.work_order_id else '')
    if form.validate_on_submit():
        new_total = form.total_amount.data or 0
        if new_total < invoice.paid_amount:
            flash('Total cannot be less than the amount already paid '
                  f'({invoice.paid_amount:g}).', 'danger')
            return render_template('procurement/invoice_form.html', form=form,
                                   vendor=vendor, invoice=invoice, is_vendor=True)
        invoice.invoice_number = form.invoice_number.data
        invoice.work_order_id = (int(form.work_order_id.data)
                                 if form.work_order_id.data else None)
        invoice.invoice_date = form.invoice_date.data
        invoice.due_date = form.due_date.data
        invoice.subtotal = form.subtotal.data or 0
        invoice.tax_amount = form.tax_amount.data or 0
        invoice.total_amount = new_total
        invoice.currency = form.currency.data
        invoice.status = form.status.data
        invoice.notes = form.notes.data
        db.session.commit()
        flash(f'Invoice {invoice.invoice_number} updated.', 'success')
        return redirect(url_for('vendor.view', id=vendor_id))
    return render_template('procurement/invoice_form.html', form=form,
                           vendor=vendor, invoice=invoice, is_vendor=True)


@procurement_bp.route('/vendor-invoices/<int:id>/delete', methods=['POST'])
@login_required
def delete_vendor_invoice(id):
    """Delete a vendor invoice only when nothing has been paid against it."""
    invoice = VendorInvoice.query.get_or_404(id)
    vendor_id = invoice.vendor_id
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    if invoice.allocations.count() > 0:
        flash('Cannot delete an invoice that has payments allocated to it.', 'danger')
        return redirect(url_for('vendor.view', id=vendor_id))
    db.session.delete(invoice)
    db.session.commit()
    flash('Invoice deleted.', 'success')
    return redirect(url_for('vendor.view', id=vendor_id))


@procurement_bp.route('/vendors/<int:vendor_id>/payment/new', methods=['GET', 'POST'])
@login_required
def new_vendor_payment(vendor_id):
    """Record a payment to a vendor."""
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    vendor = Vendor.query.get_or_404(vendor_id)
    form = VendorPaymentForm()
    if form.validate_on_submit():
        payment = VendorPayment(
            payment_number=generate_payment_number(
                model=VendorPayment, prefix='VPAY'),
            vendor_id=vendor_id,
            payment_date=form.payment_date.data,
            amount=form.amount.data or 0,
            currency=form.currency.data,
            method=form.method.data or None,
            reference=form.reference.data,
            is_advance=form.is_advance.data,
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(payment)
        db.session.commit()
        flash(f'Payment {payment.payment_number} recorded.', 'success')
        return redirect(url_for('vendor.view', id=vendor_id))
    return render_template('procurement/payment_form.html', form=form,
                           vendor=vendor, payment=None, is_vendor=True)


@procurement_bp.route('/vendor-payments/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_vendor_payment(id):
    """Edit a vendor payment (amount cannot drop below what is allocated)."""
    payment = VendorPayment.query.get_or_404(id)
    vendor_id = payment.vendor_id
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    vendor = Vendor.query.get_or_404(vendor_id)
    form = VendorPaymentForm(obj=payment)
    form.method.data = payment.method or ''
    if form.validate_on_submit():
        new_amount = form.amount.data or 0
        if new_amount < payment.allocated_amount:
            flash('Amount cannot be less than the amount already allocated '
                  f'({payment.allocated_amount:g}).', 'danger')
            return render_template('procurement/payment_form.html', form=form,
                                   vendor=vendor, payment=payment, is_vendor=True)
        payment.payment_date = form.payment_date.data
        payment.amount = new_amount
        payment.currency = form.currency.data
        payment.method = form.method.data or None
        payment.reference = form.reference.data
        payment.is_advance = form.is_advance.data
        payment.notes = form.notes.data
        db.session.commit()
        flash(f'Payment {payment.payment_number} updated.', 'success')
        return redirect(url_for('vendor.view', id=vendor_id))
    return render_template('procurement/payment_form.html', form=form,
                           vendor=vendor, payment=payment, is_vendor=True)


@procurement_bp.route('/vendor-payments/<int:id>/allocate', methods=['POST'])
@login_required
def allocate_vendor_payment(id):
    """Apply a vendor payment to one of the vendor's outstanding invoices."""
    payment = VendorPayment.query.get_or_404(id)
    vendor_id = payment.vendor_id
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    form = PaymentAllocationForm()
    form.invoice_id.choices = [('', 'Select Invoice')] + [
        (str(i.id), f'{i.invoice_number} — outstanding {i.outstanding_amount:g}')
        for i in VendorInvoice.query.filter_by(vendor_id=vendor_id)
        .filter(VendorInvoice.status != 'cancelled').all()
    ]
    if not form.validate_on_submit() or not form.invoice_id.data:
        flash('Select an invoice and amount.', 'danger')
        return redirect(url_for('vendor.view', id=vendor_id))

    invoice = VendorInvoice.query.get(int(form.invoice_id.data))
    amount = form.amount.data or 0
    if VendorPaymentAllocation.query.filter_by(
            payment_id=payment.id, invoice_id=invoice.id).first():
        flash('This payment is already allocated to that invoice.', 'danger')
        return redirect(url_for('vendor.view', id=vendor_id))
    if amount > payment.unallocated_amount + 1e-6:
        flash(f'Amount exceeds the payment balance of {payment.unallocated_amount:g}.',
              'danger')
        return redirect(url_for('vendor.view', id=vendor_id))

    db.session.add(VendorPaymentAllocation(
        payment_id=payment.id, invoice_id=invoice.id, amount=amount))
    db.session.commit()
    flash('Payment allocated.', 'success')
    return redirect(url_for('vendor.view', id=vendor_id))


@procurement_bp.route('/vendor-payments/<int:id>/delete', methods=['POST'])
@login_required
def delete_vendor_payment(id):
    """Delete a vendor payment (its allocations cascade)."""
    payment = VendorPayment.query.get_or_404(id)
    vendor_id = payment.vendor_id
    if not _require_write():
        return redirect(url_for('vendor.view', id=vendor_id))
    db.session.delete(payment)
    db.session.commit()
    flash('Payment deleted.', 'success')
    return redirect(url_for('vendor.view', id=vendor_id))


# ============================================================
# Supplier RFQs, quotes and comparison
# ============================================================
# The supplier-side request/quotes that feed a purchase order. Nothing is
# auto-awarded: a comparison is shown with a recommendation, and an authorised
# user must explicitly approve a quote before a PO can be raised from it.

RFQ_SORT_FIELDS = {
    'number': SupplierRFQ.rfq_number,
    'date': SupplierRFQ.rfq_date,
    'due': SupplierRFQ.due_date,
    'status': SupplierRFQ.status,
    'quantity': SupplierRFQ.quantity,
}


def _supplier_choices():
    return [('', 'Select Supplier')] + [
        (str(s.id), s.company_name)
        for s in Supplier.query.order_by(Supplier.company_name).all()]


def _material_select_choices():
    return [('', 'Select Material')] + [
        (str(m.id), f'{m.material_code} - {m.material_name}')
        for m in Material.query.order_by(Material.material_name).all()]


@procurement_bp.route('/supplier-rfqs')
@login_required
def supplier_rfqs():
    """List supplier RFQs with their quote counts and award status."""
    search = request.args.get('search', '').strip()
    status = request.args.get('status', '')

    query = SupplierRFQ.query
    if search:
        like = f'%{search}%'
        query = query.filter(or_(SupplierRFQ.rfq_number.ilike(like),
                                 SupplierRFQ.title.ilike(like)))
    if status:
        query = query.filter(SupplierRFQ.status == status)

    pagination, args = _paginate(query, 'date', RFQ_SORT_FIELDS)
    return render_template('procurement/supplier_rfqs.html', pagination=pagination,
                           args=args, search=search, status=status)


@procurement_bp.route('/supplier-rfqs/new', methods=['GET', 'POST'])
@login_required
def new_supplier_rfq():
    if not _require_write():
        return redirect(url_for('procurement.supplier_rfqs'))
    form = SupplierRFQForm()
    form.material_id.choices = _material_select_choices()
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number) for o in Order.query.order_by(
            Order.id.desc()).limit(300).all()]

    if form.validate_on_submit():
        rfq = SupplierRFQ(
            rfq_number=generate_supplier_rfq_number(),
            title=form.title.data.strip(),
            material_id=int(form.material_id.data) if form.material_id.data else None,
            order_id=int(form.order_id.data) if form.order_id.data else None,
            quantity=form.quantity.data or 0,
            unit=form.unit.data or 'kg',
            grade=form.grade.data,
            rfq_date=form.rfq_date.data,
            due_date=form.due_date.data,
            required_date=form.required_date.data,
            status=form.status.data or 'draft',
            delivery_terms=form.delivery_terms.data,
            payment_terms=form.payment_terms.data,
            currency=(form.currency.data or 'INR').upper(),
            notes=form.notes.data,
            created_by=current_user.id,
        )
        db.session.add(rfq)
        db.session.commit()
        flash(f'Supplier RFQ {rfq.rfq_number} created.', 'success')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))

    return render_template('procurement/supplier_rfq_form.html', form=form, rfq=None)


@procurement_bp.route('/supplier-rfqs/<int:id>')
@login_required
def view_supplier_rfq(id):
    """One RFQ: its quotes, the comparison and the recommendation."""
    rfq = SupplierRFQ.query.get_or_404(id)
    comparison = compare_supplier_quotes(rfq) if rfq.quotes.count() else None
    return render_template('procurement/supplier_rfq_view.html', rfq=rfq,
                           comparison=comparison,
                           suppliers=Supplier.query.order_by(Supplier.company_name).all(),
                           today=date.today())


@procurement_bp.route('/supplier-rfqs/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_supplier_rfq(id):
    rfq = SupplierRFQ.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    form = SupplierRFQForm(obj=rfq)
    form.material_id.choices = _material_select_choices()
    form.order_id.choices = [('', 'Not linked')] + [
        (str(o.id), o.order_number) for o in Order.query.order_by(
            Order.id.desc()).limit(300).all()]

    if form.validate_on_submit():
        rfq.title = form.title.data.strip()
        rfq.material_id = int(form.material_id.data) if form.material_id.data else None
        rfq.order_id = int(form.order_id.data) if form.order_id.data else None
        rfq.quantity = form.quantity.data or 0
        rfq.unit = form.unit.data or 'kg'
        rfq.grade = form.grade.data
        rfq.rfq_date = form.rfq_date.data
        rfq.due_date = form.due_date.data
        rfq.required_date = form.required_date.data
        rfq.status = form.status.data
        rfq.delivery_terms = form.delivery_terms.data
        rfq.payment_terms = form.payment_terms.data
        rfq.currency = (form.currency.data or 'INR').upper()
        rfq.notes = form.notes.data
        db.session.commit()
        flash('Supplier RFQ updated.', 'success')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))

    return render_template('procurement/supplier_rfq_form.html', form=form, rfq=rfq)


@procurement_bp.route('/supplier-rfqs/<int:id>/delete', methods=['POST'])
@login_required
def delete_supplier_rfq(id):
    rfq = SupplierRFQ.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('procurement.supplier_rfqs'))
    if rfq.po_id:
        flash('This RFQ has a purchase order raised from it and cannot be deleted.',
              'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    db.session.delete(rfq)
    db.session.commit()
    flash('Supplier RFQ deleted.', 'success')
    return redirect(url_for('procurement.supplier_rfqs'))


@procurement_bp.route('/supplier-rfqs/<int:id>/quote', methods=['POST'])
@login_required
def add_supplier_quote(id):
    """Record a supplier's quote against an RFQ."""
    rfq = SupplierRFQ.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    form = SupplierQuoteForm()
    form.supplier_id.choices = _supplier_choices()
    if not form.validate_on_submit():
        flash('Check the quote values.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))

    quote = SupplierQuote(
        supplier_rfq_id=rfq.id,
        supplier_id=int(form.supplier_id.data),
        quote_reference=form.quote_reference.data,
        quote_date=form.quote_date.data,
        valid_until=form.valid_until.data,
        quantity=form.quantity.data or rfq.quantity,
        unit=form.unit.data or rfq.unit,
        unit_price=form.unit_price.data or 0,
        currency=(form.currency.data or rfq.currency or 'INR').upper(),
        tax_percent=form.tax_percent.data or 0,
        lead_time_days=form.lead_time_days.data,
        payment_terms=form.payment_terms.data or rfq.payment_terms,
        delivery_terms=form.delivery_terms.data or rfq.delivery_terms,
        quality_rating=form.quality_rating.data,
        status=form.status.data or 'received',
        notes=form.notes.data,
        created_by=current_user.id,
    )
    db.session.add(quote)
    # Receiving a quote moves an RFQ from draft/sent to quoted.
    if rfq.status in ('draft', 'sent'):
        rfq.status = 'quoted'
    db.session.commit()
    flash('Supplier quote recorded.', 'success')
    return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))


@procurement_bp.route('/supplier-quotes/<int:id>/delete', methods=['POST'])
@login_required
def delete_supplier_quote(id):
    quote = SupplierQuote.query.get_or_404(id)
    rfq_id = quote.supplier_rfq_id
    if not _require_write():
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq_id))
    if quote.rfq.awarded_quote_id == quote.id:
        flash('This quote has been awarded and cannot be deleted.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq_id))
    db.session.delete(quote)
    db.session.commit()
    flash('Supplier quote deleted.', 'success')
    return redirect(url_for('procurement.view_supplier_rfq', id=rfq_id))


@procurement_bp.route('/supplier-rfqs/<int:id>/compare')
@login_required
def compare_rfq_quotes(id):
    """Show the quote comparison and recommendation (read-only, no award)."""
    rfq = SupplierRFQ.query.get_or_404(id)
    comparison = compare_supplier_quotes(rfq)
    if comparison['rows'] and rfq.status in ('quoted', 'sent', 'draft'):
        rfq.status = 'compared'
        for s in comparison['rows']:
            s['quote'].comparison_score = s['total_score']
            s['quote'].comparison_rank = s['rank']
        db.session.commit()
    return render_template('procurement/supplier_rfq_compare.html', rfq=rfq,
                           comparison=comparison)


@procurement_bp.route('/supplier-rfqs/<int:id>/compare/export')
@login_required
def export_rfq_comparison(id):
    rfq = SupplierRFQ.query.get_or_404(id)
    comparison = compare_supplier_quotes(rfq)
    output = export_quote_comparison_to_excel(rfq, comparison)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'quote_comparison_{rfq.rfq_number}.xlsx')


@procurement_bp.route('/supplier-rfqs/<int:id>/award/<int:quote_id>', methods=['POST'])
@login_required
def award_supplier_quote(id, quote_id):
    """Explicitly approve a quote. Requires an authorised (admin) user.

    This is the only place an RFQ is awarded. It does NOT create a purchase
    order - the buyer still raises the PO deliberately.
    """
    rfq = SupplierRFQ.query.get_or_404(id)
    quote = SupplierQuote.query.filter_by(id=quote_id, supplier_rfq_id=rfq.id).first_or_404()

    if not _is_admin():
        flash('Only an administrator can approve (award) a supplier quote.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    if rfq.is_awarded:
        flash('This RFQ has already been awarded.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))

    rfq.awarded_quote_id = quote.id
    rfq.awarded_at = datetime.utcnow()
    rfq.awarded_by = current_user.id
    rfq.status = 'awarded'
    quote.status = 'selected'
    for other in rfq.quotes.all():
        if other.id != quote.id and other.status != 'rejected':
            other.status = 'shortlisted'
    db.session.commit()
    flash(f'Quote from {quote.supplier.company_name if quote.supplier else "supplier"} '
          'approved. You can now raise a purchase order for it.', 'success')
    return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))


@procurement_bp.route('/supplier-rfqs/<int:id>/create-po', methods=['POST'])
@login_required
def create_po_from_rfq(id):
    """Raise a purchase order from the awarded quote (buyer-initiated)."""
    rfq = SupplierRFQ.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    if not rfq.is_awarded:
        flash('Approve (award) a quote before raising a purchase order.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))
    if rfq.po_id:
        flash('A purchase order has already been raised from this RFQ.', 'danger')
        return redirect(url_for('procurement.view_supplier_rfq', id=rfq.id))

    quote = rfq.awarded_quote
    po = SupplierPurchaseOrder(
        po_number=generate_po_number(),
        supplier_id=quote.supplier_id,
        order_date=date.today(),
        expected_date=rfq.required_date,
        status='draft',
        currency=quote.currency or rfq.currency,
        payment_terms=quote.payment_terms or rfq.payment_terms,
        notes=f'Raised from {rfq.rfq_number} (awarded quote #{quote.id})',
        created_by=current_user.id,
    )
    db.session.add(po)
    db.session.flush()
    db.session.add(SupplierPurchaseOrderLine(
        po_id=po.id,
        material_id=rfq.material_id,
        description=rfq.title,
        grade=rfq.grade,
        quantity_ordered=quote.quantity or rfq.quantity,
        unit=quote.unit or rfq.unit,
        rate=quote.unit_price,
        rate_unit=f'{quote.currency}/{quote.unit or rfq.unit}',
    ))
    rfq.po_id = po.id
    db.session.commit()
    flash(f'Purchase order {po.po_number} created from {rfq.rfq_number}.',
          'success')
    return redirect(url_for('procurement.view_purchase_order', id=po.id))
