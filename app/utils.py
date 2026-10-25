import os
import uuid
import pandas as pd
from datetime import datetime, date, timedelta
from io import BytesIO
from werkzeug.utils import secure_filename
from flask import current_app, send_file

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'step', 'stp', 'iges', 'igs', 'dxf', 'dwg', 'zip',
                     # documents
                     'xlsx', 'xls', 'csv', 'docx', 'doc', 'txt'}

# Extensions that may be rendered inline in the browser rather than
# force-downloaded. Everything else is sent as an attachment.
INLINE_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'txt'}
# Files that must never be served inline (defence against stored XSS).
NEVER_INLINE_EXTENSIONS = {'html', 'htm', 'svg', 'xml', 'xhtml', 'js'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_file(file, entity_type, entity_id):
    """Upload a file and return the file path."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        # Create directory if not exists
        upload_dir = os.path.join('uploads', entity_type, str(entity_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        
        return {
            'file_name': unique_filename,
            'original_name': filename,
            'file_path': file_path,
            'file_type': ext,
            'file_size': os.path.getsize(file_path)
        }
    return None


def delete_file(file_path):
    """Delete a file."""
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def get_file_extension(filename):
    """Get file extension."""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


# ============ Referential integrity for master-data deletes ============
#
# Master-data records (customer, product, material, machine, process, vendor)
# are referenced by transactions and child rows. Deleting a parent while
# children still point at it either raises an IntegrityError (non-nullable FK
# such as orders.customer_id) or, worse, silently nulls the link on a nullable
# FK (e.g. supplier_materials.material_id), corrupting history.
#
# ``dependent_counts`` reports every relationship that still holds rows, so a
# delete route can refuse the delete and tell the user to deactivate instead.
# Attachments are excluded: they are owned by the record and are cleaned up by
# the delete itself.

def dependent_counts(record, exclude_tables=()):
    """Return {referencing_table_name: row_count} for rows pointing at ``record``.

    Two sources are merged so nothing is missed and nothing is double-counted:

    1. Declared relationships (they know cascade semantics).
    2. A scan of every foreign key in the metadata that targets this record's
       table. This catches columns such as ``orders.created_by`` that have no
       matching relationship, which the relationship pass cannot see.

    Rows that belong to the record (``cascade='all, delete-orphan'``) are
    removed with it, so they are never treated as blockers. Tables named in
    ``exclude_tables`` are skipped too (e.g. a user's own notifications).

    An empty dict means the record can be deleted safely.
    """
    from sqlalchemy import inspect as _sa_inspect
    from sqlalchemy import func

    exclude = set(exclude_tables)
    counts = {}

    try:
        relations = _sa_inspect(type(record)).relationships
    except Exception:
        relations = []

    # Owned children go with the parent; never block on them.
    for rel in relations:
        if rel.cascade.delete or rel.cascade.delete_orphan:
            try:
                exclude.add(rel.target.name)
            except Exception:
                pass

    for rel in relations:
        try:
            table_name = rel.target.name
        except Exception:
            continue
        if table_name in exclude:
            continue
        try:
            value = getattr(record, rel.key)
        except Exception:
            continue
        if value is None:
            continue
        try:
            if rel.uselist:
                count = value.count() if hasattr(value, 'count') else len(value)
            else:
                count = 1
        except Exception:
            continue
        if count:
            counts[table_name] = counts.get(table_name, 0) + count

    # Foreign keys that have no matching relationship.
    from app.extensions import db
    table = type(record).__table__
    pk_name = list(table.primary_key)[0].name
    pk_value = getattr(record, pk_name, None)
    if pk_value is not None:
        for other in table.metadata.sorted_tables:
            if other is table or other.name in exclude or other.name in counts:
                continue
            for fk in other.foreign_keys:
                if fk.column.table is not table:
                    continue
                try:
                    count = (db.session.query(func.count())
                             .select_from(other)
                             .filter(fk.parent == pk_value)
                             .scalar()) or 0
                except Exception:
                    continue
                if count:
                    counts[other.name] = counts.get(other.name, 0) + count
    return counts


def referential_delete_block(record, action_label='delete', exclude_tables=()):
    """Build a user-facing message when a record still has dependents.

    Returns None when the record is safe to delete, otherwise a sentence naming
    which tables still reference it (with counts) and suggesting deactivation.
    """
    counts = dependent_counts(record, exclude_tables=exclude_tables)
    if not counts:
        return None
    parts = ', '.join(f'{n} in {name}' for name, n in sorted(counts.items()))
    return (f'Cannot {action_label} this record while it is referenced by '
            f'{parts}. Deactivate it instead.')


def safe_redirect_target(target, fallback):
    """Return ``target`` only when it is a safe, local path; else ``fallback``.

    Used for the post-login ``next`` parameter. Without this, an attacker can
    craft ``/auth/login?next=https://evil.example.com`` and turn the login page
    into an open redirect (useful for phishing). Only same-site absolute paths
    are allowed: anything with a scheme, a network location (``//host``), or a
    backslash is rejected.
    """
    if not target:
        return fallback
    target = target.strip()
    if not target:
        return fallback
    # Must be a path on this site: single leading slash, no scheme, no host.
    if not target.startswith('/') or target.startswith('//'):
        return fallback
    # Reject backslashes (browser-normalised to '/') and any scheme smuggled
    # into the first path segment, e.g. '/\\evil.com' or '/:443/'.
    if '\\' in target:
        return fallback
    first_segment = target[1:].split('/', 1)[0]
    if ':' in first_segment:
        return fallback
    return target


def is_image_file(filename):
    """Check if file is an image."""
    ext = get_file_extension(filename)
    return ext in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def is_pdf_file(filename):
    """Check if file is a PDF."""
    return get_file_extension(filename) == 'pdf'


def generate_quotation_number(prefix='NP'):
    """Generate unique quotation number."""
    from app.models import Quotation, db
    year = datetime.now().year
    last_quote = Quotation.query.filter(
        Quotation.quotation_number.like(f'{prefix}-{year}-%')
    ).order_by(Quotation.id.desc()).first()
    
    if last_quote:
        try:
            last_num = int(last_quote.quotation_number.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{year}-{new_num:06d}"


def generate_rfq_number(prefix='RFQ'):
    """Generate unique RFQ number."""
    from app.models import RFQ, db
    year = datetime.now().year
    last_rfq = RFQ.query.filter(
        RFQ.rfq_number.like(f'{prefix}-{year}-%')
    ).order_by(RFQ.id.desc()).first()
    
    if last_rfq:
        try:
            last_num = int(last_rfq.rfq_number.split('-')[-1])
            new_num = last_num + 1
        except:
            new_num = 1
    else:
        new_num = 1
    
    return f"{prefix}-{year}-{new_num:06d}"


def _next_code(prefix, attr, model, width=4):
    """Generate the smallest unique PREFIX-NNNN code not already used..
    Scans only rows whose attr starts with the exact prefix, so mixed
    prefixes (e.g. INACT-0001 vs CUST-*) never collide with generated codes..
    """
    largest = 0
    for row in model.query.filter(getattr(model, attr).like(f'{prefix}-%')).all():
        try:
            largest = max(largest, int(getattr(row, attr).split('-')[-1]))
        except (ValueError, TypeError):
            continue
    return f"{prefix}-{largest + 1:0{width}d}"


def generate_product_code(prefix='PRD'):
    """Generate unique product code."""
    from app.models import Product
    return _next_code(prefix, "internal_code", Product, width=5)



def generate_customer_code(prefix='CUST'):
    """Generate unique customer code."""
    from app.models import Customer
    return _next_code(prefix, "customer_code", Customer, width=4)



def generate_vendor_code(prefix='VND'):
    """Generate unique vendor code."""
    from app.models import Vendor
    return _next_code(prefix, "vendor_code", Vendor, width=4)


def generate_work_order_number(prefix='WO'):
    """Generate unique vendor work order number (WO-YYYY-NNNN)."""
    from app.models import VendorWorkOrder
    year = datetime.now().year
    last = VendorWorkOrder.query.filter(
        VendorWorkOrder.work_order_number.like(f'{prefix}-{year}-%')
    ).order_by(VendorWorkOrder.id.desc()).first()

    new_num = 1
    if last:
        try:
            new_num = int(last.work_order_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def compute_vendor_summary(vendor):
    """Compute work/order, delivery and quality metrics for a vendor.

    Status rules (single source of truth for Current / Past / All views):
      - Current work: status in ('pending', 'in_progress')
      - Past work:    status in ('completed', 'cancelled')
      - All work:     every work order

    Quantity rules (no double counting):
      - ordered   = sum(quantity_ordered) over ALL work orders
      - received  = sum(quantity_received) over ALL work orders
      - rejected  = sum(quantity_rejected) over ALL work orders
      - accepted  = max(received - rejected, 0) per order, summed
      - pending   = sum(pending_quantity) over OPEN work orders only
      - reworked  = sum(quantity_affected) of quality issues of type 'rework'
                    (kept separate from rejected; a rework may also be counted
                     as a rejection, so the two are never added together)

    Delivery performance is computed ONLY from completed work orders that have
    both a due date and a completion date. When no such pair exists the metrics
    are returned as ``None`` (unavailable) rather than 0, so an untracked vendor
    is never mistaken for a perfectly performing one.

    Returns a dict of counts and quantities used on the vendor page and API.
    """
    from app.models import VendorQualityIssue

    orders = vendor.work_orders.all()

    current = [w for w in orders if w.is_open]
    past = [w for w in orders if not w.is_open]

    ordered_qty = sum((w.quantity_ordered or 0) for w in orders)
    received_qty = sum((w.quantity_received or 0) for w in orders)
    rejected_qty = sum((w.quantity_rejected or 0) for w in orders)
    accepted_qty = sum(max((w.quantity_received or 0) - (w.quantity_rejected or 0), 0)
                       for w in orders)
    pending_qty = sum(w.pending_quantity for w in current)
    overdue = [w for w in current if w.is_overdue]

    issues = VendorQualityIssue.query.filter_by(vendor_id=vendor.id).all()
    open_issues = [i for i in issues if i.is_open]
    rework_issues = [i for i in issues if i.issue_type == 'rework']
    reworked_qty = sum((i.quantity_affected or 0) for i in rework_issues)

    # ---- Delivery performance: completed orders with real due + completed dates ----
    measured = [w for w in orders
                if w.is_completed and w.due_date and w.completed_date]
    on_time = [w for w in measured if w.completed_date <= w.due_date]
    late = [w for w in measured if w.completed_date > w.due_date]
    delays = [(w.completed_date - w.due_date).days for w in late]

    if measured:
        on_time_percent = round(len(on_time) / len(measured) * 100, 1)
        avg_delay_days = round(sum(delays) / len(delays), 1) if delays else 0.0
        max_delay_days = max(delays) if delays else 0
    else:
        on_time_percent = None      # unavailable, not zero
        avg_delay_days = None
        max_delay_days = None

    # ---- Job-work value: only from real rate/quantity data ----
    completed_value = sum(w.order_value for w in orders if w.is_completed)
    open_value = sum(w.order_value for w in current)
    rejected_value = sum((w.quantity_rejected or 0) * (w.rate or 0) for w in orders)

    return {
        'assigned_work': len(orders),
        'current_work': len(current),
        'past_work': len(past),
        'pending_quantity': pending_qty,
        'completed_quantity': sum((w.quantity_received or 0) for w in orders if w.is_completed),
        'rejected_quantity': rejected_qty,
        'ordered_quantity': ordered_qty,
        'received_quantity': received_qty,
        'accepted_quantity': accepted_qty,
        'reworked_quantity': reworked_qty,
        'overdue_work': len(overdue),
        'total_order_value': sum(w.order_value for w in orders),
        'completed_value': completed_value,
        'open_value': open_value,
        'rejected_value': rejected_value,
        'quality_issues': len(issues),
        'open_quality_issues': len(open_issues),
        'rework_issues': len(rework_issues),
        'vendor_processes': vendor.processes.count(),
        # delivery performance (None = not enough data)
        'measured_orders': len(measured),
        'on_time_orders': len(on_time),
        'late_orders': len(late),
        'on_time_percent': on_time_percent,
        'avg_delay_days': avg_delay_days,
        'max_delay_days': max_delay_days,
    }



def generate_machine_code(prefix='MCH'):
    """Generate unique machine code."""
    from app.models import Machine
    return _next_code(prefix, "machine_code", Machine, width=3)



def generate_material_code(prefix='MAT'):
    """Generate unique material code."""
    from app.models import Material
    return _next_code(prefix, "material_code", Material, width=4)



def generate_process_code(prefix='PRC'):
    """Generate unique process code."""
    from app.models import Process
    return _next_code(prefix, "process_code", Process, width=4)



def generate_template_code(prefix='TMP'):
    """Generate unique template code."""
    from app.models import ProcessTemplate
    return _next_code(prefix, "template_code", ProcessTemplate, width=4)


def generate_supplier_code(prefix='SUP'):
    """Generate unique supplier code."""
    from app.models import Supplier
    return _next_code(prefix, "supplier_code", Supplier, width=4)


def generate_supplier_type_code(prefix='SUPT'):
    """Generate unique supplier type code."""
    from app.models import SupplierType
    return _next_code(prefix, "type_code", SupplierType, width=3)


def compute_supplier_summary(supplier):
    """Compute summary metrics shown on the supplier grid and detail page.

    Everything here is derived from real records only. Where a value cannot be
    computed from existing data (e.g. no supplied-material lead times) the field
    is returned as ``None`` so the template can show "not available" instead of
    a misleading zero.
    """
    from app.models import Attachment, MaterialRate, Material

    links = supplier.supplied_materials.all()

    lead_times = [l.lead_time for l in links if l.lead_time is not None]
    rates = [l.rate for l in links if l.rate is not None]

    documents = Attachment.query.filter_by(
        entity_type='supplier', entity_id=supplier.id).count()

    # Materials whose master record points at this supplier (materials.supplier_id)
    linked_materials = Material.query.filter_by(supplier_id=supplier.id).count()

    return {
        'supplied_materials': len(links),
        'preferred_materials': sum(1 for l in links if l.is_preferred),
        'material_grades': len({l.grade for l in links if l.grade}),
        'documents': documents,
        'linked_materials': linked_materials,
        # lead time: supplier-level default vs. material-level specifics
        'default_lead_time': supplier.lead_time,
        'avg_lead_time': round(sum(lead_times) / len(lead_times), 1) if lead_times else None,
        'min_lead_time': min(lead_times) if lead_times else None,
        'max_lead_time': max(lead_times) if lead_times else None,
        # quoted rate range across supplied materials
        'min_rate': min(rates) if rates else None,
        'max_rate': max(rates) if rates else None,
    }


def compute_supplier_rate_history(supplier, limit=50):
    """Build a material rate history for a supplier from real records.

    Sources (combined, de-duplicated by id):
      1. MaterialRate rows whose free-text ``supplier`` matches this supplier's
         company name (legacy data captured before the supplier master existed).
      2. MaterialRate rows for materials linked to this supplier via
         ``materials.supplier_id``.

    Returns a list of dicts sorted newest-first. An empty list means no rate
    history is recorded (not that rates are zero).
    """
    from app.models import MaterialRate, Material

    seen = {}
    name = (supplier.company_name or '').strip()

    if name:
        for r in MaterialRate.query.filter(MaterialRate.supplier == name).all():
            seen[r.id] = r

    linked_ids = [m.id for m in Material.query.filter_by(supplier_id=supplier.id).all()]
    if linked_ids:
        for r in MaterialRate.query.filter(MaterialRate.material_id.in_(linked_ids)).all():
            seen[r.id] = r

    rows = sorted(seen.values(), key=lambda r: (r.effective_date or date.min), reverse=True)
    return rows[:limit]


def generate_po_number(prefix='PO'):
    """Generate unique purchase order number (PO-YYYY-NNNN)."""
    from app.models import SupplierPurchaseOrder
    year = datetime.now().year
    last = SupplierPurchaseOrder.query.filter(
        SupplierPurchaseOrder.po_number.like(f'{prefix}-{year}-%')
    ).order_by(SupplierPurchaseOrder.id.desc()).first()
    new_num = 1
    if last:
        try:
            new_num = int(last.po_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def generate_receipt_number(prefix='GRN'):
    """Generate unique goods receipt note number (GRN-YYYY-NNNN)."""
    from app.models import SupplierOrderReceipt
    year = datetime.now().year
    last = SupplierOrderReceipt.query.filter(
        SupplierOrderReceipt.receipt_number.like(f'{prefix}-{year}-%')
    ).order_by(SupplierOrderReceipt.id.desc()).first()
    new_num = 1
    if last:
        try:
            new_num = int(last.receipt_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def generate_payment_number(prefix='PAY', model=None):
    """Generate unique payment number (PAY-YYYY-NNNN)."""
    from app.models import SupplierPayment
    model = model or SupplierPayment
    year = datetime.now().year
    last = model.query.filter(
        model.payment_number.like(f'{prefix}-{year}-%')
    ).order_by(model.id.desc()).first()
    new_num = 1
    if last:
        try:
            new_num = int(last.payment_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


AGEING_BUCKETS = ('not_due', '0-30', '31-60', '61-90', '90+')


def _ageing_from_invoices(invoices):
    """Return an ageing breakdown dict over outstanding invoices.

    Counts each invoice exactly once, so an invoice is never double counted.
    """
    buckets = {b: 0.0 for b in AGEING_BUCKETS}
    for inv in invoices:
        out = inv.outstanding_amount
        if out <= 0:
            continue
        buckets[inv.ageing_bucket] = buckets.get(inv.ageing_bucket, 0.0) + out
    buckets['total'] = sum(buckets[b] for b in AGEING_BUCKETS)
    return buckets


def compute_supplier_procurement_summary(supplier):
    """Procurement + payables metrics for a supplier.

    Quantities come from PO lines (received derived from receipt lines), so
    ordered/received/pending reconcile. Financial figures come from invoices and
    payments; a payment is counted once via its allocation rows. When there are
    no PO / invoice / payment records the corresponding fields are 0 (genuinely
    no activity) while the template shows the section as "not tracked" only when
    the feature is unused overall.
    """
    from app.models import (SupplierPurchaseOrder, SupplierInvoice, SupplierPayment)

    pos = SupplierPurchaseOrder.query.filter_by(supplier_id=supplier.id).all()
    open_pos = [p for p in pos if p.is_open and not p.is_cancelled]
    overdue_pos = [p for p in open_pos if p.is_overdue]

    ordered = sum(p.ordered_quantity for p in pos if not p.is_cancelled)
    received = sum(p.received_quantity for p in pos if not p.is_cancelled)
    rejected = sum(p.rejected_quantity for p in pos if not p.is_cancelled)
    accepted = sum(p.accepted_quantity for p in pos if not p.is_cancelled)
    pending = sum(p.pending_quantity for p in open_pos)

    invoices = SupplierInvoice.query.filter_by(supplier_id=supplier.id).all()
    posted_invoices = [i for i in invoices if i.status != 'cancelled']
    invoiced = sum((i.total_amount or 0) for i in posted_invoices)
    paid_on_invoices = sum(i.paid_amount for i in posted_invoices)
    outstanding = sum(i.outstanding_amount for i in posted_invoices)

    payments = SupplierPayment.query.filter_by(supplier_id=supplier.id).all()
    total_paid = sum((p.amount or 0) for p in payments)
    unallocated = sum(p.unallocated_amount for p in payments)  # advances / credits

    # Delivery performance: receipts with a receipt date vs the PO expected date
    measured, on_time = 0, 0
    for p in pos:
        if not p.expected_date:
            continue
        for r in p.receipts.all():
            if not r.receipt_date:
                continue
            measured += 1
            if r.receipt_date <= p.expected_date:
                on_time += 1
    on_time_percent = round(on_time / measured * 100, 1) if measured else None

    return {
        'po_count': len(pos),
        'open_po_count': len(open_pos),
        'overdue_po_count': len(overdue_pos),
        'ordered_quantity': ordered,
        'received_quantity': received,
        'accepted_quantity': accepted,
        'rejected_quantity': rejected,
        'pending_quantity': pending,
        'po_value': sum(p.total_value for p in pos if not p.is_cancelled),
        'open_po_value': sum(p.total_value for p in open_pos),
        'invoiced_amount': invoiced,
        'paid_on_invoices': paid_on_invoices,
        'outstanding_amount': outstanding,
        'payments_total': total_paid,
        'advances_credits': unallocated,
        'ageing': _ageing_from_invoices(posted_invoices),
        'deliveries_measured': measured,
        'deliveries_on_time': on_time,
        'on_time_percent': on_time_percent,
        'has_procurement_data': bool(pos or invoices or payments),
    }


def compute_vendor_financial_summary(vendor):
    """Job-work payables metrics for a vendor (invoices, payments, ageing)."""
    from app.models import VendorInvoice, VendorPayment

    invoices = VendorInvoice.query.filter_by(vendor_id=vendor.id).all()
    posted = [i for i in invoices if i.status != 'cancelled']
    invoiced = sum((i.total_amount or 0) for i in posted)
    paid_on_invoices = sum(i.paid_amount for i in posted)
    outstanding = sum(i.outstanding_amount for i in posted)

    payments = VendorPayment.query.filter_by(vendor_id=vendor.id).all()
    total_paid = sum((p.amount or 0) for p in payments)
    unallocated = sum(p.unallocated_amount for p in payments)

    return {
        'invoice_count': len(invoices),
        'invoiced_amount': invoiced,
        'paid_on_invoices': paid_on_invoices,
        'outstanding_amount': outstanding,
        'payments_total': total_paid,
        'advances_credits': unallocated,
        'ageing': _ageing_from_invoices(posted),
        'has_financial_data': bool(invoices or payments),
    }



# ============ Material Calculator ============

def calculate_blank_length(finished_length, parting=3, facing=2, machining_allowance=1, 
                           grinding_allowance=0.5, chamfer_allowance=0.5, custom_allowances=None):
    """
    Calculate blank length from finished dimensions.
    
    Args:
        finished_length: Finished part length in mm
        parting: Parting allowance in mm
        facing: Facing allowance (both ends)
        machining_allowance: General machining allowance
        grinding_allowance: Grinding allowance
        chamfer_allowance: Chamfer allowance
        custom_allowances: Dict of custom allowance name -> value
    """
    total_allowance = parting + facing + machining_allowance + grinding_allowance + chamfer_allowance
    
    if custom_allowances:
        total_allowance += sum(custom_allowances.values())
    
    return finished_length + total_allowance


def calculate_material_weight(volume_mm3, density_kg_m3):
    """
    Calculate material weight from volume and density.
    
    Args:
        volume_mm3: Volume in cubic mm
        density_kg_m3: Density in kg/m³
    
    Returns:
        Weight in grams
    """
    volume_m3 = volume_mm3 / 1e9  # Convert mm³ to m³
    weight_kg = volume_m3 * density_kg_m3
    return weight_kg * 1000  # Convert to grams


def calculate_round_bar_volume(diameter_mm, length_mm):
    """Calculate volume of round bar in mm³."""
    import math
    radius = diameter_mm / 2
    return math.pi * radius ** 2 * length_mm


def calculate_hex_bar_volume(af_mm, length_mm):
    """Calculate volume of hexagonal bar in mm³."""
    import math
    # Area of hexagon = (3√3/2) * (AF/2)²
    area = (3 * math.sqrt(3) / 2) * (af_mm / 2) ** 2
    return area * length_mm


def calculate_square_bar_volume(side_mm, length_mm):
    """Calculate volume of square bar in mm³."""
    return side_mm ** 2 * length_mm


def calculate_tube_volume(od_mm, id_mm, length_mm):
    """Calculate volume of tube/pipe in mm³."""
    import math
    outer_area = math.pi * (od_mm / 2) ** 2
    inner_area = math.pi * (id_mm / 2) ** 2 if id_mm > 0 else 0
    return (outer_area - inner_area) * length_mm


def calculate_sheet_volume(width_mm, thickness_mm, length_mm):
    """Calculate volume of sheet/flat in mm³."""
    return width_mm * thickness_mm * length_mm


def calculate_pieces_per_bar(blank_length_mm, bar_length_mm=3000):
    """Calculate how many pieces can be made from one bar."""
    return int(bar_length_mm / blank_length_mm) if blank_length_mm > 0 else 0


def calculate_waste_percentage(blank_length_mm, bar_length_mm=3000):
    """Calculate material waste percentage."""
    pieces = calculate_pieces_per_bar(blank_length_mm, bar_length_mm)
    if pieces == 0:
        return 100
    used_length = pieces * blank_length_mm
    waste = bar_length_mm - used_length
    return (waste / bar_length_mm) * 100


def calculate_utilization(blank_length_mm, bar_length_mm=3000):
    """Calculate material utilization percentage."""
    return 100 - calculate_waste_percentage(blank_length_mm, bar_length_mm)


def calculate_material_cost(weight_grams, rate_per_kg, rate_unit='₹/kg'):
    """
    Calculate material cost based on weight and rate.
    
    Args:
        weight_grams: Weight in grams
        rate_per_kg: Rate per kg
        rate_unit: Unit of rate (₹/kg, ₹/bar, ₹/bundle, ₹/piece)
    """
    weight_kg = weight_grams / 1000
    
    if rate_unit == '₹/kg':
        return weight_kg * rate_per_kg
    elif rate_unit == '₹/piece':
        return rate_per_kg  # Rate is per piece
    else:
        return weight_kg * rate_per_kg  # Default to per kg


# ============ Costing Engine ============

def calculate_operation_cost(process, quantity, machine_rate=None, vendor_rate=None):
    """
    Calculate cost for a single operation.
    
    Args:
        process: Process object with cost parameters
        quantity: Number of pieces
        machine_rate: Machine hourly rate (for internal operations)
        vendor_rate: Vendor rate (for vendor operations)
    """
    cost = 0
    
    if process.cost_type == 'cycle_time':
        if machine_rate:
            cycle_time_seconds = process.cycle_time
            setup_time_hours = process.setup_time / 60
            cycle_time_hours = (cycle_time_seconds * quantity) / 3600
            cost = machine_rate * (setup_time_hours + cycle_time_hours)
        cost = (process.cost_value or 0) * quantity
    
    elif process.cost_type == 'per_piece':
        cost = (process.cost_value or 0) * quantity
    
    elif process.cost_type == 'per_kg':
        # Requires weight - handled separately
        cost = process.cost_value or 0
    
    elif process.cost_type == 'per_batch':
        cost = process.cost_value or 0
    
    elif process.cost_type == 'percentage':
        # Percentage of material cost - handled separately
        cost = process.cost_value or 0
    
    elif process.cost_type == 'manual':
        cost = process.cost_value or 0
    
    elif process.cost_type == 'formula':
        # Evaluate formula - requires variables
        cost = process.cost_value or 0
    
    return cost


def calculate_tooling_cost(tool_cost, tool_life, quantity):
    """Calculate tooling cost per piece."""
    if tool_life > 0 and quantity > 0:
        return tool_cost / min(tool_life, quantity)
    return tool_cost / max(quantity, 1)


def calculate_total_cost(material_cost, machining_cost, vendor_cost, tooling_cost, 
                         overhead_percent, profit_percent, scrap_percent=0):
    """Calculate total costing including overheads and profit."""
    # Adjust for scrap
    adjusted_material = material_cost / (1 - scrap_percent / 100) if scrap_percent > 0 else material_cost
    
    # Direct costs
    direct_cost = adjusted_material + machining_cost + vendor_cost + tooling_cost
    
    # Overheads
    overhead_amount = direct_cost * (overhead_percent / 100)
    
    # Subtotal before profit
    subtotal = direct_cost + overhead_amount
    
    # Profit
    profit_amount = subtotal * (profit_percent / 100)
    
    # Final price
    total = subtotal + profit_amount
    
    # Margin
    margin = (total - direct_cost) / total * 100 if total > 0 else 0
    
    return {
        'material_cost': material_cost,
        'adjusted_material': adjusted_material,
        'machining_cost': machining_cost,
        'vendor_cost': vendor_cost,
        'tooling_cost': tooling_cost,
        'direct_cost': direct_cost,
        'overhead_percent': overhead_percent,
        'overhead_amount': overhead_amount,
        'subtotal': subtotal,
        'profit_percent': profit_percent,
        'profit_amount': profit_amount,
        'total': total,
        'margin_percent': margin
    }


# ============ PDF Generation ============

def generate_quotation_pdf(quotation):
    """Generate PDF for quotation."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    from reportlab.pdfgen import canvas
    import io
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, 
                           topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    elements = []
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER)
    header_style = ParagraphStyle('CustomHeader', parent=styles['Heading2'], fontSize=12)
    
    # Header
    company_name = get_company_settings()['COMPANY_NAME'] or 'NPL Fasteners'
    
    elements.append(Paragraph(f"<b>{company_name}</b>", title_style))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph("QUOTATION", ParagraphStyle('QuoteTitle', parent=styles['Heading1'], 
                                                            fontSize=24, alignment=TA_CENTER)))
    elements.append(Spacer(1, 10*mm))
    
    # Quotation details table
    quote_data = [
        ['Quotation Number:', quotation.quotation_number, 'Date:', str(quotation.quotation_date)],
        ['Customer:', quotation.customer.company_name if quotation.customer else '', '', ''],
        ['Valid Until:', str(quotation.valid_until) if quotation.valid_until else '', '', ''],
    ]
    
    quote_table = Table(quote_data, colWidths=[40*mm, 60*mm, 30*mm, 50*mm])
    quote_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(quote_table)
    elements.append(Spacer(1, 15*mm))
    
    # Items table
    items_header = ['#', 'Description', 'Material', 'Qty', 'Unit Price', 'Total']
    items_data = [items_header]
    
    for i, item in enumerate(quotation.items.all(), 1):
        row = [
            str(i),
            item.part_description or '',
            item.material or '',
            str(item.quantity_1) if item.quantity_1 else '',
            f"₹ {item.unit_price_1:.2f}" if item.unit_price_1 else '',
            f"₹ {(item.quantity_1 or 0) * (item.unit_price_1 or 0):.2f}"
        ]
        items_data.append(row)
    
    # Add totals
    items_data.append(['', '', '', '', 'Subtotal:', f"₹ {quotation.subtotal:.2f}"])
    items_data.append(['', '', '', '', 'Tax ({}%):'.format(int(quotation.tax_percent)), 
                       f"₹ {quotation.tax_amount:.2f}"])
    items_data.append(['', '', '', '', 'TOTAL:', f"₹ {quotation.total_amount:.2f}"])
    
    items_table = Table(items_data, colWidths=[10*mm, 70*mm, 30*mm, 20*mm, 30*mm, 30*mm])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -4), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (-2, -3), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.black),
        ('LINEABOVE', (4, -3), (-1, -3), 1, colors.black),
        ('LINEABOVE', (4, -1), (-1, -1), 1.5, colors.black),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 15*mm))
    
    # Terms and conditions
    if quotation.terms_conditions:
        elements.append(Paragraph("<b>Terms & Conditions:</b>", header_style))
        elements.append(Paragraph(quotation.terms_conditions, styles['Normal']))
        elements.append(Spacer(1, 10*mm))
    
    # Signature area
    sig_data = [
        ['', '', 'For ' + company_name],
        ['', '', ''],
        ['', '', 'Authorised Signatory'],
    ]
    sig_table = Table(sig_data, colWidths=[80*mm, 20*mm, 90*mm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(sig_table)
    
    # Page numbers
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============ Excel Export ============

def export_quotations_to_excel(quotations):
    """Export quotations to Excel."""
    data = []
    for q in quotations:
        for item in q.items.all():
            data.append({
                'Quotation No': q.quotation_number,
                'Date': str(q.quotation_date),
                'Customer': q.customer.company_name if q.customer else '',
                'Part Description': item.part_description,
                'Material': item.material,
                'Quantity': item.quantity_1,
                'Unit Price': item.unit_price_1,
                'Total': (item.quantity_1 or 0) * (item.unit_price_1 or 0),
                'Status': q.status,
                'Currency': q.currency
            })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Quotations', index=False)
    
    output.seek(0)
    return output


def export_products_to_excel(products):
    """Export products to Excel."""
    data = []
    for p in products:
        data.append({
            'Internal Code': p.internal_code,
            'Product Name': p.product_name,
            'Customer Part No': p.customer_part_number,
            'Drawing Number': p.drawing_number,
            'Material': p.material,
            'Weight (g)': p.weight,
            'Standard': p.standard,
            'Thread': p.thread,
            'Revision': p.revision,
            'Status': p.status,
            'Created': str(p.created_at)[:10]
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Products', index=False)
    
    output.seek(0)
    return output


def export_customers_to_excel(customers):
    """Export customers to Excel."""
    data = []
    for c in customers:
        data.append({
            'Customer Code': c.customer_code,
            'Company Name': c.company_name,
            'Contact Person': c.contact_person,
            'Email': c.email,
            'Phone': c.phone,
            'GST/VAT': c.gst_vat,
            'Country': c.country,
            'Currency': c.currency,
            'Payment Terms': c.payment_terms,
            'Status': 'Active' if c.is_active else 'Inactive'
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Customers', index=False)
    
    output.seek(0)
    return output


def export_materials_to_excel(materials):
    """Export materials to Excel."""
    data = []
    for m in materials:
        data.append({
            'Material Code': m.material_code,
            'Material Name': m.material_name,
            'Density (kg/m³)': m.density,
            'Grade': m.grade,
            'HSN Code': m.hsn_code,
            'Default Rate': m.default_rate,
            'Rate Unit': m.rate_unit,
            'Default Supplier': m.default_supplier,
            'Status': 'Active' if m.is_active else 'Inactive'
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Materials', index=False)
    
    output.seek(0)
    return output


def export_vendors_to_excel(vendors):
    """Export vendors (with work summary) to Excel."""
    data = []
    for v in vendors:
        summary = compute_vendor_summary(v)
        data.append({
            'Vendor Code': v.vendor_code,
            'Vendor Name': v.vendor_name,
            'Contact Person': v.contact_person,
            'Email': v.email,
            'Phone': v.phone,
            'Mobile': v.mobile,
            'GST': v.gst,
            'Currency': v.default_currency,
            'Lead Time (days)': v.lead_time,
            'Payment Terms': v.payment_terms,
            'Assigned Work': summary['assigned_work'],
            'Current Work': summary['current_work'],
            'Past Work': summary['past_work'],
            'Pending Qty': summary['pending_quantity'],
            'Completed Qty': summary['completed_quantity'],
            'Overdue Work': summary['overdue_work'],
            'Quality Issues': summary['quality_issues'],
            'Status': 'Active' if v.is_active else 'Inactive'
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Vendors', index=False)

    output.seek(0)
    return output


def export_suppliers_to_excel(suppliers):
    """Export suppliers (with material summary) to Excel."""
    data = []
    for s in suppliers:
        summary = compute_supplier_summary(s)
        data.append({
            'Supplier Code': s.supplier_code,
            'Company Name': s.company_name,
            'Supplier Type': s.supplier_type.type_name if s.supplier_type else '',
            'Contact Person': s.contact_person,
            'Email': s.email,
            'Phone': s.phone,
            'Mobile': s.mobile,
            'Address': s.full_address,
            'GST/VAT': s.gst_vat,
            'PAN': s.pan,
            'Currency': s.default_currency,
            'Payment Terms': s.payment_terms,
            'Lead Time (days)': s.lead_time,
            'Materials Supplied': summary['supplied_materials'],
            'Preferred Materials': summary['preferred_materials'],
            'Status': 'Active' if s.is_active else 'Inactive'
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Suppliers', index=False)

    output.seek(0)
    return output


def export_vendor_work_orders_to_excel(work_orders):
    """Export vendor work orders to Excel."""
    data = []
    for w in work_orders:
        data.append({
            'Work Order': w.work_order_number,
            'Vendor': w.vendor.vendor_name if w.vendor else '',
            'Title': w.title,
            'Service': w.service_name,
            'Part': w.part_description,
            'Drawing No': w.drawing_number,
            'Process': w.process.process_name if w.process else '',
            'Ordered Qty': w.quantity_ordered,
            'Received Qty': w.quantity_received,
            'Pending Qty': w.pending_quantity,
            'Rejected Qty': w.quantity_rejected,
            'Rate': w.rate,
            'Order Value': w.order_value,
            'Currency': w.currency,
            'Order Date': str(w.order_date) if w.order_date else '',
            'Due Date': str(w.due_date) if w.due_date else '',
            'Completed Date': str(w.completed_date) if w.completed_date else '',
            'Status': w.status,
            'Priority': w.priority,
            'Overdue': 'Yes' if w.is_overdue else 'No',
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Vendor Work Orders', index=False)

    output.seek(0)
    return output


def export_purchase_orders_to_excel(purchase_orders):
    """Export purchase orders (with receipt progress) to Excel."""
    data = []
    for po in purchase_orders:
        data.append({
            'PO Number': po.po_number,
            'Supplier': po.supplier.company_name if po.supplier else '',
            'Order Date': str(po.order_date) if po.order_date else '',
            'Expected Date': str(po.expected_date) if po.expected_date else '',
            'Ordered Qty': po.ordered_quantity,
            'Received Qty': po.received_quantity,
            'Accepted Qty': po.accepted_quantity,
            'Rejected Qty': po.rejected_quantity,
            'Pending Qty': po.pending_quantity,
            'Completion %': po.completion_percent,
            'Order Value': po.total_value,
            'Currency': po.currency,
            'Payment Terms': po.payment_terms,
            'Status': po.status,
            'Overdue': 'Yes' if po.is_overdue else 'No',
            'Days Overdue': po.days_overdue,
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Purchase Orders', index=False)

    output.seek(0)
    return output


def export_supplier_invoices_to_excel(invoices):
    """Export supplier invoices (with outstanding and ageing) to Excel."""
    data = []
    for inv in invoices:
        data.append({
            'Invoice Number': inv.invoice_number,
            'Supplier': inv.supplier.company_name if inv.supplier else '',
            'PO Number': inv.purchase_order.po_number if inv.purchase_order else '',
            'Invoice Date': str(inv.invoice_date) if inv.invoice_date else '',
            'Due Date': str(inv.due_date) if inv.due_date else '',
            'Subtotal': inv.subtotal,
            'Tax': inv.tax_amount,
            'Total': inv.total_amount,
            'Paid': inv.paid_amount,
            'Outstanding': inv.outstanding_amount,
            'Ageing Bucket': inv.ageing_bucket,
            'Currency': inv.currency,
            'Status': inv.status,
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Supplier Invoices', index=False)

    output.seek(0)
    return output


def export_supplier_payments_to_excel(payments):
    """Export supplier payments (with allocated / advance balances) to Excel."""
    data = []
    for p in payments:
        data.append({
            'Payment Number': p.payment_number,
            'Supplier': p.supplier.company_name if p.supplier else '',
            'Payment Date': str(p.payment_date) if p.payment_date else '',
            'Amount': p.amount,
            'Allocated': p.allocated_amount,
            'Unallocated (Advance)': p.unallocated_amount,
            'Currency': p.currency,
            'Method': p.method,
            'Reference': p.reference,
            'Is Advance': 'Yes' if p.is_advance else 'No',
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Supplier Payments', index=False)

    output.seek(0)
    return output


# ============ Currency Conversion ============

def convert_currency(amount, from_currency, to_currency, rates=None):
    """Convert amount between currencies."""
    if from_currency == to_currency:
        return amount
    
    if rates is None:
        rates = {
            'USD': 83.0,
            'EUR': 90.0,
            'GBP': 100.0,
            'AED': 22.5,
            'INR': 1.0
        }
    
    # Convert to INR first
    inr_amount = amount
    if from_currency != 'INR':
        rate = rates.get(from_currency, 1)
        inr_amount = amount * rate
    
    # Convert from INR to target
    if to_currency == 'INR':
        return inr_amount
    
    rate = rates.get(to_currency, 1)
    return inr_amount / rate


# ============ Revision History ============

def track_revision(entity_type, entity_id, field_name, old_value, new_value, user_id):
    """Track changes in revision history."""
    from app.models import RevisionHistory, db
    
    history = RevisionHistory(
        entity_type=entity_type,
        entity_id=entity_id,
        field_name=field_name,
        old_value=str(old_value) if old_value else None,
        new_value=str(new_value) if new_value else None,
        changed_by=user_id
    )
    db.session.add(history)
    return history


# ============ Orders & Manufacturing Tracking ============

def generate_order_number(prefix='SO'):
    """Generate a unique order number (SO-YYYY-NNNN)."""
    from app.models import Order
    year = datetime.now().year
    last = Order.query.filter(
        Order.order_number.like(f'{prefix}-{year}-%')
    ).order_by(Order.id.desc()).first()

    new_num = 1
    if last:
        try:
            new_num = int(last.order_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def generate_inspection_number(prefix='INS'):
    """Generate a unique inspection number (INS-YYYY-NNNN)."""
    from app.models import Inspection
    year = datetime.now().year
    last = Inspection.query.filter(
        Inspection.inspection_number.like(f'{prefix}-{year}-%')
    ).order_by(Inspection.id.desc()).first()

    new_num = 1
    if last:
        try:
            new_num = int(last.inspection_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def generate_lot_number(prefix='LOT'):
    """Generate a unique material lot number (LOT-YYYY-NNNN)."""
    from app.models import MaterialLot
    year = datetime.now().year
    last = MaterialLot.query.filter(
        MaterialLot.lot_number.like(f'{prefix}-{year}-%')
    ).order_by(MaterialLot.id.desc()).first()

    new_num = 1
    if last:
        try:
            new_num = int(last.lot_number.split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


ORDER_STATUSES = ('draft', 'confirmed', 'in_production', 'completed',
                  'on_hold', 'cancelled')
ORDER_OPEN_STATUSES = ('draft', 'confirmed', 'in_production', 'on_hold')
ORDER_COMPLETE_STATUSES = ('completed', 'cancelled')


def compute_order_summary(order):
    """Aggregate an order's planned vs actual numbers for summary tiles.

    Returns a dict; quantities are all derived from the operation and
    packaging ledgers so they can never disagree with the detail views.
    """
    operations = order.operations.all()
    return {
        'quantity_ordered': order.quantity_ordered or 0,
        'quantity_produced': order.quantity_produced,
        'quantity_rejected': order.quantity_rejected,
        'quantity_rework': order.quantity_rework,
        'quantity_warehoused': order.quantity_warehoused,
        'quantity_shipped': order.quantity_shipped,
        'quantity_dispatched': order.quantity_shipped,
        'pending_quantity': order.pending_quantity,
        'completion_percent': order.completion_percent,
        'operation_count': len(operations),
        'operations_complete': sum(1 for o in operations if o.is_complete),
        'vendor_operations': sum(1 for o in operations if o.is_vendor),
        'open_inspections': sum(1 for i in order.inspections.all() if i.is_open),
        'is_overdue': order.is_overdue,
        'days_overdue': order.days_overdue,
        'order_value': order.order_value,
    }


def compute_material_requirement(order):
    """Planned material requirement vs what has actually been issued.

    Compares the planned operation quantity with the total issued across the
    order's vendor/in-house operations, and reports any recorded shortage.
    """
    planned = sum((o.quantity_planned or 0) for o in order.operations.all())
    issued = sum((m.quantity or 0) for m in order.material_movements.filter_by(
        movement_type='issue').all())
    shortage = sum((m.quantity or 0) for m in order.material_movements.filter_by(
        movement_type='shortage').all())
    return {
        'planned': planned,
        'issued': issued,
        'outstanding': max(planned - issued, 0),
        'shortage': shortage,
    }


def build_order_timeline(order):
    """Build a chronological, traceable order timeline.

    Merges material receipts, operation issues/returns/acceptances,
    inspections and packaging/warehouse receipts into one ordered list of
    events, so an order can be traced from material procurement through
    production to dispatch.
    """
    events = []

    def add(when, kind, title, detail='', badge='secondary'):
        if when is None:
            return
        events.append({
            'date': when,
            'sort_key': when,
            'kind': kind,
            'title': title,
            'detail': detail,
            'badge': badge,
        })

    add(order.order_date, 'order', f'Order {order.order_number} created',
        order.part_description or '', 'primary')
    add(order.planned_start, 'plan', 'Planned production start', '', 'light')

    for lot in order.lots.all():
        detail = f'{lot.quantity_received:g} {lot.unit or ""}'.strip()
        refs = [r for r in (lot.heat_number, lot.batch_number, lot.certificate_ref) if r]
        if refs:
            detail += ' · ' + ', '.join(refs)
        add(lot.receipt_date, 'material',
            f'Material lot {lot.lot_number} received', detail, 'info')

    for op in order.operations.all():
        label = op.operation_name
        if op.is_vendor and op.vendor:
            label += f' ({op.vendor.vendor_name})'
        for txn in op.transactions.order_by('transaction_date').all():
            add(txn.transaction_date, f'txn_{txn.transaction_type}',
                f'{txn.transaction_type.title()} {txn.quantity} — {label}',
                txn.reference or '', 'secondary')

    for insp in order.inspections.all():
        add(insp.inspection_date, 'inspection',
            f'{insp.inspection_type.replace("_", " ").title()} inspection '
            f'{insp.inspection_number}',
            f'{insp.result} · accepted {insp.quantity_accepted}, '
            f'rejected {insp.quantity_rejected}',
            'success' if insp.is_passed else 'warning')

    for pack in order.packaging.all():
        add(pack.packaging_date, 'packaging',
            f'Packed {pack.quantity_packed}',
            f'{pack.packaging_type or ""} · accepted {pack.quantity_accepted}'.strip(' ·'),
            'info')
        add(pack.warehouse_receipt_date, 'warehouse',
            f'Warehouse receipt {pack.warehouse_receipt_number or ""}'.strip(),
            f'{pack.quantity_accepted} accepted · {pack.location or ""}'.strip(' ·'),
            'success')

    add(order.actual_completion, 'completion',
        'Order completed', '', 'success')

    events.sort(key=lambda e: (e['sort_key'], e['kind']))
    return events


def export_orders_to_excel(orders):
    """Export orders with planned vs actual progress to Excel."""
    data = []
    for o in orders:
        summary = compute_order_summary(o)
        data.append({
            'Order Number': o.order_number,
            'Customer': o.customer.company_name if o.customer else '',
            'Part': o.part_description,
            'Drawing No': o.drawing_number,
            'Revision': o.revision,
            'Material Grade': o.material_grade,
            'Qty Ordered': summary['quantity_ordered'],
            'Qty Produced': summary['quantity_produced'],
            'Qty Rejected': summary['quantity_rejected'],
            'Qty Rework': summary['quantity_rework'],
            'Qty Dispatched': summary['quantity_dispatched'],
            'Pending Qty': summary['pending_quantity'],
            'Completion %': summary['completion_percent'],
            'Order Value': summary['order_value'],
            'Currency': o.currency,
            'Order Date': str(o.order_date) if o.order_date else '',
            'Due Date': str(o.due_date) if o.due_date else '',
            'Actual Start': str(o.actual_start) if o.actual_start else '',
            'Actual Completion': str(o.actual_completion) if o.actual_completion else '',
            'Status': o.status,
            'Priority': o.priority,
            'Overdue': 'Yes' if o.is_overdue else 'No',
            'Days Overdue': o.days_overdue,
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Orders', index=False)

    output.seek(0)
    return output


def export_material_lots_to_excel(lots):
    """Export material lots with their derived balances to Excel."""
    data = []
    for lot in lots:
        data.append({
            'Lot Number': lot.lot_number,
            'Material': lot.material.material_name if lot.material else '',
            'Grade': lot.grade,
            'Specification': lot.specification,
            'Supplier': lot.supplier.company_name if lot.supplier else '',
            'Heat No': lot.heat_number,
            'Batch No': lot.batch_number,
            'Certificate Ref': lot.certificate_ref,
            'Qty Received': lot.quantity_received,
            'Qty Allocated': lot.quantity_allocated,
            'Qty Issued': lot.quantity_issued,
            'Qty Returned': lot.quantity_returned,
            'Qty Scrapped': lot.quantity_scrapped,
            'Qty Shortage': lot.quantity_shortage,
            'Qty Available': lot.quantity_available,
            'Qty Remaining': lot.quantity_remaining,
            'Unit': lot.unit,
            'Receipt Date': str(lot.receipt_date) if lot.receipt_date else '',
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Material Lots', index=False)

    output.seek(0)
    return output


def export_vendor_process_to_excel(operations):
    """Export vendor/subcontract operations with planned vs actual progress."""
    data = []
    for op in operations:
        data.append({
            'Order': op.order.order_number if op.order else '',
            'Sequence': op.sequence,
            'Operation': op.operation_name,
            'Vendor': op.vendor.vendor_name if op.vendor else '',
            'Process': op.process.process_name if op.process else '',
            'Qty Planned': op.quantity_planned,
            'Qty Issued': op.issued_quantity,
            'Qty Processed': op.processed_quantity,
            'Qty Returned': op.returned_quantity,
            'Qty Accepted': op.accepted_quantity,
            'Qty Rejected': op.rejected_quantity,
            'Qty Rework': op.rework_quantity,
            'Qty Scrapped': op.scrapped_quantity,
            'Pending': op.pending_quantity,
            'Completion %': op.completion_percent,
            'Planned Issue': str(op.planned_issue_date) if op.planned_issue_date else '',
            'Due Date': str(op.due_date) if op.due_date else '',
            'Actual Start': str(op.actual_start) if op.actual_start else '',
            'Actual End': str(op.actual_end) if op.actual_end else '',
            'Status': op.status,
            'Overdue': 'Yes' if op.is_overdue else 'No',
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Vendor Processes', index=False)

    output.seek(0)
    return output




# ============ Inventory: stock position & ledger ============
#
# Balances are never stored twice. Each stage of the flow is derived from the
# ledger that already owns it:
#   raw material     <- MaterialLot.quantity_available (receipt - issue - scrap)
#   allocated        <- MaterialLot.quantity_allocated (reserved, not moved)
#   WIP              <- OperationTransaction issued - returned (in-house ops)
#   at subcontractor <- OperationTransaction issued - returned (vendor ops)
#   finished goods   <- PackagingRecord.quantity_accepted (warehoused/dispatched)
#   rejected         <- receipt-line rejections + operation/inspection rejects
#   scrap            <- MaterialMovement.scrap + OperationTransaction.scrap
# Adjustments and transfers are their own ledgers and are netted in.

INVENTORY_CATEGORIES = (
    ('raw_material', 'Raw Material'),
    ('allocated', 'Allocated (reserved)'),
    ('wip', 'Work in Progress'),
    ('subcontractor', 'At Subcontractor'),
    ('finished_goods', 'Finished Goods'),
    ('rejected', 'Rejected'),
    ('scrap', 'Scrap'),
)

# Adjustment types that add stock (everything else removes it).
STOCK_ADD_TYPES = ('increase', 'reversal')


def get_company_settings():
    """Return company identity from SystemSetting, falling back to config.

    The General Settings page writes these to SystemSetting, but the document
    header and page footer used to read them straight from app.config (i.e. the
    environment), so editing the page had no effect. Callers now use this helper
    so a saved value actually shows up, while an unset value still falls back to
    the configured default.
    """
    from flask import current_app
    from app.models import SystemSetting

    keys = ('COMPANY_NAME', 'COMPANY_ADDRESS', 'COMPANY_PHONE',
            'COMPANY_EMAIL', 'COMPANY_GST')
    out = {}
    try:
        rows = {s.setting_key: s.setting_value
                for s in SystemSetting.query.filter(
                    SystemSetting.setting_key.in_(keys)).all()}
    except Exception:
        rows = {}
    for key in keys:
        value = rows.get(key)
        if value is None or str(value).strip() == '':
            value = current_app.config.get(key, '')
        out[key] = value
    return out


def get_costing_defaults():
    """Return overhead/profit/scrap percentages, DB setting first, config second."""
    from flask import current_app
    from app.models import SystemSetting

    defaults = {
        'DEFAULT_OVERHEAD_PERCENT': current_app.config.get('DEFAULT_OVERHEAD_PERCENT', 15),
        'DEFAULT_PROFIT_PERCENT': current_app.config.get('DEFAULT_PROFIT_PERCENT', 20),
        'DEFAULT_SCRAP_PERCENT': current_app.config.get('DEFAULT_SCRAP_PERCENT', 2),
    }
    try:
        rows = {s.setting_key: s.setting_value
                for s in SystemSetting.query.filter(
                    SystemSetting.setting_key.in_(list(defaults))).all()}
    except Exception:
        rows = {}
    out = {}
    for key, fallback in defaults.items():
        raw = rows.get(key)
        try:
            out[key] = float(raw) if raw not in (None, '') else float(fallback)
        except (TypeError, ValueError):
            out[key] = float(fallback)
    return out


def get_stock_policy():
    """Read the stock policy from SystemSetting.

    Business rules are configurable rather than hard-coded:
      * allow_negative_stock     - permit a negative lot balance (default off)
      * allow_over_allocation    - permit allocating more than is free (default off)
    Both default to False, i.e. the safe behaviour the app already enforced.
    """
    from app.models import SystemSetting

    def _flag(key, default=False):
        row = SystemSetting.query.filter_by(setting_key=key).first()
        if row is None or row.setting_value is None:
            return default
        return str(row.setting_value).strip().lower() in ('1', 'true', 'yes', 'on')

    return {
        'allow_negative_stock': _flag('allow_negative_stock', False),
        'allow_over_allocation': _flag('allow_over_allocation', False),
    }


def lot_adjustment_delta(lot):
    """Net quantity added to a lot by its stock adjustments."""
    return sum(a.signed_quantity for a in lot.adjustments.all())


def lot_position(lot):
    """Full derived position for one material lot (available/allocated/outstanding)."""
    received = lot.quantity_received or 0
    issued = lot.quantity_issued
    scrapped = lot.quantity_scrapped
    returned = lot.quantity_returned
    allocated = lot.quantity_allocated
    shortage = lot.quantity_shortage
    adjustment = lot_adjustment_delta(lot)

    available = received - issued - scrapped + adjustment
    return {
        'lot': lot,
        'received': received,
        'issued': issued,
        'returned': returned,
        'scrapped': scrapped,
        'shortage': shortage,
        'adjustment': adjustment,
        'allocated': allocated,
        'available': available,
        'remaining': available - allocated,
        'unit': lot.unit,
    }


def compute_inventory_position(location_kind=None):
    """Whole-plant stock position, grouped by category.

    ``location_kind`` (optional) restricts the transfer-based views to one kind
    of location; the derived balances are plant-wide by nature and always shown.
    """
    from app.models import (MaterialLot, OrderOperation, OperationTransaction,
                            PackagingRecord, SupplierOrderReceiptLine)

    totals = {key: 0.0 for key, _ in INVENTORY_CATEGORIES}

    for lot in MaterialLot.query.all():
        pos = lot_position(lot)
        totals['raw_material'] += pos['available']
        totals['allocated'] += pos['allocated']
        totals['scrap'] += pos['scrapped']

    # WIP and subcontractor stock from the operation ledgers.
    ops = OrderOperation.query.all()
    for op in ops:
        outstanding = (op.issued_quantity - op.returned_quantity
                       - op.rejected_quantity - op.scrapped_quantity)
        if op.is_vendor:
            totals['subcontractor'] += max(outstanding, 0)
        else:
            totals['wip'] += max(outstanding, 0)

    # Finished goods still in the warehouse (packed but not yet dispatched).
    for pack in PackagingRecord.query.filter(
            PackagingRecord.status.in_(('packed', 'warehoused'))).all():
        totals['finished_goods'] += pack.quantity_accepted or 0

    # Rejected: receipt-line rejections + operation rejects.
    totals['rejected'] += sum(
        (l.quantity_rejected or 0) for l in SupplierOrderReceiptLine.query.all())
    totals['rejected'] += sum(
        t.quantity or 0 for t in OperationTransaction.query.filter_by(
            transaction_type='reject').all())

    # Scrap from operation transactions (lot scrap already counted above).
    totals['scrap'] += sum(
        t.quantity or 0 for t in OperationTransaction.query.filter_by(
            transaction_type='scrap').all())

    return totals


def build_stock_ledger(lot=None, order=None, limit=None):
    """Merge every stock-affecting event into one chronological ledger.

    Combines material movements, operation transactions, transfers and
    adjustments so a single list shows receipts, allocations, issues, returns,
    transfers, adjustments, rejections and warehouse receipts with their links
    back to POs, lots/heats, orders, vendors and documents.
    """
    from app.models import (MaterialMovement, OperationTransaction, StockTransfer,
                            StockAdjustment, PackagingRecord)

    events = []

    def add(when, kind, qty, unit, title, detail='', refs=None, direction='neutral'):
        if when is None:
            return
        events.append({
            'date': when,
            'sort_key': when,
            'kind': kind,
            'quantity': qty,
            'unit': unit or '',
            'title': title,
            'detail': detail,
            'refs': refs or [],
            'direction': direction,
        })

    mv_query = MaterialMovement.query
    if lot is not None:
        mv_query = mv_query.filter_by(lot_id=lot.id)
    if order is not None:
        mv_query = mv_query.filter_by(order_id=order.id)
    for m in mv_query.all():
        l = m.lot
        title = f'{m.movement_type.title()} {m.quantity:g} {l.unit if l else ""}'.strip()
        detail = f'Lot {l.lot_number}' if l else ''
        if m.order is not None:
            detail += f' · Order {m.order.order_number}'
        add(m.movement_date, f'movement_{m.movement_type}', m.quantity,
            l.unit if l else '', title, detail,
            refs=[r for r in (m.reference, l.heat_number if l else None,
                              l.batch_number if l else None) if r])

    txn_query = OperationTransaction.query
    if order is not None:
        txn_query = txn_query.join(OrderOperation).filter(
            OrderOperation.order_id == order.id)
    for t in txn_query.all():
        op = t.operation
        label = op.operation_name if op else ''
        if op is not None and op.is_vendor and op.vendor:
            label += f' ({op.vendor.vendor_name})'
        title = f'{t.transaction_type.title()} {t.quantity} — {label}'
        add(t.transaction_date, f'txn_{t.transaction_type}', t.quantity, 'pcs',
            title, t.reference or '', refs=[t.reference] if t.reference else [])

    tr_query = StockTransfer.query
    if lot is not None:
        tr_query = tr_query.filter_by(material_lot_id=lot.id)
    if order is not None:
        tr_query = tr_query.filter_by(order_id=order.id)
    for tr in tr_query.all():
        src = tr.from_location.name if tr.from_location else '—'
        dst = tr.to_location.name if tr.to_location else '—'
        add(tr.transfer_date, 'transfer', tr.quantity, tr.unit,
            f'Transfer {tr.quantity:g} {tr.unit or ""}: {src} → {dst}',
            tr.reference or '', refs=[tr.reference] if tr.reference else [],
            direction='neutral' if tr.is_effective else 'pending')

    adj_query = StockAdjustment.query
    if lot is not None:
        adj_query = adj_query.filter_by(material_lot_id=lot.id)
    if order is not None:
        adj_query = adj_query.filter_by(order_id=order.id)
    for a in adj_query.all():
        add(a.adjustment_date, f'adjustment_{a.adjustment_type}', a.quantity,
            a.unit, f'Adjustment ({a.adjustment_type.replace("_", " ")}) '
            f'{a.quantity:g} {a.unit or ""}', a.reason or '',
            refs=[a.reference] if a.reference else [])

    if order is not None:
        for p in order.packaging.all():
            if p.status in ('warehoused', 'dispatched'):
                add(p.warehouse_receipt_date or p.packaging_date, 'warehouse_receipt',
                    p.quantity_accepted, 'pcs',
                    f'Warehouse receipt {p.warehouse_receipt_number or ""}'.strip(),
                    f'{p.quantity_accepted} accepted · {p.location or ""}'.strip(' ·'),
                    refs=[p.warehouse_receipt_number] if p.warehouse_receipt_number else [])

    events.sort(key=lambda e: (e['sort_key'], e['kind']))
    if limit:
        events = events[-limit:]
    return events


def generate_transfer_number(prefix='TRF'):
    """Generate a unique stock transfer number (TRF-YYYY-NNNN)."""
    from app.models import StockTransfer
    return _next_document_number(StockTransfer, 'transfer_number', prefix)


def generate_adjustment_number(prefix='ADJ'):
    """Generate a unique stock adjustment number (ADJ-YYYY-NNNN)."""
    from app.models import StockAdjustment
    return _next_document_number(StockAdjustment, 'adjustment_number', prefix)


def _next_document_number(model, field, prefix):
    """Shared YYYY-scoped sequential number generator."""
    year = datetime.now().year
    column = getattr(model, field)
    last = model.query.filter(column.like(f'{prefix}-{year}-%')).order_by(
        model.id.desc()).first()
    new_num = 1
    if last:
        try:
            new_num = int(getattr(last, field).split('-')[-1]) + 1
        except (ValueError, TypeError):
            new_num = 1
    return f"{prefix}-{year}-{new_num:04d}"


def export_stock_position_to_excel(rows):
    """Export the stock position rows to Excel."""
    data = [{
        'Category': r['label'],
        'Quantity': r['quantity'],
    } for r in rows]
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Stock Position', index=False)
    output.seek(0)
    return output


def export_stock_ledger_to_excel(events):
    """Export the merged stock ledger to Excel."""
    data = [{
        'Date': str(e['date']) if e['date'] else '',
        'Type': e['kind'],
        'Quantity': e['quantity'],
        'Unit': e['unit'],
        'Title': e['title'],
        'Detail': e['detail'],
        'References': ', '.join(e['refs']),
    } for e in events]
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Stock Ledger', index=False)
    output.seek(0)
    return output


# ============ Costing: estimated vs actual ============

COST_CATEGORIES = ('material', 'subcontracting', 'machining', 'finishing',
                   'inspection', 'packaging', 'freight', 'other')


def compute_order_cost(order):
    """Compare the ESTIMATED and ACTUAL cost of an order.

    Estimates come from the quotation chain (QuotationItem / ProductCostHistory)
    when the order was raised from a won quotation; otherwise they are reported
    as missing rather than invented. Actuals come only from recorded
    OrderCostEntry rows, so nothing is guessed and nothing is counted twice.
    """
    from app.models import QuotationItem, ProductCostHistory

    entries = order.cost_entries.all()

    actual = {c: 0.0 for c in COST_CATEGORIES}
    for e in entries:
        if e.category in actual:
            actual[e.category] += e.effective_amount

    estimate = {c: 0.0 for c in COST_CATEGORIES}
    estimate_source = None

    item = None
    if order.quotation_item_id:
        item = QuotationItem.query.get(order.quotation_item_id)
    if item is None and order.quotation_id:
        q = order.quotation
        if q is not None:
            item = q.items.order_by(QuotationItem.id).first()
    if item is not None:
        estimate_source = f'Quotation item #{item.id}'
        qty = item.quantity_1 or 0
        estimate['material'] = (item.material_cost or 0) * qty
        estimate['machining'] = (item.machining_cost or 0) * qty
        estimate['subcontracting'] = (item.vendor_cost or 0) * qty
        estimate['other'] = (item.other_cost or 0) * qty
        estimate['freight'] = 0.0
    elif order.product_id:
        hist = ProductCostHistory.query.filter_by(
            product_id=order.product_id).order_by(
            ProductCostHistory.recorded_at.desc()).first()
        if hist is not None:
            estimate_source = f'Product cost history #{hist.id}'
            qty = hist.quantity or order.quantity_ordered or 0
            estimate['material'] = (hist.raw_material_cost or 0) * qty
            estimate['machining'] = (hist.machining_cost or 0) * qty
            estimate['subcontracting'] = (hist.vendor_cost or 0) * qty
            estimate['other'] = (hist.tooling_cost or 0) * qty

    actual_total = sum(actual.values())
    estimate_total = sum(estimate.values())
    categories = [(c, c.replace('_', ' ').title()) for c in COST_CATEGORIES]

    # Categories with an estimate but no actual (or vice versa) are surfaced so
    # the user can see exactly which input is missing.
    missing = []
    for c in COST_CATEGORIES:
        if estimate[c] > 0 and actual[c] <= 0:
            missing.append(f'No actual {c.replace("_", " ")} recorded')
        if estimate[c] <= 0 and actual[c] > 0 and c != 'other':
            missing.append(f'No estimate for {c.replace("_", " ")}')

    variance = actual_total - estimate_total
    variance_percent = (variance / estimate_total * 100) if estimate_total else None

    qty = order.quantity_ordered or 0
    return {
        'categories': categories,
        'estimate': estimate,
        'actual': actual,
        'estimate_total': estimate_total,
        'actual_total': actual_total,
        'variance': variance,
        'variance_percent': variance_percent,
        'estimate_source': estimate_source,
        'has_estimate': estimate_total > 0,
        'has_actual': actual_total > 0,
        'missing': missing,
        'entry_count': len(entries),
        'unit_cost_actual': (actual_total / qty) if qty else None,
        'unit_cost_estimate': (estimate_total / qty) if qty else None,
        'order_value': order.order_value,
        'margin_actual': (order.order_value - actual_total) if actual_total else None,
        'margin_percent_actual': (
            (order.order_value - actual_total) / order.order_value * 100
            if order.order_value and actual_total else None),
    }


def export_order_costs_to_excel(rows):
    """Export the order cost comparison rows to Excel."""
    data = []
    for r in rows:
        data.append({
            'Order': r['order'].order_number,
            'Customer': r['order'].customer.company_name if r['order'].customer else '',
            'Part': r['order'].part_description,
            'Qty': r['order'].quantity_ordered,
            'Estimate Source': r['cost']['estimate_source'] or '—',
            'Estimated Total': r['cost']['estimate_total'],
            'Actual Total': r['cost']['actual_total'],
            'Variance': r['cost']['variance'],
            'Variance %': r['cost']['variance_percent'],
            'Order Value': r['cost']['order_value'],
            'Margin %': r['cost']['margin_percent_actual'],
            'Missing Inputs': '; '.join(r['cost']['missing']),
        })
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Order Costs', index=False)
    output.seek(0)
    return output


# ============ Procurement: supplier quote comparison ============

def supplier_quality_history(supplier_id):
    """Derive a 0-100 quality score from recorded evidence (no invention).

    Uses receipt-line rejection ratio and any vendor quality issues. Returns
    ``None`` when there is no evidence at all, so the caller can say "not
    measurable" instead of assuming a value.
    """
    from app.models import (SupplierOrderReceiptLine, SupplierOrderReceipt,
                            VendorQualityIssue, Vendor, VendorWorkOrder)

    receipts = SupplierOrderReceipt.query.filter_by(supplier_id=supplier_id).all()
    received = 0.0
    rejected = 0.0
    for r in receipts:
        for line in r.lines.all():
            received += line.quantity_received or 0
            rejected += line.quantity_rejected or 0

    if received <= 0:
        return None, 'no receipt history'
    score = max(0.0, min(100.0, 100.0 * (1 - rejected / received)))
    return round(score, 1), f'{rejected:g}/{received:g} rejected'


def compare_supplier_quotes(rfq, weights=None):
    """Score and rank the quotes on an RFQ, and recommend one.

    Comparable factors, all normalised to 0-100 so they can be weighted:
      * price       - landed (tax-inclusive) unit price converted to the RFQ
                      currency; cheapest scores 100
      * lead_time   - shorter is better; fastest scores 100
      * quality     - evidence-based supplier history (or the manual rating)
      * payment     - earlier payment terms score higher (cash-flow friendly)
      * delivery    - more favourable delivery terms score higher

    Returns a dict with the scored quotes, the recommended quote and the
    reasons. It never awards anything: awarding is a separate authorised action.
    """
    from app.models import Currency

    weights = weights or {'price': 40, 'lead_time': 20, 'quality': 20,
                          'payment': 10, 'delivery': 10}
    target_currency = rfq.currency or 'INR'
    rates = {c.currency_code: c.exchange_rate for c in Currency.query.all()
             if c.exchange_rate} or None

    quotes = rfq.quotes.all()
    scored = []
    for q in quotes:
        price = convert_currency(q.landed_price, q.currency or 'INR',
                                 target_currency, rates)
        history_score, history_note = supplier_quality_history(q.supplier_id)
        quality = q.quality_rating if q.quality_rating is not None else history_score
        scored.append({
            'quote': q,
            'price_base': price,
            'quality': quality,
            'quality_note': history_note,
        })

    if not scored:
        return {'rows': [], 'recommended': None, 'reasons': [],
                'weights': weights, 'target_currency': target_currency,
                'missing': ['No quotes recorded yet']}

    prices = [s['price_base'] for s in scored if s['price_base'] > 0]
    leads = [s['quote'].lead_time_days for s in scored
             if s['quote'].lead_time_days is not None]
    min_price, max_price = (min(prices), max(prices)) if prices else (0, 0)
    min_lead, max_lead = (min(leads), max(leads)) if leads else (0, 0)

    missing = []
    for s in scored:
        if s['price_base'] <= 0:
            missing.append(f"{s['quote'].supplier.company_name if s['quote'].supplier else 'supplier'}: no price")
        if s['quote'].lead_time_days is None:
            missing.append(f"{s['quote'].supplier.company_name if s['quote'].supplier else 'supplier'}: no lead time")
        if s['quality'] is None:
            missing.append(f"{s['quote'].supplier.company_name if s['quote'].supplier else 'supplier'}: no quality evidence")

    def _score_low_is_best(value, lo, hi):
        if value is None or hi <= lo:
            return 50.0  # neutral when there is nothing to compare
        return 100.0 * (hi - value) / (hi - lo)

    for s in scored:
        q = s['quote']
        price_score = _score_low_is_best(s['price_base'], min_price, max_price) \
            if s['price_base'] > 0 else 50.0
        lead_score = _score_low_is_best(q.lead_time_days, min_lead, max_lead)
        quality_score = s['quality'] if s['quality'] is not None else 50.0
        payment_score = _payment_terms_score(q.payment_terms)
        delivery_score = _delivery_terms_score(q.delivery_terms)
        total = (price_score * weights['price'] + lead_score * weights['lead_time']
                 + quality_score * weights['quality']
                 + payment_score * weights['payment']
                 + delivery_score * weights['delivery']) / sum(weights.values())
        s.update({
            'price_score': round(price_score, 1),
            'lead_score': round(lead_score, 1),
            'quality_score': round(quality_score, 1),
            'payment_score': round(payment_score, 1),
            'delivery_score': round(delivery_score, 1),
            'total_score': round(total, 1),
        })

    scored.sort(key=lambda s: s['total_score'], reverse=True)
    for rank, s in enumerate(scored, start=1):
        s['rank'] = rank

    best = scored[0]
    reasons = [
        f"Weighted score {best['total_score']}/100 "
        f"(price {best['price_score']}, lead time {best['lead_score']}, "
        f"quality {best['quality_score']}, payment {best['payment_score']}, "
        f"delivery {best['delivery_score']})",
    ]
    cheapest = min(scored, key=lambda s: s['price_base'] or float('inf'))
    if cheapest is best:
        reasons.append('Lowest landed price')
    else:
        reasons.append(
            f"Not the cheapest ({best['price_base']:.2f} vs "
            f"{cheapest['price_base']:.2f} {target_currency}); better overall terms")

    return {
        'rows': scored,
        'recommended': best,
        'reasons': reasons,
        'weights': weights,
        'target_currency': target_currency,
        'missing': missing,
    }


def _payment_terms_score(terms):
    """Score payment terms 0-100; longer credit is better for the buyer."""
    if not terms:
        return 50.0
    import re
    m = re.search(r'(\d+)', str(terms))
    if not m:
        return 50.0
    days = int(m.group(1))
    return max(0.0, min(100.0, days / 60.0 * 100))


def _delivery_terms_score(terms):
    """Score delivery terms 0-100 from keywords; neutral when unknown."""
    if not terms:
        return 50.0
    t = str(terms).lower()
    if any(k in t for k in ('door', 'delivered', 'ddp', 'free', 'included')):
        return 100.0
    if any(k in t for k in ('ex-works', 'exworks', 'ex works', 'fob', 'exw')):
        return 40.0
    return 50.0


def generate_supplier_rfq_number(prefix='SRFQ'):
    """Generate a unique supplier RFQ number (SRFQ-YYYY-NNNN)."""
    from app.models import SupplierRFQ
    return _next_document_number(SupplierRFQ, 'rfq_number', prefix)


def generate_supplier_quote_number(prefix='SQT'):
    """Supplier quotes are keyed by supplier + reference; kept for symmetry."""
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def export_quote_comparison_to_excel(rfq, comparison):
    """Export a supplier quote comparison to Excel."""
    data = []
    for s in comparison['rows']:
        q = s['quote']
        data.append({
            'Rank': s['rank'],
            'Supplier': q.supplier.company_name if q.supplier else '',
            'Quote Ref': q.quote_reference,
            'Unit Price': q.unit_price,
            'Currency': q.currency,
            'Landed Unit Price': q.landed_price,
            f'Price ({comparison["target_currency"]})': s['price_base'],
            'Lead Time (days)': q.lead_time_days,
            'Payment Terms': q.payment_terms,
            'Delivery Terms': q.delivery_terms,
            'Quality': s['quality'],
            'Price Score': s['price_score'],
            'Lead Score': s['lead_score'],
            'Quality Score': s['quality_score'],
            'Payment Score': s['payment_score'],
            'Delivery Score': s['delivery_score'],
            'Total Score': s['total_score'],
            'Recommended': 'Yes' if s is comparison['recommended'] else 'No',
        })
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Quote Comparison', index=False)
    output.seek(0)
    return output


def export_documents_to_excel(documents):
    """Export the document register to XLSX."""
    rows = []
    for d in documents:
        rows.append({
            'Title': d.display_name,
            'File Name': d.original_name,
            'Category': (d.category or '').replace('_', ' ').title(),
            'Revision': d.revision or '',
            'Current': 'Yes' if d.is_current else 'No',
            'Linked To': d.entity_type,
            'Record ID': d.entity_id,
            'Issue Date': d.issue_date or '',
            'Size (KB)': round((d.file_size or 0) / 1024, 1),
            'Uploaded': d.uploaded_at,
        })
    output = BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output


# ============ Notifications ============

def notify_roles(roles, title, message, notification_type,
                 reference_type=None, reference_id=None, exclude_user_id=None):
    """Create one notification per active user whose role is in ``roles``.

    A thin fan-out over ``create_notification`` so a single workflow event can
    reach the people who care about it (e.g. a new order notifies admin and
    production). Users are looked up by their stored role, so no role list is
    hard-coded here beyond what the caller passes. Returns the notifications
    created (not yet committed).
    """
    from app.models import User

    created = []
    for user in User.query.filter(User.role.in_(list(roles)),
                                  User.is_active.is_(True)).all():
        if exclude_user_id and user.id == exclude_user_id:
            continue
        created.append(create_notification(
            user.id, title, message, notification_type,
            reference_type=reference_type, reference_id=reference_id))
    return created


# ============ Secure file handling ============

def resolve_upload_path(file_path, upload_root='uploads'):
    """Resolve a stored attachment path to an absolute path inside the upload root.

    Returns the absolute path, or None when the file is missing or would resolve
    outside the upload root (path-traversal guard). Downloads must go through
    this so a manipulated DB value can never escape the uploads directory.
    """
    if not file_path:
        return None
    root = os.path.abspath(upload_root)
    candidate = os.path.abspath(file_path)
    if not candidate.startswith(root + os.sep):
        joined = os.path.abspath(os.path.join(upload_root, file_path))
        if not joined.startswith(root + os.sep):
            return None
        candidate = joined
    if not os.path.isfile(candidate):
        return None
    return candidate


def inline_ok(file_type):
    """Whether a file type is safe to render inline in the browser."""
    ext = (file_type or '').lower().lstrip('.')
    return ext in INLINE_EXTENSIONS and ext not in NEVER_INLINE_EXTENSIONS


# ============ Dispatch: shipments against orders ============

def packaging_available(record):
    """Accepted quantity on a packing record that has not yet been dispatched."""
    shipped = sum((l.quantity or 0) for l in record.dispatch_lines.all())
    return max((record.quantity_accepted or 0) - shipped, 0)


def compute_order_dispatch_summary(order):
    """Order-level dispatch position derived from dispatches and packing records.

    ``warehoused`` is what has been accepted into finished-goods stock;
    ``shipped`` is what has left on a dispatch; ``remaining`` is the difference.
    ``remaining_to_produce`` keeps the production picture separate so a partial
    dispatch never looks like a completed order.
    """
    warehoused = sum((p.quantity_accepted or 0) for p in order.packaging.all()
                     if p.status in ('warehoused', 'dispatched'))
    dispatches = order.dispatches.all()
    shipped = sum(d.quantity_shipped for d in dispatches
                  if d.status != 'cancelled')
    remaining = max(warehoused - shipped, 0)
    delivered = sum(d.quantity_shipped for d in dispatches if d.is_delivered)
    ordered = order.quantity_ordered or 0
    return {
        'ordered': ordered,
        'produced': order.quantity_produced,
        'warehoused': warehoused,
        'shipped': shipped,
        'remaining': remaining,
        'delivered': delivered,
        'remaining_to_produce': max(ordered - order.quantity_produced, 0),
        'dispatch_count': len(dispatches),
        'open_dispatches': sum(1 for d in dispatches if d.is_open),
        'is_partial': 0 < shipped < ordered,
        'is_fully_shipped': ordered > 0 and shipped >= ordered,
    }


def generate_dispatch_number(prefix='DSP'):
    """Generate a unique dispatch number like DSP-2026-0001."""
    from app.models import Dispatch
    year = datetime.now().year
    like = f'{prefix}-{year}-%'
    last = Dispatch.query.filter(Dispatch.dispatch_number.like(like)).order_by(
        Dispatch.dispatch_number.desc()).first()
    seq = 1
    if last:
        try:
            seq = int(last.dispatch_number.rsplit('-', 1)[1]) + 1
        except (ValueError, IndexError):
            seq = 1
    return f'{prefix}-{year}-{seq:04d}'


def export_dispatches_to_excel(dispatches):
    """Export dispatch register rows to XLSX."""
    rows = []
    for d in dispatches:
        rows.append({
            'Dispatch No': d.dispatch_number,
            'Date': d.dispatch_date,
            'Order': d.order.order_number if d.order else '',
            'Customer': d.customer.company_name if d.customer else '',
            'Qty Shipped': d.quantity_shipped,
            'Mode': d.mode,
            'Transporter': d.transporter or '',
            'Vehicle': d.vehicle_number or '',
            'Shipment Ref': d.shipment_reference or '',
            'Invoice': d.invoice_number or '',
            'Status': d.status,
            'Delivery': d.delivery_status,
            'Delivered On': d.delivered_date or '',
        })
    output = BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output


# ============ Operational alerts (derived, never duplicated) ============

def _alert(key, category, severity, title, message, url=None, days=0, date_ref=None):
    return {
        'key': key, 'category': category, 'severity': severity,
        'title': title, 'message': message, 'url': url,
        'days': days, 'date': date_ref,
    }


def compute_alerts(user=None, include_financials=True):
    """Derive operational alerts from the live ledgers.

    Every alert carries a deterministic ``alert_key`` built from the source
    record, so the same underlying condition can never produce two alerts. Only
    conditions that the records actually support are raised - nothing is
    invented. ``include_financials`` lets callers hide payment alerts from users
    without the relevant permission.
    """
    from app.models import (
        Order, OrderOperation, Inspection, SupplierPurchaseOrder,
        SupplierInvoice, VendorInvoice, Dispatch, MaterialMovement,
    )
    today = date.today()
    alerts = []

    for o in Order.query.filter(Order.due_date.isnot(None)).all():
        if o.is_overdue:
            alerts.append(_alert(
                f'order:{o.id}:overdue', 'overdue_order', 'danger',
                f'Order {o.order_number} is overdue',
                f'{o.days_overdue} day(s) past due ({o.due_date}); '
                f'{o.pending_quantity} pcs still pending.',
                url=('orders.view_order', {'id': o.id}), days=o.days_overdue,
                date_ref=o.due_date))

    for op in OrderOperation.query.filter(
            OrderOperation.is_vendor.is_(True),
            OrderOperation.due_date.isnot(None)).all():
        if op.is_open and op.due_date < today:
            alerts.append(_alert(
                f'op:{op.id}:vendor_delay', 'vendor_delay', 'warning',
                f'Vendor step overdue on {op.order.order_number}',
                f'#{op.sequence} {op.operation_name}'
                + (f' ({op.vendor.vendor_name})' if op.vendor else '')
                + f' was due {op.due_date}; {op.pending_quantity} pcs pending.',
                url=('orders.vendor_processes', None),
                days=(today - op.due_date).days, date_ref=op.due_date))

    for o in Order.query.filter(Order.status.notin_(['cancelled', 'completed'])).all():
        req = compute_material_requirement(o)
        if req['shortage'] > 0:
            alerts.append(_alert(
                f'order:{o.id}:shortage', 'material_shortage', 'danger',
                f'Material shortage on {o.order_number}',
                f'{req["shortage"]:g} unit(s) recorded short '
                f'(planned {req["planned"]:g}, issued {req["issued"]:g}).',
                url=('orders.materials', None), date_ref=o.due_date))

    for i in Inspection.query.filter_by(result='pending').all():
        alerts.append(_alert(
            f'insp:{i.id}:pending', 'pending_inspection', 'warning',
            f'Inspection {i.inspection_number} pending',
            f'{i.inspection_type.replace("_", " ").title()} inspection on '
            f'{i.order.order_number if i.order else "-"} awaiting a result.',
            url=('orders.inspections', None), date_ref=i.inspection_date))

    for o in Order.query.filter(Order.status.notin_(['cancelled'])).all():
        rej = o.quantity_rejected
        rw = o.quantity_rework
        if rej or rw:
            alerts.append(_alert(
                f'order:{o.id}:rejection', 'rejection_rework',
                'danger' if rej else 'warning',
                f'Rejected/rework on {o.order_number}',
                f'{rej} pcs rejected, {rw} pcs rework across the route.',
                url=('orders.view_order', {'id': o.id}), date_ref=o.due_date))

    for po in SupplierPurchaseOrder.query.all():
        if po.is_overdue:
            alerts.append(_alert(
                f'po:{po.id}:late', 'late_receipt', 'warning',
                f'PO {po.po_number} delivery late',
                f'Expected {po.expected_date}; {po.pending_quantity:g} '
                f'unit(s) still pending from '
                f'{po.supplier.company_name if po.supplier else "-"}.',
                url=('procurement.view_purchase_order', {'id': po.id}),
                days=po.days_overdue, date_ref=po.expected_date))

    if include_financials:
        for inv in SupplierInvoice.query.all():
            if inv.status != 'cancelled' and inv.outstanding_amount > 0 \
                    and inv.due_date and inv.due_date < today:
                alerts.append(_alert(
                    f'sinv:{inv.id}:overdue', 'overdue_supplier_payment', 'danger',
                    f'Supplier invoice {inv.invoice_number} overdue',
                    f'{inv.outstanding_amount:,.2f} {inv.currency or "INR"} outstanding '
                    f'to {inv.supplier.company_name if inv.supplier else "-"}, '
                    f'{inv.days_overdue} day(s) past due.',
                    url=('procurement.supplier_invoices', None),
                    days=inv.days_overdue, date_ref=inv.due_date))

        for inv in VendorInvoice.query.all():
            if inv.status != 'cancelled' and inv.outstanding_amount > 0 \
                    and inv.due_date and inv.due_date < today:
                alerts.append(_alert(
                    f'vinv:{inv.id}:overdue', 'overdue_vendor_payment', 'danger',
                    f'Vendor invoice {inv.invoice_number} overdue',
                    f'{inv.outstanding_amount:,.2f} outstanding to '
                    f'{inv.vendor.vendor_name if inv.vendor else "-"}, '
                    f'{inv.days_overdue} day(s) past due.',
                    url=('vendor.view', {'id': inv.vendor_id})
                    if inv.vendor_id else None,
                    days=inv.days_overdue, date_ref=inv.due_date))

    for d in Dispatch.query.all():
        if d.is_open and d.dispatch_date and (today - d.dispatch_date).days >= 7:
            alerts.append(_alert(
                f'dispatch:{d.id}:pending', 'dispatch_pending', 'warning',
                f'Dispatch {d.dispatch_number} not delivered',
                f'Shipped {d.dispatch_date} ({d.quantity_shipped} pcs) to '
                f'{d.customer.company_name if d.customer else "-"} '
                f'{(today - d.dispatch_date).days} day(s) ago, delivery pending.',
                url=('orders.view_dispatch', {'id': d.id}),
                days=(today - d.dispatch_date).days, date_ref=d.dispatch_date))

    severity_order = {'danger': 0, 'warning': 1, 'info': 2}
    alerts.sort(key=lambda a: (severity_order.get(a['severity'], 3), -a['days']))
    return alerts


def _may_see_financials(user):
    if user is None:
        return False
    return any(user.has_permission(p) for p in ('admin', 'reports'))


def count_unacknowledged_alerts(user):
    """Number of derived alerts the given user has not yet acknowledged."""
    from app.models import AlertAcknowledgement
    alerts = compute_alerts(user=user,
                            include_financials=_may_see_financials(user))
    if not user:
        return len(alerts)
    acked = {a.alert_key for a in AlertAcknowledgement.query.filter_by(
        user_id=user.id).all()}
    return sum(1 for a in alerts if a['key'] not in acked)


# ============ Dashboard: operational metrics ============

def compute_operational_metrics(user=None, include_financials=True):
    """Accurate operational dashboard metrics, all derived from live records.

    Nothing here is a sample or estimate: each figure is a sum/count over the
    ledgers that own it. Financial totals are only meaningful when the payables
    feature is in use and the caller is allowed to see them.
    """
    from app.models import (
        Order, OrderOperation, Inspection, PackagingRecord, Dispatch,
        SupplierPurchaseOrder, SupplierInvoice, VendorInvoice, MaterialMovement,
    )
    today = date.today()
    horizon = today + timedelta(days=7)

    orders = Order.query.all()
    active = [o for o in orders if o.status not in ('cancelled',)]
    open_orders = [o for o in orders if o.is_open]
    overdue_orders = [o for o in open_orders if o.is_overdue]
    due_soon = [o for o in open_orders
                if o.due_date and today <= o.due_date <= horizon]

    by_status = {}
    for o in orders:
        by_status[o.status] = by_status.get(o.status, 0) + 1

    shortages = 0
    shortage_qty = 0.0
    for o in active:
        if o.status == 'completed':
            continue
        req = compute_material_requirement(o)
        if req['shortage'] > 0:
            shortages += 1
            shortage_qty += req['shortage']

    vendor_ops = OrderOperation.query.filter_by(is_vendor=True).all()
    open_vendor_ops = [op for op in vendor_ops if op.is_open]
    work_at_vendors = sum(op.pending_quantity for op in open_vendor_ops)
    delayed_vendor_ops = [op for op in open_vendor_ops
                          if op.due_date and op.due_date < today]

    inspections_pending = Inspection.query.filter_by(result='pending').count()
    inspections_total = Inspection.query.count()

    rejected_total = sum(o.quantity_rejected for o in orders)
    rework_total = sum(o.quantity_rework for o in orders)

    fg_records = PackagingRecord.query.filter(
        PackagingRecord.status.in_(('warehoused', 'dispatched'))).all()
    finished_goods_available = sum(packaging_available(r) for r in fg_records)
    finished_goods_warehoused = sum((r.quantity_accepted or 0) for r in fg_records)

    dispatches = Dispatch.query.all()
    open_dispatches = [d for d in dispatches if d.is_open]
    dispatched_qty = sum(d.quantity_shipped for d in dispatches if d.status != 'cancelled')
    pending_delivery = sum(1 for d in open_dispatches if not d.is_delivered)

    metrics = {
        'open_orders': len(open_orders),
        'total_orders': len(orders),
        'overdue_orders': len(overdue_orders),
        'due_soon_orders': len(due_soon),
        'orders_by_status': by_status,
        'orders_at_risk_qty': sum(o.pending_quantity for o in overdue_orders),
        'material_shortage_orders': shortages,
        'material_shortage_qty': shortage_qty,
        'open_vendor_operations': len(open_vendor_ops),
        'delayed_vendor_operations': len(delayed_vendor_ops),
        'work_at_vendors_qty': work_at_vendors,
        'inspections_pending': inspections_pending,
        'inspections_total': inspections_total,
        'rejected_quantity': rejected_total,
        'rework_quantity': rework_total,
        'finished_goods_available': finished_goods_available,
        'finished_goods_warehoused': finished_goods_warehoused,
        'dispatches_open': len(open_dispatches),
        'dispatches_total': len(dispatches),
        'dispatched_quantity': dispatched_qty,
        'dispatches_pending_delivery': pending_delivery,
        'has_operational_data': bool(orders or dispatches or inspections_total),
    }

    if include_financials:
        sup_invoices = [i for i in SupplierInvoice.query.all() if i.status != 'cancelled']
        ven_invoices = [i for i in VendorInvoice.query.all() if i.status != 'cancelled']
        supplier_outstanding = sum(i.outstanding_amount for i in sup_invoices)
        supplier_overdue = sum(i.outstanding_amount for i in sup_invoices
                               if i.due_date and i.due_date < today)
        vendor_outstanding = sum(i.outstanding_amount for i in ven_invoices)
        vendor_overdue = sum(i.outstanding_amount for i in ven_invoices
                             if i.due_date and i.due_date < today)
        open_pos = [p for p in SupplierPurchaseOrder.query.all()
                    if p.is_open and not p.is_cancelled]
        metrics.update({
            'show_financials': True,
            'supplier_payable_outstanding': supplier_outstanding,
            'supplier_payable_overdue': supplier_overdue,
            'vendor_payable_outstanding': vendor_outstanding,
            'vendor_payable_overdue': vendor_overdue,
            'open_po_count': len(open_pos),
            'open_po_value': sum(p.total_value for p in open_pos),
            'has_financial_data': bool(sup_invoices or ven_invoices or open_pos),
        })
    else:
        metrics['show_financials'] = False

    return metrics


def export_alerts_to_excel(alerts):
    """Export the current derived alert list to XLSX."""
    rows = [{
        'Severity': a['severity'],
        'Category': a['category'].replace('_', ' ').title(),
        'Title': a['title'],
        'Detail': a['message'],
        'Days': a['days'],
        'Reference Date': a['date'] or '',
    } for a in alerts]
    output = BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output


def create_notification(user_id, title, message, notification_type, reference_type=None, reference_id=None):
    """Create a notification for user."""
    from app.models import Notification, db
    
    notification = Notification(
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
        reference_type=reference_type,
        reference_id=reference_id
    )
    db.session.add(notification)
    return notification
