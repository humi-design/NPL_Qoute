"""Document management: link drawings, certificates and reports to records.

The ``Attachment`` model already stored files against an entity. This module
adds the missing pieces - a central register, document categories, revision
control and a *secure* download - without duplicating that table. Every document
still hangs off ``Attachment`` keyed by ``entity_type`` / ``entity_id``, so
documents uploaded from the product, supplier or order screens appear here too.

Security:
  * Viewing/downloading needs the ``read`` permission (all authenticated roles).
  * Uploading, editing and deleting need the ``write`` permission.
  * Files are served through ``resolve_upload_path`` which refuses any path that
    escapes the uploads directory, and non-renderable types are forced to
    download rather than being served inline.
"""
import os
from datetime import date

from flask import (render_template, redirect, url_for, request, flash,
                   send_file, abort, current_app)
from flask_login import login_required, current_user
from sqlalchemy import or_, case

from app.documents import documents_bp
from app.models import (
    db, Attachment, Product, Order, MaterialLot, Inspection, Dispatch,
    Supplier, Vendor, Customer,
)
from app.forms import DocumentForm, DocumentSearchForm
from app.utils import (upload_file, delete_file, resolve_upload_path, inline_ok,
                       export_documents_to_excel)

PER_PAGE_CHOICES = (10, 25, 50, 100)

# entity_type -> (model, label attribute, view endpoint, id kwarg)
ENTITY_REGISTRY = {
    'product': (Product, 'internal_code', 'product.view', 'id'),
    'order': (Order, 'order_number', 'orders.view_order', 'id'),
    'lot': (MaterialLot, 'lot_number', 'inventory.lot_detail', 'id'),
    'inspection': (Inspection, 'inspection_number', None, None),
    'dispatch': (Dispatch, 'dispatch_number', 'orders.view_dispatch', 'id'),
    'supplier': (Supplier, 'supplier_code', 'supplier.view', 'id'),
    'vendor': (Vendor, 'vendor_code', 'vendor.view', 'id'),
    'customer': (Customer, 'customer_code', 'customer.view', 'id'),
}

DOCUMENT_SORT_FIELDS = {
    'uploaded': Attachment.uploaded_at,
    'title': Attachment.title,
    'category': Attachment.category,
    'revision': Attachment.revision,
    'type': Attachment.entity_type,
}


def _can_write():
    return current_user.has_permission('write')


def _require_write():
    if not _can_write():
        flash('You do not have permission to modify documents.', 'danger')
        return False
    return True


def resolve_entity(entity_type, entity_id):
    """Return (record, label, url) for a linked entity, or (None, None, None)."""
    entry = ENTITY_REGISTRY.get(entity_type)
    if not entry or not entity_id:
        return None, None, None
    model, label_attr, endpoint, id_kwarg = entry
    record = model.query.get(entity_id)
    if not record:
        return None, None, None
    label = getattr(record, label_attr, None) or f'{entity_type} #{entity_id}'
    url = None
    if endpoint:
        try:
            url = url_for(endpoint, **{id_kwarg: record.id})
        except Exception:
            url = None
    return record, label, url


def _decorate(document):
    """Attach a resolved label/url to a document for display."""
    record, label, url = resolve_entity(document.entity_type, document.entity_id)
    return {
        'doc': document,
        'record': record,
        'record_label': label,
        'record_url': url,
    }


@documents_bp.route('/')
@login_required
def index():
    """Central document register with filters."""
    q = (request.args.get('q') or '').strip()
    category = request.args.get('category', '')
    entity_type = request.args.get('entity_type', '')
    current_only = request.args.get('current_only', '1') == '1'
    per_page = request.args.get('per_page', 25, type=int)
    if per_page not in PER_PAGE_CHOICES:
        per_page = 25

    query = Attachment.query
    if q:
        query = query.filter(or_(
            Attachment.original_name.ilike(f'%{q}%'),
            Attachment.title.ilike(f'%{q}%'),
            Attachment.revision.ilike(f'%{q}%'),
        ))
    if category:
        query = query.filter(Attachment.category == category)
    if entity_type:
        query = query.filter(Attachment.entity_type == entity_type)
    if current_only:
        query = query.filter(Attachment.is_current.is_(True))

    sort = request.args.get('sort', 'uploaded')
    direction = request.args.get('dir', 'desc')
    column = DOCUMENT_SORT_FIELDS.get(sort, Attachment.uploaded_at)
    order_by = column.asc() if direction == 'asc' else column.desc()

    page = request.args.get('page', 1, type=int)
    pagination = query.order_by(order_by).paginate(
        page=page, per_page=per_page, error_out=False)

    rows = [_decorate(d) for d in pagination.items]

    category_counts = dict(
        db.session.query(Attachment.category, db.func.count(Attachment.id))
        .group_by(Attachment.category).all())

    return render_template(
        'documents/index.html', rows=rows, pagination=pagination, q=q,
        category=category, entity_type=entity_type, current_only=current_only,
        per_page=per_page, sort=sort, dir=direction,
        categories=Attachment.DOCUMENT_CATEGORIES, entity_types=list(ENTITY_REGISTRY),
        category_counts=category_counts, total=Attachment.query.count())


@documents_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Upload a new document and link it to a record."""
    if not _require_write():
        return redirect(url_for('documents.index'))
    form = DocumentForm()
    # Pre-select the entity when the caller passed one in the query string.
    if request.method == 'GET':
        form.entity_type.data = request.args.get('entity_type', 'product')
        eid = request.args.get('entity_id', type=int)
        if eid:
            form.entity_id.data = eid
        form.issue_date.data = date.today()

    if form.validate_on_submit():
        file = request.files.get('file')
        if not file or not file.filename:
            flash('Please choose a file to upload.', 'danger')
            return render_template('documents/form.html', form=form, document=None)

        entity_type = form.entity_type.data
        entity_id = form.entity_id.data
        if not entity_id:
            flash('A linked record ID is required.', 'danger')
            return render_template('documents/form.html', form=form, document=None)
        record, label, _url = resolve_entity(entity_type, entity_id)
        if record is None:
            flash(f'No {entity_type} found with ID {entity_id}.', 'danger')
            return render_template('documents/form.html', form=form, document=None)

        result = upload_file(file, entity_type + 's', entity_id)
        if not result:
            flash('That file type is not allowed.', 'danger')
            return render_template('documents/form.html', form=form, document=None)

        # If a current revision already exists for the same title+category+entity,
        # mark it superseded and link the new one so history is preserved.
        prev = None
        if form.title.data:
            prev = Attachment.query.filter_by(
                entity_type=entity_type, entity_id=entity_id,
                title=form.title.data, category=form.category.data,
                is_current=True).first()

        doc = Attachment(
            entity_type=entity_type, entity_id=entity_id,
            file_name=result['file_name'], original_name=result['original_name'],
            file_type=result['file_type'], file_size=result['file_size'],
            file_path=result['file_path'],
            title=form.title.data, category=form.category.data,
            revision=form.revision.data, issue_date=form.issue_date.data,
            notes=form.notes.data, is_current=form.is_current.data,
            supersedes_id=prev.id if prev else None,
            uploaded_by=current_user.id)
        db.session.add(doc)
        if prev and form.is_current.data:
            prev.is_current = False
        db.session.commit()
        flash('Document uploaded.', 'success')
        return redirect(url_for('documents.index'))

    return render_template('documents/form.html', form=form, document=None)


@documents_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit document metadata (not the file itself)."""
    document = Attachment.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('documents.index'))
    form = DocumentForm(obj=document)
    if form.validate_on_submit():
        document.title = form.title.data
        document.category = form.category.data
        document.revision = form.revision.data
        document.issue_date = form.issue_date.data
        document.notes = form.notes.data
        document.is_current = form.is_current.data
        db.session.commit()
        flash('Document updated.', 'success')
        return redirect(url_for('documents.index'))
    return render_template('documents/form.html', form=form, document=document)


@documents_bp.route('/<int:id>/download')
@login_required
def download(id):
    """Serve a document securely (never trusting the stored path)."""
    document = Attachment.query.get_or_404(id)
    if not current_user.has_permission('read'):
        abort(403)
    path = resolve_upload_path(document.file_path,
                               current_app.config.get('UPLOAD_FOLDER', 'uploads'))
    if not path:
        flash('The stored file could not be found on disk.', 'danger')
        return redirect(url_for('documents.index'))

    inline = request.args.get('inline') == '1' and inline_ok(document.file_type)
    return send_file(path, as_attachment=not inline,
                     download_name=document.original_name,
                     conditional=True)


@documents_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete a document and its file."""
    document = Attachment.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('documents.index'))
    delete_file(document.file_path)
    db.session.delete(document)
    db.session.commit()
    flash('Document deleted.', 'success')
    return redirect(url_for('documents.index'))


@documents_bp.route('/<int:id>/supersede', methods=['POST'])
@login_required
def supersede(id):
    """Mark this document as no longer the current revision."""
    document = Attachment.query.get_or_404(id)
    if not _require_write():
        return redirect(url_for('documents.index'))
    document.is_current = False
    db.session.commit()
    flash(f'Revision {document.revision or "-"} marked as superseded.', 'success')
    return redirect(url_for('documents.index'))


@documents_bp.route('/for/<entity_type>/<int:entity_id>')
@login_required
def for_entity(entity_type, entity_id):
    """All documents linked to one record."""
    record, label, url = resolve_entity(entity_type, entity_id)
    if record is None:
        abort(404)
    docs = Attachment.query.filter_by(
        entity_type=entity_type, entity_id=entity_id).order_by(
        Attachment.uploaded_at.desc()).all()
    return render_template('documents/for_entity.html', record=record,
                           record_label=label, record_url=url, docs=docs,
                           entity_type=entity_type, entity_id=entity_id)


@documents_bp.route('/export')
@login_required
def export():
    """Export the document register to XLSX."""
    q = (request.args.get('q') or '').strip()
    category = request.args.get('category', '')
    entity_type = request.args.get('entity_type', '')
    query = Attachment.query
    if q:
        query = query.filter(or_(
            Attachment.original_name.ilike(f'%{q}%'),
            Attachment.title.ilike(f'%{q}%')))
    if category:
        query = query.filter(Attachment.category == category)
    if entity_type:
        query = query.filter(Attachment.entity_type == entity_type)
    docs = query.order_by(Attachment.uploaded_at.desc()).all()
    output = export_documents_to_excel(docs)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'documents_{date.today()}.xlsx')
