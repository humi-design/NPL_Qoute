"""Notification centre.

Notifications are already produced across the app by
``app.utils.create_notification`` (quotation created, material rate changed,
etc.) and shown in the topbar dropdown, but there was no way to see them all,
open the referenced record or mark them read. This module adds exactly that,
without changing how notifications are created.

A notification's ``reference_type`` / ``reference_id`` point back at the record
that caused it; ``resolve_notification_link`` turns that pair into a URL so the
dropdown and the centre can link straight to the source.
"""
from datetime import date

from flask import (render_template, redirect, url_for, request, flash, send_file)
from flask_login import login_required, current_user

from app.notifications import notifications_bp
from app.models import db, Notification

PER_PAGE_CHOICES = (10, 25, 50, 100)

# reference_type -> (view endpoint, id kwarg). Kept explicit so only known,
# safe endpoints are ever linked to.
REFERENCE_ENDPOINTS = {
    'quotation': ('quotation.view', 'id'),
    'rfq': ('rfq.view', 'id'),
    'product': ('product.view', 'id'),
    'customer': ('customer.view', 'id'),
    'supplier': ('supplier.view', 'id'),
    'vendor': ('vendor.view', 'id'),
    'order': ('orders.view_order', 'id'),
    'material': ('material.view', 'id'),
    'purchase_order': ('procurement.view_purchase_order', 'id'),
}


def resolve_notification_link(notification):
    """Return a URL for the notification's referenced record, or None."""
    entry = REFERENCE_ENDPOINTS.get(notification.reference_type)
    if not entry or not notification.reference_id:
        return None
    endpoint, kwarg = entry
    try:
        return url_for(endpoint, **{kwarg: notification.reference_id})
    except Exception:
        return None


@notifications_bp.route('/')
@login_required
def index():
    """All of the current user's notifications with read/unread filter."""
    show = request.args.get('show', 'all')  # all | unread | read
    per_page = request.args.get('per_page', 25, type=int)
    if per_page not in PER_PAGE_CHOICES:
        per_page = 25

    query = Notification.query.filter_by(user_id=current_user.id)
    if show == 'unread':
        query = query.filter_by(is_read=False)
    elif show == 'read':
        query = query.filter_by(is_read=True)

    page = request.args.get('page', 1, type=int)
    pagination = query.order_by(Notification.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False)

    rows = [{'n': n, 'link': resolve_notification_link(n)}
            for n in pagination.items]

    counts = {
        'all': Notification.query.filter_by(user_id=current_user.id).count(),
        'unread': Notification.query.filter_by(
            user_id=current_user.id, is_read=False).count(),
        'read': Notification.query.filter_by(
            user_id=current_user.id, is_read=True).count(),
    }
    return render_template('notifications/index.html', rows=rows,
                           pagination=pagination, show=show, counts=counts,
                           per_page=per_page)


@notifications_bp.route('/<int:id>/read', methods=['POST'])
@login_required
def mark_read(id):
    """Mark one notification read (only the current user's own)."""
    n = Notification.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    n.is_read = True
    db.session.commit()
    # `next` comes from a form field, so validate it before redirecting.
    from app.utils import safe_redirect_target
    return redirect(safe_redirect_target(request.form.get('next'),
                                         url_for('notifications.index')))


@notifications_bp.route('/read-all', methods=['POST'])
@login_required
def mark_all_read():
    """Mark every notification for the current user as read."""
    updated = Notification.query.filter_by(
        user_id=current_user.id, is_read=False).update({'is_read': True})
    db.session.commit()
    flash(f'{updated} notification(s) marked as read.', 'success')
    return redirect(request.referrer or url_for('notifications.index'))


@notifications_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete one of the current user's notifications."""
    n = Notification.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    db.session.delete(n)
    db.session.commit()
    flash('Notification deleted.', 'success')
    return redirect(url_for('notifications.index'))
