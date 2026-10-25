"""Operational alerts.

Alerts are DERIVED from the live ledgers by ``compute_alerts`` so they cannot go
stale and cannot be duplicated - each carries a deterministic key built from its
source record. The only state stored here is a user's acknowledgement, so a user
can say "I have seen this" without hiding a genuine, still-open condition from
everyone else. When the underlying condition clears the alert disappears on its
own; no cleanup job is needed.

Payment alerts are only shown to users allowed to see financials (admin /
accounts / reports) so payables data does not leak to other roles.
"""
from datetime import date

from flask import (render_template, redirect, url_for, request, flash,
                   send_file)
from flask_login import login_required, current_user
from sqlalchemy.exc import IntegrityError

from app.alerts import alerts_bp
from app.models import db, AlertAcknowledgement
from app.utils import (compute_alerts, count_unacknowledged_alerts,
                       _may_see_financials, export_alerts_to_excel)

CATEGORY_LABELS = {
    'overdue_order': 'Overdue Order',
    'vendor_delay': 'Delayed Vendor Operation',
    'material_shortage': 'Material Shortage',
    'pending_inspection': 'Pending Inspection',
    'rejection_rework': 'Rejected / Rework',
    'late_receipt': 'Late Receipt',
    'overdue_supplier_payment': 'Overdue Supplier Payment',
    'overdue_vendor_payment': 'Overdue Vendor Payment',
    'dispatch_pending': 'Dispatch Pending Delivery',
}


@alerts_bp.route('/')
@login_required
def index():
    """Alert centre: every derived alert with acknowledgement state."""
    include_fin = _may_see_financials(current_user)
    category = request.args.get('category', '')
    show = request.args.get('show', 'open')  # open | acked | all

    all_alerts = compute_alerts(user=current_user, include_financials=include_fin)

    acked = {a.alert_key for a in AlertAcknowledgement.query.filter_by(
        user_id=current_user.id).all()}
    for a in all_alerts:
        a['acknowledged'] = a['key'] in acked
        a['category_label'] = CATEGORY_LABELS.get(
            a['category'], a['category'].replace('_', ' ').title())

    # Severity counts describe the whole alert set, not the filtered view.
    counts = {'danger': 0, 'warning': 0, 'info': 0}
    for a in all_alerts:
        counts[a['severity']] = counts.get(a['severity'], 0) + 1

    alerts = all_alerts
    if category:
        alerts = [a for a in alerts if a['category'] == category]
    if show == 'open':
        alerts = [a for a in alerts if not a['acknowledged']]
    elif show == 'acked':
        alerts = [a for a in alerts if a['acknowledged']]

    return render_template('alerts/index.html', alerts=alerts, counts=counts,
                           category=category, show=show,
                           categories=CATEGORY_LABELS,
                           show_financials=include_fin)


@alerts_bp.route('/ack', methods=['POST'])
@login_required
def acknowledge():
    """Acknowledge one or more alerts for the current user."""
    keys = request.form.getlist('alert_key')
    if not keys:
        flash('No alerts selected.', 'warning')
        return redirect(url_for('alerts.index'))

    note = (request.form.get('note') or '').strip() or None
    existing = {a.alert_key for a in AlertAcknowledgement.query.filter_by(
        user_id=current_user.id).all()}
    added = 0
    for key in keys:
        if key in existing:
            continue
        db.session.add(AlertAcknowledgement(
            alert_key=key, user_id=current_user.id, note=note))
        added += 1
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
    flash(f'{added} alert(s) acknowledged.', 'success')
    return redirect(request.referrer or url_for('alerts.index'))


@alerts_bp.route('/<path:alert_key>/unacknowledge', methods=['POST'])
@login_required
def unacknowledge(alert_key):
    """Remove an acknowledgement so the alert shows as open again."""
    ack = AlertAcknowledgement.query.filter_by(
        alert_key=alert_key, user_id=current_user.id).first()
    if ack:
        db.session.delete(ack)
        db.session.commit()
        flash('Alert reopened.', 'success')
    return redirect(request.referrer or url_for('alerts.index'))


@alerts_bp.route('/export')
@login_required
def export():
    """Export the current alert list to XLSX."""
    include_fin = _may_see_financials(current_user)
    alerts = compute_alerts(user=current_user, include_financials=include_fin)
    output = export_alerts_to_excel(alerts)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'alerts_{date.today()}.xlsx')
