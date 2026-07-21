from flask import render_template, request, send_file
from flask_login import login_required
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from io import BytesIO
import pandas as pd
from app.report import report_bp
from app.models import db, Quotation, QuotationItem, Product, Customer, ProductCostHistory


@report_bp.route('/')
@login_required
def index():
    """Reports list page."""
    return render_template('report/index.html')


@report_bp.route('/quotations')
@login_required
def quotations():
    """Quotation report."""
    status = request.args.get('status', '')
    date_from = request.args.get('date_from', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    date_to = request.args.get('date_to', datetime.now().strftime('%Y-%m-%d'))
    
    query = Quotation.query
    
    if status:
        query = query.filter(Quotation.status == status)
    
    if date_from:
        query = query.filter(Quotation.quotation_date >= date_from)
    if date_to:
        query = query.filter(Quotation.quotation_date <= date_to)
    
    quotations = query.order_by(Quotation.quotation_date.desc()).all()
    
    # Calculate stats
    total_count = len(quotations)
    won_count = sum(1 for q in quotations if q.status == 'won')
    lost_count = sum(1 for q in quotations if q.status == 'lost')
    total_value = sum(q.total_amount for q in quotations if q.status == 'won')
    
    return render_template('report/quotations.html',
                         quotations=quotations,
                         status=status,
                         date_from=date_from,
                         date_to=date_to,
                         total_count=total_count,
                         won_count=won_count,
                         lost_count=lost_count,
                         total_value=total_value)


@report_bp.route('/customers')
@login_required
def customers():
    """Customer-wise sales report."""
    date_from = request.args.get('date_from', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    date_to = request.args.get('date_to', datetime.now().strftime('%Y-%m-%d'))
    
    results = db.session.query(
        Customer.id,
        Customer.customer_code,
        Customer.company_name,
        func.count(Quotation.id).label('quote_count'),
        func.sum(Quotation.total_amount).label('total_value')
    ).join(Quotation).filter(
        Quotation.status == 'won',
        Quotation.won_date >= date_from,
        Quotation.won_date <= date_to
    ).group_by(Customer.id).order_by(desc('total_value')).all()
    
    return render_template('report/customers.html',
                         results=results,
                         date_from=date_from,
                         date_to=date_to)


@report_bp.route('/products')
@login_required
def products():
    """Product performance report."""
    results = db.session.query(
        Product.id,
        Product.internal_code,
        Product.product_name,
        func.count(QuotationItem.id).label('quote_count'),
        func.sum(QuotationItem.quantity_1).label('total_qty')
    ).join(QuotationItem).join(Quotation).filter(
        Quotation.status == 'won'
    ).group_by(Product.id).order_by(desc('total_qty')).limit(50).all()
    
    return render_template('report/products.html', results=results)


@report_bp.route('/margin')
@login_required
def margin():
    """Margin analysis report."""
    items = db.session.query(
        QuotationItem,
        Quotation,
        Product
    ).join(Quotation).join(Product, QuotationItem.product_id == Product.id).filter(
        Quotation.status == 'won',
        QuotationItem.unit_price_1 > 0
    ).order_by(desc(Quotation.quotation_date)).limit(100).all()
    
    return render_template('report/margin.html', items=items)


@report_bp.route('/export')
@login_required
def export_report():
    """Export report to Excel."""
    report_type = request.args.get('type', 'quotations')
    
    output = BytesIO()
    
    if report_type == 'quotations':
        quotations = Quotation.query.order_by(Quotation.quotation_date.desc()).all()
        data = [{
            'Quote #': q.quotation_number,
            'Date': q.quotation_date,
            'Customer': q.customer.company_name if q.customer else '',
            'Amount': q.total_amount,
            'Status': q.status,
            'Currency': q.currency
        } for q in quotations]
    elif report_type == 'customers':
        results = db.session.query(
            Customer.id,
            Customer.company_name,
            func.sum(Quotation.total_amount).label('total')
        ).join(Quotation).filter(Quotation.status == 'won').group_by(Customer.id).all()
        data = [{'Customer': r[1], 'Total Value': r[2]} for r in results]
    else:
        data = []
    
    df = pd.DataFrame(data)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Report', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'{report_type}_report_{datetime.now().strftime("%Y%m%d")}.xlsx'
    )
