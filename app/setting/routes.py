from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required
from app.setting import setting_bp
from app.models import db, SystemSetting, Currency, Allowance
from app.forms import CurrencyForm, SystemSettingForm, AllowanceForm
from app.utils import generate_template_code


@setting_bp.route('/')
@login_required
def index():
    """Settings index."""
    return render_template('setting/index.html')


@setting_bp.route('/currencies')
@login_required
def currencies():
    """Currency settings."""
    currencies = Currency.query.order_by(Currency.currency_code).all()
    return render_template('setting/currencies.html', currencies=currencies)


@setting_bp.route('/currencies/new', methods=['GET', 'POST'])
@login_required
def new_currency():
    """Add new currency."""
    form = CurrencyForm()
    
    if form.validate_on_submit():
        currency = Currency(
            currency_code=form.currency_code.data,
            currency_name=form.currency_name.data,
            symbol=form.symbol.data,
            exchange_rate=form.exchange_rate.data,
            is_base=form.is_base.data,
            is_active=form.is_active.data
        )
        
        if form.is_base.data:
            Currency.query.update({'is_base': False})
        
        db.session.add(currency)
        db.session.commit()
        
        flash(f'Currency "{currency.currency_name}" added.', 'success')
        return redirect(url_for('setting.currencies'))
    
    return render_template('setting/currency_form.html', form=form)


@setting_bp.route('/allowances')
@login_required
def allowances():
    """Material allowances settings."""
    allowances = Allowance.query.order_by(Allowance.allowance_name).all()
    return render_template('setting/allowances.html', allowances=allowances)


@setting_bp.route('/allowances/new', methods=['GET', 'POST'])
@login_required
def new_allowance():
    """Add new allowance."""
    form = AllowanceForm()
    
    if form.validate_on_submit():
        allowance = Allowance(
            allowance_code=form.allowance_code.data,
            allowance_name=form.allowance_name.data,
            allowance_type=form.allowance_type.data,
            value=form.value.data,
            is_active=form.is_active.data
        )
        
        db.session.add(allowance)
        db.session.commit()
        
        flash(f'Allowance "{allowance.allowance_name}" added.', 'success')
        return redirect(url_for('setting.allowances'))
    
    return render_template('setting/allowance_form.html', form=form)


@setting_bp.route('/allowances/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_allowance(id):
    """Edit allowance."""
    allowance = Allowance.query.get_or_404(id)
    form = AllowanceForm(obj=allowance)
    
    if form.validate_on_submit():
        allowance.allowance_name = form.allowance_name.data
        allowance.allowance_type = form.allowance_type.data
        allowance.value = form.value.data
        allowance.is_active = form.is_active.data
        
        db.session.commit()
        flash('Allowance updated.', 'success')
        return redirect(url_for('setting.allowances'))
    
    return render_template('setting/allowance_form.html', form=form)


@setting_bp.route('/general')
@login_required
def general():
    """General settings."""
    if request.method == 'POST':
        # Update settings
        settings_to_update = [
            'COMPANY_NAME', 'COMPANY_ADDRESS', 'COMPANY_PHONE', 'COMPANY_EMAIL', 'COMPANY_GST',
            'DEFAULT_CURRENCY', 'DEFAULT_OVERHEAD_PERCENT', 'DEFAULT_PROFIT_PERCENT', 'DEFAULT_SCRAP_PERCENT'
        ]
        
        for key in settings_to_update:
            value = request.form.get(key.lower())
            setting = SystemSetting.query.filter_by(setting_key=key).first()
            
            if setting:
                setting.setting_value = value
            else:
                setting = SystemSetting(setting_key=key, setting_value=value)
                db.session.add(setting)
        
        db.session.commit()
        flash('Settings saved.', 'success')
        return redirect(url_for('setting.general'))
    
    # Load current settings
    settings = {s.setting_key: s.setting_value for s in SystemSetting.query.all()}
    
    return render_template('setting/general.html', settings=settings)
