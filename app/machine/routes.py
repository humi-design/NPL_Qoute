from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required
from app.machine import machine_bp
from app.models import db, Machine
from app.forms import MachineForm
from app.utils import generate_machine_code


@machine_bp.route('/')
@login_required
def index():
    """Machine list page."""
    machines = Machine.query.order_by(Machine.machine_name).all()
    return render_template('machine/index.html', machines=machines)


@machine_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new machine."""
    form = MachineForm()
    form.machine_code.data = generate_machine_code()
    
    if form.validate_on_submit():
        machine = Machine(
            machine_code=form.machine_code.data,
            machine_name=form.machine_name.data,
            machine_type=form.machine_type.data,
            hourly_rate=form.hourly_rate.data,
            operator_name=form.operator_name.data,
            power_consumption=form.power_consumption.data,
            efficiency=form.efficiency.data,
            department=form.department.data,
            default_setup_time=form.default_setup_time.data,
            notes=form.notes.data,
            is_active=form.is_active.data
        )
        
        db.session.add(machine)
        db.session.commit()
        
        flash(f'Machine "{machine.machine_name}" created.', 'success')
        return redirect(url_for('machine.view', id=machine.id))
    
    return render_template('machine/form.html', form=form, machine=None)


@machine_bp.route('/<int:id>')
@login_required
def view(id):
    """View machine details."""
    machine = Machine.query.get_or_404(id)
    return render_template('machine/view.html', machine=machine)


@machine_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit machine."""
    machine = Machine.query.get_or_404(id)
    form = MachineForm(obj=machine)
    
    if form.validate_on_submit():
        machine.machine_name = form.machine_name.data
        machine.machine_type = form.machine_type.data
        machine.hourly_rate = form.hourly_rate.data
        machine.operator_name = form.operator_name.data
        machine.power_consumption = form.power_consumption.data
        machine.efficiency = form.efficiency.data
        machine.department = form.department.data
        machine.default_setup_time = form.default_setup_time.data
        machine.notes = form.notes.data
        machine.is_active = form.is_active.data
        
        db.session.commit()
        flash(f'Machine "{machine.machine_name}" updated.', 'success')
        return redirect(url_for('machine.view', id=id))
    
    return render_template('machine/form.html', form=form, machine=machine)


@machine_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete machine."""
    machine = Machine.query.get_or_404(id)
    db.session.delete(machine)
    db.session.commit()
    flash('Machine deleted.', 'success')
    return redirect(url_for('machine.index'))
