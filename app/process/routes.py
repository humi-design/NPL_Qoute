from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required
from app.process import process_bp
from app.models import db, Process, ProcessTemplate, ProcessTemplateStep, Machine, Vendor
from app.forms import ProcessForm, ProcessTemplateForm
from app.utils import generate_process_code, generate_template_code


@process_bp.route('/')
@login_required
def index():
    """Process list page."""
    processes = Process.query.order_by(Process.process_name).all()
    templates = ProcessTemplate.query.order_by(ProcessTemplate.template_name).all()
    return render_template('process/index.html', processes=processes, templates=templates)


@process_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new():
    """Create new process."""
    form = ProcessForm()
    form.process_code.data = generate_process_code()
    
    if form.validate_on_submit():
        process = Process(
            process_code=form.process_code.data,
            process_name=form.process_name.data,
            process_type=form.process_type.data,
            description=form.description.data,
            machine_id=form.machine_id.data.id if form.machine_id.data else None,
            vendor_id=form.vendor_id.data.id if form.vendor_id.data else None,
            department=form.department.data,
            setup_time=form.setup_time.data,
            cycle_time=form.cycle_time.data,
            cost_type=form.cost_type.data,
            cost_value=form.cost_value.data,
            cost_formula=form.cost_formula.data,
            is_active=form.is_active.data,
            is_custom=form.is_custom.data
        )
        
        db.session.add(process)
        db.session.commit()
        
        flash(f'Process "{process.process_name}" created.', 'success')
        return redirect(url_for('process.index'))
    
    return render_template('process/form.html', form=form, process=None)


@process_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    """Edit process."""
    process = Process.query.get_or_404(id)
    form = ProcessForm(obj=process)
    
    if form.validate_on_submit():
        process.process_name = form.process_name.data
        process.process_type = form.process_type.data
        process.description = form.description.data
        process.machine_id = form.machine_id.data.id if form.machine_id.data else None
        process.vendor_id = form.vendor_id.data.id if form.vendor_id.data else None
        process.department = form.department.data
        process.setup_time = form.setup_time.data
        process.cycle_time = form.cycle_time.data
        process.cost_type = form.cost_type.data
        process.cost_value = form.cost_value.data
        process.cost_formula = form.cost_formula.data
        process.is_active = form.is_active.data
        process.is_custom = form.is_custom.data
        
        db.session.commit()
        flash(f'Process "{process.process_name}" updated.', 'success')
        return redirect(url_for('process.index'))
    
    return render_template('process/form.html', form=form, process=process)


@process_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    """Delete process."""
    process = Process.query.get_or_404(id)
    db.session.delete(process)
    db.session.commit()
    flash('Process deleted.', 'success')
    return redirect(url_for('process.index'))


# ============ Process Templates ============

@process_bp.route('/templates/new', methods=['GET', 'POST'])
@login_required
def new_template():
    """Create new process template."""
    form = ProcessTemplateForm()
    form.template_code.data = generate_template_code()
    
    if form.validate_on_submit():
        template = ProcessTemplate(
            template_code=form.template_code.data,
            template_name=form.template_name.data,
            description=form.description.data,
            is_active=form.is_active.data
        )
        
        db.session.add(template)
        db.session.commit()
        
        flash(f'Template "{template.template_name}" created.', 'success')
        return redirect(url_for('process.index'))
    
    return render_template('process/template_form.html', form=form, template=None)


@process_bp.route('/templates/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_template(id):
    """Edit process template."""
    template = ProcessTemplate.query.get_or_404(id)
    form = ProcessTemplateForm(obj=template)
    
    if form.validate_on_submit():
        template.template_name = form.template_name.data
        template.description = form.description.data
        template.is_active = form.is_active.data
        db.session.commit()
        
        flash(f'Template "{template.template_name}" updated.', 'success')
        return redirect(url_for('process.index'))
    
    return render_template('process/template_form.html', form=form, template=template)


@process_bp.route('/templates/<int:id>/step/add', methods=['POST'])
@login_required
def add_step(id):
    """Add step to template."""
    template = ProcessTemplate.query.get_or_404(id)
    
    process_id = request.form.get('process_id', type=int)
    sequence = request.form.get('sequence', type=int)
    
    step = ProcessTemplateStep(
        template_id=template.id,
        process_id=process_id,
        sequence=sequence or template.steps.count() + 1
    )
    
    db.session.add(step)
    db.session.commit()
    
    flash('Step added.', 'success')
    return redirect(url_for('process.index'))


@process_bp.route('/templates/<int:id>/step/<int:step_id>/delete', methods=['POST'])
@login_required
def delete_step(id, step_id):
    """Delete template step."""
    step = ProcessTemplateStep.query.get_or_404(step_id)
    db.session.delete(step)
    db.session.commit()
    flash('Step deleted.', 'success')
    return redirect(url_for('process.index'))
