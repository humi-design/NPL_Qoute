from flask import render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime
from app.auth import auth_bp
from app.models import db, User
from app.forms import LoginForm, UserForm, ChangePasswordForm
from app.extensions import bcrypt


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact administrator.', 'danger')
                return render_template('auth/login.html', form=form)
            
            login_user(user, remember=form.remember.data)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.dashboard'))
        
        flash('Invalid username or password.', 'danger')
    
    return render_template('auth/login.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page."""
    form = ChangePasswordForm()
    
    if form.validate_on_submit():
        if not bcrypt.check_password_hash(current_user.password_hash, form.current_password.data):
            flash('Current password is incorrect.', 'danger')
            return render_template('auth/profile.html', form=form)
        
        if form.new_password.data != form.confirm_password.data:
            flash('New passwords do not match.', 'danger')
            return render_template('auth/profile.html', form=form)
        
        current_user.password_hash = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
        db.session.commit()
        flash('Password changed successfully.', 'success')
        return redirect(url_for('auth.profile'))
    
    return render_template('auth/profile.html', form=form)


@auth_bp.route('/users')
@login_required
def users():
    """User list page."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('auth/users.html', users=users)


@auth_bp.route('/users/new', methods=['GET', 'POST'])
@login_required
def new_user():
    """Create new user."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    form = UserForm()
    
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists.', 'danger')
            return render_template('auth/user_form.html', form=form, user=None)
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already exists.', 'danger')
            return render_template('auth/user_form.html', form=form, user=None)
        
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data,
            role=form.role.data,
            is_active=form.is_active.data
        )
        
        if form.password.data:
            user.password_hash = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        else:
            user.password_hash = bcrypt.generate_password_hash('password123').decode('utf-8')
        
        db.session.add(user)
        db.session.commit()
        
        flash(f'User "{user.username}" created successfully.', 'success')
        return redirect(url_for('auth.users'))
    
    return render_template('auth/user_form.html', form=form, user=None)


@auth_bp.route('/users/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(id):
    """Edit user."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    user = User.query.get_or_404(id)
    form = UserForm(obj=user)
    form.edit_id = id  # For validation
    
    if form.validate_on_submit():
        # Check username uniqueness
        existing = User.query.filter_by(username=form.username.data).first()
        if existing and existing.id != id:
            flash('Username already exists.', 'danger')
            return render_template('auth/user_form.html', form=form, user=user)
        
        # Check email uniqueness
        existing = User.query.filter_by(email=form.email.data).first()
        if existing and existing.id != id:
            flash('Email already exists.', 'danger')
            return render_template('auth/user_form.html', form=form, user=user)
        
        user.username = form.username.data
        user.email = form.email.data
        user.full_name = form.full_name.data
        user.role = form.role.data
        user.is_active = form.is_active.data
        
        if form.password.data:
            user.password_hash = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        
        db.session.commit()
        flash(f'User "{user.username}" updated successfully.', 'success')
        return redirect(url_for('auth.users'))
    
    return render_template('auth/user_form.html', form=form, user=user)


@auth_bp.route('/users/<int:id>/delete', methods=['POST'])
@login_required
def delete_user(id):
    """Delete user."""
    if not current_user.has_permission('admin'):
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    user = User.query.get_or_404(id)
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('auth.users'))
    
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User "{user.username}" deleted.', 'success')
    return redirect(url_for('auth.users'))
