"""
NPL Fasteners ERP - Database Seed Script
Run this script to populate the database with initial data
"""

import os
import sys
from datetime import datetime, date

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Customer, Product, Material, MaterialRate, Machine, Vendor, 
    VendorRate, Process, ProcessTemplate, ProcessTemplateStep, Currency,
    Allowance, SystemSetting
)


def seed_database():
    """Seed the database with initial data."""
    app = create_app('development')
    
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if already seeded
        if User.query.first():
            print("Database already seeded. Skipping...")
            return
        
        print("Seeding database...")
        
        # ============ Users ============
        admin = User(
            username='admin',
            email='admin@nplfasteners.com',
            full_name='System Administrator',
            role='admin',
            password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
            is_active=True
        )
        sales = User(
            username='sales',
            email='sales@nplfasteners.com',
            full_name='Sales Manager',
            role='sales',
            password_hash=bcrypt.generate_password_hash('sales123').decode('utf-8'),
            is_active=True
        )
        production = User(
            username='production',
            email='production@nplfasteners.com',
            full_name='Production Manager',
            role='production',
            password_hash=bcrypt.generate_password_hash('prod123').decode('utf-8'),
            is_active=True
        )
        db.session.add_all([admin, sales, production])
        
        # ============ Materials ============
        materials_data = [
            ('MAT-0001', 'SS304', 'Stainless Steel 304', 8000, 200, '₹/kg', 'AISI 304'),
            ('MAT-0002', 'SS316', 'Stainless Steel 316', 8020, 300, '₹/kg', 'AISI 316'),
            ('MAT-0003', 'SS303', 'Stainless Steel 303', 8010, 250, '₹/kg', 'AISI 303'),
            ('MAT-0004', 'EN8', 'Carbon Steel EN8', 7850, 85, '₹/kg', '080M40'),
            ('MAT-0005', 'EN1A', 'Free Cutting Steel EN1A', 7870, 90, '₹/kg', '230M07'),
            ('MAT-0006', 'BRASS', 'Brass CW614N', 8530, 550, '₹/kg', 'CW614N'),
            ('MAT-0007', 'ALU', 'Aluminium 6061', 2700, 280, '₹/kg', '6061-T6'),
            ('MAT-0008', 'COPPER', 'Copper C101', 8960, 700, '₹/kg', 'C10100'),
            ('MAT-0009', 'TITAN', 'Titanium Grade 2', 4510, 1500, '₹/kg', 'ASTM B348'),
            ('MAT-0010', 'MS', 'Mild Steel MS', 7870, 70, '₹/kg', 'SAE 1018'),
        ]
        
        for code, short, name, density, rate, unit, grade in materials_data:
            material = Material(
                material_code=code,
                material_name=name,
                density=density,
                grade=grade,
                hsn_code='7228',
                default_rate=rate,
                rate_unit=unit,
                is_active=True
            )
            db.session.add(material)
            db.session.flush()
            
            # Add rate history
            rate_record = MaterialRate(
                material_id=material.id,
                rate=rate,
                unit=unit,
                effective_date=date.today()
            )
            db.session.add(rate_record)
        
        # ============ Machines ============
        machines_data = [
            ('MCH-001', 'Traub TNL32', 'CNC Lathe', 350, 'Production'),
            ('MCH-002', 'Traub TNL18', 'CNC Lathe', 350, 'Production'),
            ('MCH-003', 'VMC 750', 'Vertical Machining Center', 450, 'Milling'),
            ('MCH-004', 'VMC 850', 'Vertical Machining Center', 450, 'Milling'),
            ('MCH-005', 'Thread Rolling', 'Thread Rolling Machine', 300, 'Threading'),
            ('MCH-006', 'Press 25T', 'Power Press', 200, 'Pressing'),
            ('MCH-007', 'Grinding', 'Cylindrical Grinder', 350, 'Grinding'),
            ('MCH-008', 'Heat Treatment', 'Heat Treatment Furnace', 250, 'Heat Treatment'),
        ]
        
        for code, name, mtype, rate, dept in machines_data:
            machine = Machine(
                machine_code=code,
                machine_name=name,
                machine_type=mtype,
                hourly_rate=rate,
                department=dept,
                is_active=True
            )
            db.session.add(machine)
        
        # ============ Vendors ============
        vendors_data = [
            ('VND-0001', 'Sun Metal Alloys', 'Heat Treatment Services', '+91-9876543210', 'INR'),
            ('VND-0002', 'Precision Plating', 'Electroplating Services', '+91-9876543211', 'INR'),
            ('VND-0003', 'Quality Fasteners Steel', 'Raw Material Supplier', '+91-9876543212', 'INR'),
            ('VND-0004', 'Pack Pro Industries', 'Packaging Materials', '+91-9876543213', 'INR'),
            ('VND-0005', 'Logistics Express', 'Transportation', '+91-9876543214', 'INR'),
        ]
        
        for code, name, service, phone, currency in vendors_data:
            vendor = Vendor(
                vendor_code=code,
                vendor_name=name,
                contact_person=service,
                phone=phone,
                default_currency=currency,
                is_active=True
            )
            db.session.add(vendor)
            db.session.flush()
            
            # Add vendor rate
            vendor_rate = VendorRate(
                vendor_id=vendor.id,
                service_name='General Services',
                rate=100,
                unit='₹/piece',
                effective_date=date.today()
            )
            db.session.add(vendor_rate)
        
        # ============ Processes ============
        processes_data = [
            ('PRC-0001', 'Raw Material', 'material', None, None, 'percentage', 100, 'Raw material cost'),
            ('PRC-0002', 'Traub Operation', 'internal', 'MCH-001', 10, 'cycle_time', 0, 'CNC Lathe operation'),
            ('PRC-0003', 'Thread Rolling', 'internal', 'MCH-005', 5, 'cycle_time', 0, 'Thread rolling'),
            ('PRC-0004', 'Milling/Slotting', 'internal', 'MCH-003', 8, 'cycle_time', 0, 'Milling operation'),
            ('PRC-0005', 'Drilling', 'internal', 'MCH-001', 3, 'cycle_time', 0, 'Drilling holes'),
            ('PRC-0006', 'Reaming', 'internal', 'MCH-001', 2, 'cycle_time', 0, 'Reaming operation'),
            ('PRC-0007', 'Chamfering', 'internal', 'MCH-001', 1, 'cycle_time', 0, 'Edge chamfering'),
            ('PRC-0008', 'Heat Treatment', 'vendor', None, 'VND-0001', 'per_piece', 2.5, 'Heat treatment'),
            ('PRC-0009', 'Plating', 'vendor', None, 'VND-0002', 'per_piece', 1.5, 'Electroplating'),
            ('PRC-0010', 'Passivation', 'vendor', None, 'VND-0002', 'per_piece', 0.5, 'SS Passivation'),
            ('PRC-0011', 'Inspection', 'internal', None, None, 'per_piece', 0.5, 'Quality inspection'),
            ('PRC-0012', 'Packing', 'internal', None, None, 'per_piece', 0.25, 'Packaging'),
            ('PRC-0013', 'Marking', 'internal', 'MCH-001', 1, 'cycle_time', 0, 'Part marking'),
            ('PRC-0014', 'Ultrasonic Cleaning', 'internal', None, None, 'per_piece', 0.3, 'Cleaning'),
            ('PRC-0015', 'Centerless Grinding', 'internal', 'MCH-007', 5, 'cycle_time', 0, 'Precision grinding'),
        ]
        
        for code, name, ptype, machine_id, vendor_id, cost_type, cost_val, desc in processes_data:
            # Map machine code to ID
            machine = None
            if machine_id:
                machine = Machine.query.filter_by(machine_code=machine_id).first()
            
            # Map vendor code to ID
            vendor = None
            if vendor_id:
                vendor = Vendor.query.filter_by(vendor_code=vendor_id).first()
            
            process = Process(
                process_code=code,
                process_name=name,
                process_type=ptype,
                machine_id=machine.id if machine else None,
                vendor_id=vendor.id if vendor else None,
                cost_type=cost_type,
                cost_value=cost_val,
                description=desc,
                is_active=True
            )
            db.session.add(process)
        
        # ============ Process Templates ============
        # Template 1: Hex Bolt
        hex_bolt = ProcessTemplate(
            template_code='TMP-0001',
            template_name='Hex Bolt Standard',
            description='Standard manufacturing route for hex bolts'
        )
        db.session.add(hex_bolt)
        db.session.flush()
        
        hex_steps = [
            ('Raw Material', 1),
            ('Traub Operation', 2),
            ('Thread Rolling', 3),
            ('Chamfering', 4),
            ('Heat Treatment', 5),
            ('Plating', 6),
            ('Passivation', 7),
            ('Inspection', 8),
            ('Packing', 9),
        ]
        
        for step_name, seq in hex_steps:
            process = Process.query.filter_by(process_name=step_name).first()
            if process:
                template_step = ProcessTemplateStep(
                    template_id=hex_bolt.id,
                    process_id=process.id,
                    sequence=seq
                )
                db.session.add(template_step)
        
        # Template 2: Precision Part
        precision = ProcessTemplate(
            template_code='TMP-0002',
            template_name='Precision Part',
            description='High precision part with grinding'
        )
        db.session.add(precision)
        db.session.flush()
        
        precision_steps = [
            ('Raw Material', 1),
            ('Traub Operation', 2),
            ('Milling/Slotting', 3),
            ('Drilling', 4),
            ('Heat Treatment', 5),
            ('Centerless Grinding', 6),
            ('Plating', 7),
            ('Inspection', 8),
            ('Packing', 9),
        ]
        
        for step_name, seq in precision_steps:
            process = Process.query.filter_by(process_name=step_name).first()
            if process:
                template_step = ProcessTemplateStep(
                    template_id=precision.id,
                    process_id=process.id,
                    sequence=seq
                )
                db.session.add(template_step)
        
        # ============ Currencies ============
        currencies_data = [
            ('INR', 'Indian Rupee', '₹', 1.0, True),
            ('USD', 'US Dollar', '$', 83.0, False),
            ('EUR', 'Euro', '€', 90.0, False),
            ('GBP', 'British Pound', '£', 100.0, False),
            ('AED', 'UAE Dirham', 'د.إ', 22.5, False),
        ]
        
        for code, name, symbol, rate, is_base in currencies_data:
            currency = Currency(
                currency_code=code,
                currency_name=name,
                symbol=symbol,
                exchange_rate=rate,
                is_base=is_base,
                is_active=True
            )
            db.session.add(currency)
        
        # ============ Allowances ============
        allowances_data = [
            ('ALL-0001', 'Parting', 'machining', 3.0),
            ('ALL-0002', 'Facing', 'machining', 2.0),
            ('ALL-0003', 'Machining Stock', 'machining', 1.0),
            ('ALL-0004', 'Grinding Stock', 'grinding', 0.5),
            ('ALL-0005', 'Chamfer', 'facing', 0.5),
            ('ALL-0006', 'Polishing', 'custom', 0.3),
        ]
        
        for code, name, atype, val in allowances_data:
            allowance = Allowance(
                allowance_code=code,
                allowance_name=name,
                allowance_type=atype,
                value=val,
                is_active=True
            )
            db.session.add(allowance)
        
        # ============ System Settings ============
        settings_data = [
            ('COMPANY_NAME', 'NPL Fasteners', 'Company name'),
            ('COMPANY_ADDRESS', '123 Industrial Area, Mumbai 400001', 'Company address'),
            ('COMPANY_PHONE', '+91-22-12345678', 'Company phone'),
            ('COMPANY_EMAIL', 'sales@nplfasteners.com', 'Company email'),
            ('COMPANY_GST', '27AAACH1234C1ZB', 'Company GST number'),
            ('DEFAULT_CURRENCY', 'INR', 'Default currency'),
            ('DEFAULT_OVERHEAD_PERCENT', '15', 'Default overhead %'),
            ('DEFAULT_PROFIT_PERCENT', '20', 'Default profit %'),
            ('DEFAULT_SCRAP_PERCENT', '2', 'Default scrap %'),
            ('USD_RATE', '83.0', 'USD exchange rate'),
            ('EUR_RATE', '90.0', 'EUR exchange rate'),
            ('GBP_RATE', '100.0', 'GBP exchange rate'),
        ]
        
        for key, value, desc in settings_data:
            setting = SystemSetting(
                setting_key=key,
                setting_value=value,
                description=desc
            )
            db.session.add(setting)
        
        # Commit all
        db.session.commit()
        
        print("Database seeded successfully!")
        print("\nDefault login credentials:")
        print("  Admin: admin / admin123")
        print("  Sales: sales / sales123")
        print("  Production: production / prod123")


if __name__ == '__main__':
    seed_database()
