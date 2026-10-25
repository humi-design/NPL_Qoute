"""
NPL Fasteners ERP - Database Seed Script
Run this script to populate the database with initial data
"""

import os
import sys
from datetime import datetime, date, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.extensions import create_app, db, bcrypt
from app.models import (
    User, Customer, Product, Material, MaterialRate, Machine, Vendor, 
    VendorRate, Process, ProcessTemplate, ProcessTemplateStep, Currency,
    Allowance, SystemSetting, Supplier, SupplierType, SupplierMaterial,
    SupplierPurchaseOrder, SupplierPurchaseOrderLine, SupplierOrderReceipt,
    SupplierOrderReceiptLine, SupplierInvoice, SupplierPayment,
    VendorInvoice, VendorPayment,
)
from app.utils import (generate_po_number, generate_receipt_number,
                       generate_payment_number)


def _seed_procurement():
    """Add illustrative procurement/payables records for the first supplier and vendor.

    Non-fatal: if the master data needed is missing, it simply does nothing.
    """
    supplier = Supplier.query.order_by(Supplier.id).first()
    vendor = Vendor.query.order_by(Vendor.id).first()
    if supplier is None:
        return

    material = None
    if supplier.supplied_materials:
        first_link = supplier.supplied_materials.first()
        material = first_link.material if first_link else None
    if material is None:
        material = Material.query.order_by(Material.id).first()

    po = SupplierPurchaseOrder(
        po_number=generate_po_number(),
        supplier_id=supplier.id,
        order_date=date.today() - timedelta(days=20),
        expected_date=date.today() - timedelta(days=5),
        status='partially_received',
        payment_terms=supplier.payment_terms or 'Net 30',
        notes='Seeded demo purchase order',
    )
    db.session.add(po)
    db.session.flush()

    line = SupplierPurchaseOrderLine(
        po_id=po.id, material_id=material.id if material else None,
        description=material.material_name if material else 'Raw material',
        grade='Demo grade', quantity_ordered=1000, unit='kg', rate=210,
        rate_unit='INR/kg',
    )
    db.session.add(line)
    db.session.flush()

    receipt = SupplierOrderReceipt(
        receipt_number=generate_receipt_number(), po_id=po.id,
        supplier_id=supplier.id, receipt_date=date.today() - timedelta(days=6),
        supplier_document='DC-1001', notes='Seeded demo goods receipt',
    )
    db.session.add(receipt)
    db.session.flush()
    db.session.add(SupplierOrderReceiptLine(
        receipt_id=receipt.id, po_line_id=line.id,
        material_id=material.id if material else None,
        quantity_received=600, quantity_rejected=20, batch_number='B-2026-001',
        heat_number='HT-2026-001', lot_number='LOT-001',
        certificate_ref='MC-2026-001', inspection_status='passed',
    ))

    invoice = SupplierInvoice(
        invoice_number='SINV-DEMO-001', supplier_id=supplier.id, po_id=po.id,
        invoice_date=date.today() - timedelta(days=4),
        due_date=date.today() + timedelta(days=26),
        subtotal=124950, tax_amount=22491, total_amount=147441, status='posted',
        notes='Seeded demo supplier invoice',
    )
    db.session.add(invoice)
    db.session.flush()

    db.session.add(SupplierPayment(
        payment_number=generate_payment_number(), supplier_id=supplier.id,
        payment_date=date.today() - timedelta(days=2), amount=50000,
        method='bank', reference='NEFT-DEMO-001',
        notes='Seeded demo supplier payment',
    ))

    if vendor is not None:
        vinvoice = VendorInvoice(
            invoice_number='VINV-DEMO-001', vendor_id=vendor.id,
            invoice_date=date.today() - timedelta(days=3),
            due_date=date.today() + timedelta(days=27),
            subtotal=9000, tax_amount=1620, total_amount=10620, status='posted',
            notes='Seeded demo job-work invoice',
        )
        db.session.add(vinvoice)
        db.session.flush()
        db.session.add(VendorPayment(
            payment_number=generate_payment_number(model=VendorPayment, prefix='VPAY'),
            vendor_id=vendor.id, payment_date=date.today() - timedelta(days=1),
            amount=4000, method='bank', notes='Seeded demo job-work payment',
        ))

def _seed_orders():
    """Add illustrative orders + tracking records (route, lots, inspections, packaging).

    Non-fatal: needs a customer, material and (for the vendor step) a vendor.
    Creates a demo customer when none exists so a fresh database still shows
    the Live Operations screens with real data.
    """
    from app.models import (Order, OrderOperation, OperationTransaction,
                            MaterialLot, MaterialMovement, Inspection,
                            PackagingRecord)
    from app.utils import (generate_order_number, generate_inspection_number,
                           generate_lot_number)

    material = Material.query.order_by(Material.id).first()
    vendor = Vendor.query.order_by(Vendor.id).first()
    if material is None:
        return

    customer = Customer.query.order_by(Customer.id).first()
    if customer is None:
        customer = Customer(customer_code='CUST-0001',
                            company_name='Demo Customer Pvt Ltd',
                            currency='INR', payment_terms='Net 30')
        db.session.add(customer)
        db.session.flush()

    today = date.today()

    def make_order(qty, status, due_off, priority, part, drawing):
        order = Order(
            order_number=generate_order_number(),
            customer_id=customer.id,
            part_description=part, drawing_number=drawing, revision='A',
            material_grade=material.grade or 'SS304',
            quantity_ordered=qty, unit='pcs', rate=12.5, currency='INR',
            order_date=today - timedelta(days=15),
            due_date=today + timedelta(days=due_off),
            planned_start=today - timedelta(days=12),
            actual_start=today - timedelta(days=10),
            status=status, priority=priority,
            notes='Seeded demo order',
        )
        db.session.add(order)
        db.session.flush()
        return order

    # --- Order 1: in production, one in-house step done, one vendor step open ---
    o1 = make_order(500, 'in_production', 10, 'high', 'Hex Bolt M10x40', 'DWG-1001')
    op1 = OrderOperation(order_id=o1.id, sequence=1, operation_name='Cold Forging',
                         quantity_planned=500, status='completed',
                         actual_start=today - timedelta(days=9),
                         actual_end=today - timedelta(days=6))
    op2 = OrderOperation(order_id=o1.id, sequence=2, operation_name='Zinc Plating',
                         quantity_planned=500, is_vendor=True,
                         vendor_id=vendor.id if vendor else None,
                         status='in_progress', is_final=True,
                         planned_issue_date=today - timedelta(days=5),
                         due_date=today + timedelta(days=3))
    db.session.add_all([op1, op2])
    db.session.flush()
    db.session.add_all([
        OperationTransaction(order_operation_id=op1.id, transaction_type='issue',
                             transaction_date=today - timedelta(days=9),
                             quantity=520, reference='DC-1001'),
        OperationTransaction(order_operation_id=op1.id, transaction_type='return',
                             transaction_date=today - timedelta(days=6),
                             quantity=520, reference='RET-1001'),
        OperationTransaction(order_operation_id=op1.id, transaction_type='accept',
                             transaction_date=today - timedelta(days=6),
                             quantity=500, reference='ACC-1001'),
        OperationTransaction(order_operation_id=op1.id, transaction_type='reject',
                             transaction_date=today - timedelta(days=6),
                             quantity=15, reference='REJ-1001'),
        OperationTransaction(order_operation_id=op1.id, transaction_type='rework',
                             transaction_date=today - timedelta(days=6),
                             quantity=5, reference='RWK-1001'),
        OperationTransaction(order_operation_id=op2.id, transaction_type='issue',
                             transaction_date=today - timedelta(days=5),
                             quantity=500, reference='DC-1002'),
    ])

    lot1 = MaterialLot(lot_number=generate_lot_number(), material_id=material.id,
                       order_id=o1.id, supplier_id=(Supplier.query.first().id
                                                    if Supplier.query.first() else None),
                       grade=material.grade or 'SS304', heat_number='HT-2026-001',
                       batch_number='B-2026-001', certificate_ref='MC-2026-001',
                       quantity_received=1200, unit='kg',
                       receipt_date=today - timedelta(days=14))
    db.session.add(lot1)
    db.session.flush()
    db.session.add_all([
        MaterialMovement(lot_id=lot1.id, movement_type='receipt',
                         movement_date=today - timedelta(days=14), quantity=1200,
                         reference=lot1.lot_number),
        MaterialMovement(lot_id=lot1.id, movement_type='allocation',
                         movement_date=today - timedelta(days=12), quantity=400,
                         reference='ALLOC-1001', order_id=o1.id),
        MaterialMovement(lot_id=lot1.id, movement_type='issue',
                         movement_date=today - timedelta(days=10), quantity=350,
                         reference='ISS-1001', order_id=o1.id,
                         order_operation_id=op1.id),
    ])
    db.session.add(Inspection(
        inspection_number=generate_inspection_number(), order_id=o1.id,
        order_operation_id=op1.id, lot_id=lot1.id, inspection_type='incoming',
        inspection_date=today - timedelta(days=13), tolerance='±0.05 mm',
        acceptance_criteria='Material certificate matches grade; no surface defects.',
        measurements='Dia 10.02, 10.01, 10.03', sample_size=5,
        quantity_inspected=1200, quantity_accepted=1190, quantity_rejected=10,
        result='partial'))

    # --- Order 2: completed, fully packed and warehoused ---
    o2 = make_order(200, 'completed', -2, 'normal', 'Flange Nut M8', 'DWG-1002')
    op3 = OrderOperation(order_id=o2.id, sequence=1, operation_name='Machining',
                         quantity_planned=200, status='completed', is_final=True,
                         actual_start=today - timedelta(days=8),
                         actual_end=today - timedelta(days=4))
    db.session.add(op3)
    db.session.flush()
    db.session.add_all([
        OperationTransaction(order_operation_id=op3.id, transaction_type='issue',
                             transaction_date=today - timedelta(days=8),
                             quantity=200, reference='DC-2001'),
        OperationTransaction(order_operation_id=op3.id, transaction_type='accept',
                             transaction_date=today - timedelta(days=4),
                             quantity=200, reference='ACC-2001'),
    ])
    db.session.add(PackagingRecord(
        order_id=o2.id, packaging_date=today - timedelta(days=3),
        packaging_type='box', quantity_packed=200, quantity_accepted=200,
        warehouse_receipt_number='WR-2026-001',
        warehouse_receipt_date=today - timedelta(days=2), location='Rack A1',
        status='warehoused'))





def _seed_inventory_costing():
    """Add illustrative stock locations, a transfer, an adjustment, actual costs,
    and a supplier RFQ with comparable quotes.

    Non-fatal: skips gracefully when there is no material/order to hang the
    records off. Uses the existing order lots so the stock position and ledger
    screens are populated on a freshly seeded database.
    """
    from app.models import (StockLocation, StockTransfer, StockAdjustment,
                            OrderCostEntry, SupplierRFQ, SupplierQuote,
                            Order, MaterialLot)
    from app.utils import (generate_transfer_number, generate_adjustment_number,
                           generate_supplier_rfq_number)

    if StockLocation.query.first() is not None:
        return  # already seeded

    material = Material.query.order_by(Material.id).first()
    order = Order.query.order_by(Order.id).first()
    lot = MaterialLot.query.order_by(MaterialLot.id).first()
    vendor = Vendor.query.order_by(Vendor.id).first()
    supplier = Supplier.query.order_by(Supplier.id).first()
    if material is None:
        return

    today = date.today()

    raw = StockLocation(code='STORE-RM', name='Raw Material Store',
                        kind='raw_material')
    wip = StockLocation(code='WIP-01', name='Work in Progress Floor', kind='wip')
    fg = StockLocation(code='STORE-FG', name='Finished Goods Store',
                       kind='finished_goods')
    rej = StockLocation(code='STORE-REJ', name='Rejected Stock Cage',
                        kind='rejected')
    db.session.add_all([raw, wip, fg, rej])
    db.session.flush()
    if vendor is not None:
        db.session.add(StockLocation(code=f'VEND-{vendor.id}',
                                     name=f'{vendor.vendor_name} (subcontractor)',
                                     kind='subcontractor', vendor_id=vendor.id))
    db.session.flush()

    # A store -> WIP transfer and a scrap write-off, both traceable.
    db.session.add(StockTransfer(
        transfer_number=generate_transfer_number(),
        from_location_id=raw.id, to_location_id=wip.id,
        transfer_date=today - timedelta(days=3), quantity=350, unit='kg',
        material_lot_id=lot.id if lot else None,
        order_id=order.id if order else None,
        status='received', reference='DC-DEMO-001',
        notes='Seeded demo transfer to the shop floor'))
    db.session.add(StockAdjustment(
        adjustment_number=generate_adjustment_number(),
        adjustment_date=today - timedelta(days=1),
        material_lot_id=lot.id if lot else None,
        location_id=rej.id, order_id=order.id if order else None,
        adjustment_type='write_off', quantity=5, unit='kg',
        reason='Physical count difference (seeded demo)'))

    # Actual cost entries against the first order, so costing has data.
    if order is not None:
        db.session.add_all([
            OrderCostEntry(order_id=order.id, category='material', amount=18000,
                           currency='INR', amount_base=18000,
                           entry_date=today - timedelta(days=8),
                           description='Material issued (seeded)'),
            OrderCostEntry(order_id=order.id, category='subcontracting',
                           amount=4500, currency='INR', amount_base=4500,
                           entry_date=today - timedelta(days=3),
                           description='Vendor plating charges (seeded)',
                           vendor_id=vendor.id if vendor else None),
            OrderCostEntry(order_id=order.id, category='machining', amount=6200,
                           currency='INR', amount_base=6200,
                           entry_date=today - timedelta(days=6),
                           description='In-house machining (seeded)'),
        ])

    # A supplier RFQ with three comparable quotes.
    if supplier is not None:
        rfq = SupplierRFQ(
            rfq_number=generate_supplier_rfq_number(),
            title=f'{material.material_name} bar stock (seeded)',
            material_id=material.id, order_id=order.id if order else None,
            quantity=1000, unit='kg', grade=material.grade,
            rfq_date=today - timedelta(days=10),
            due_date=today - timedelta(days=4),
            required_date=today + timedelta(days=7),
            status='quoted', payment_terms='Net 30',
            delivery_terms='Delivered to store', currency='INR',
            notes='Seeded demo supplier RFQ')
        db.session.add(rfq)
        db.session.flush()

        quote_data = [
            (supplier, 'SQ-A', 210.0, 7, 'Net 30', 'Delivered to store'),
            (supplier, 'SQ-B', 205.0, 14, 'Net 45', 'Ex-works'),
            (supplier, 'SQ-C', 218.0, 4, 'Net 15', 'Delivered to store'),
        ]
        for sup, ref, price, lead, pay, delivery in quote_data:
            db.session.add(SupplierQuote(
                supplier_rfq_id=rfq.id, supplier_id=sup.id, quote_reference=ref,
                quote_date=today - timedelta(days=6), quantity=1000, unit='kg',
                unit_price=price, currency='INR', tax_percent=18,
                lead_time_days=lead, payment_terms=pay, delivery_terms=delivery,
                status='received'))


def _seed_documents_dispatch():
    """Add illustrative documents, a dispatch and a notification.

    Non-fatal: creates a controlled document against a product, a dispatch of
    already-warehoused finished goods with one line, and a notification so the
    Documents, Dispatch and Notifications screens are populated on a freshly
    seeded database.
    """
    from app.models import (Attachment, Dispatch, DispatchLine, Notification,
                            PackagingRecord, Product, User, Order)
    from app.utils import generate_dispatch_number
    from datetime import date, timedelta
    today = date.today()

    product = Product.query.order_by(Product.id).first()
    if product is not None and not Attachment.query.filter_by(
            entity_type='product', entity_id=product.id).first():
        db.session.add(Attachment(
            entity_type='product', entity_id=product.id,
            file_name='drawing_demo.pdf', original_name='drawing_demo.pdf',
            file_type='pdf', file_size=2048,
            file_path='uploads/demo/drawing_demo.pdf',
            title='Product drawing', category='drawing', revision='A',
            is_current=True, notes='Seeded demo document'))

    # Dispatch the finished goods that were warehoused by _seed_orders().
    packed = PackagingRecord.query.filter_by(status='warehoused').first()
    if packed is not None and not Dispatch.query.filter_by(
            order_id=packed.order_id).first():
        order = Order.query.get(packed.order_id)
        dispatch = Dispatch(
            dispatch_number=generate_dispatch_number(),
            order_id=packed.order_id,
            customer_id=order.customer_id if order else None,
            dispatch_date=today - timedelta(days=1),
            mode='surface', transporter='Blue Dart', vehicle_number='MH-12-AB-1234',
            shipment_reference='LR-2026-001', destination='Pune',
            status='dispatched', delivery_status='pending',
            remarks='Seeded demo dispatch')
        db.session.add(dispatch)
        db.session.flush()
        db.session.add(DispatchLine(
            dispatch_id=dispatch.id, packaging_record_id=packed.id,
            quantity=packed.quantity_accepted or 0, label_ref=packed.label_ref))
        packed.status = 'dispatched'

    admin = User.query.filter_by(username='admin').first()
    if admin is not None and Notification.query.filter_by(
            user_id=admin.id).count() == 0:
        db.session.add(Notification(
            user_id=admin.id, title='Welcome to NPL ERP',
            message='Seeded demo notification',
            notification_type='system'))


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
        
        # ============ Vendor Work Orders ============
        from datetime import timedelta
        from app.models import VendorWorkOrder, VendorQualityIssue

        today = date.today()
        work_orders_data = [
            # (number, vendor_code, process_name, service, ordered, received, rejected,
            #  rate, order_date_offset, due_date_offset, status, priority)
            ('WO-{}-0001'.format(today.year), 'VND-0001', 'Heat Treatment', 'Heat Treatment Batch', 500, 500, 5, 2.5, -20, -5, 'completed', 'normal'),
            ('WO-{}-0002'.format(today.year), 'VND-0002', 'Plating', 'Zinc Plating', 1000, 400, 0, 1.5, -10, 5, 'in_progress', 'high'),
            ('WO-{}-0003'.format(today.year), 'VND-0002', 'Passivation', 'SS Passivation', 750, 0, 0, 0.5, -5, -2, 'pending', 'urgent'),
            ('WO-{}-0004'.format(today.year), 'VND-0001', 'Heat Treatment', 'Case Hardening', 300, 0, 0, 3.0, -2, 10, 'pending', 'normal'),
        ]

        for (num, vcode, pname, service, ordered, received, rejected,
             rate, order_off, due_off, status, priority) in work_orders_data:
            vendor = Vendor.query.filter_by(vendor_code=vcode).first()
            process = Process.query.filter_by(process_name=pname).first()
            if not vendor:
                continue
            wo = VendorWorkOrder(
                work_order_number=num,
                vendor_id=vendor.id,
                process_id=process.id if process else None,
                title=service,
                service_name=service,
                quantity_ordered=ordered,
                quantity_received=received,
                quantity_rejected=rejected,
                unit='pcs',
                rate=rate,
                order_date=today + timedelta(days=order_off),
                due_date=today + timedelta(days=due_off),
                completed_date=(today + timedelta(days=due_off)) if status == 'completed' else None,
                status=status,
                priority=priority
            )
            db.session.add(wo)
            db.session.flush()

            # Add a quality issue for the completed heat treatment job
            if num.endswith('0001'):
                db.session.add(VendorQualityIssue(
                    vendor_id=vendor.id,
                    work_order_id=wo.id,
                    issue_date=today + timedelta(days=-6),
                    issue_type='rejection',
                    severity='minor',
                    description='5 pieces rejected due to surface hardness below specification.',
                    quantity_affected=5,
                    status='resolved',
                    resolution='Vendor re-processed the affected batch at no cost.',
                    resolved_date=today + timedelta(days=-3)
                ))

        # ============ Supplier Types & Suppliers (Master Data) ============
        supplier_types_data = [
            ('SUPT-001', 'Raw Material', 'Ferrous and non-ferrous raw material suppliers', 10),
            ('SUPT-002', 'Consumables', 'Cutting tools, coolant, packaging consumables', 20),
            ('SUPT-003', 'Hardware & Fasteners', 'Standard hardware and fastener components', 30),
            ('SUPT-004', 'Services', 'Job work and outsourced service providers', 40),
        ]
        supplier_type_objs = {}
        for code, name, desc, order in supplier_types_data:
            st = SupplierType(type_code=code, type_name=name, description=desc,
                              sort_order=order, is_active=True)
            db.session.add(st)
            db.session.flush()
            supplier_type_objs[code] = st

        # Map seed suppliers to materials by material_code for the link table
        material_by_code = {m.material_code: m for m in Material.query.all()}

        suppliers_data = [
            # code, company, type_code, contact, email, phone, city, state,
            # gst, pan, currency, terms, lead, [(material_code, grade, rate, preferred)]
            ('SUP-0001', 'Bharat Steel Traders', 'SUPT-001', 'Rakesh Sharma',
             'sales@bharatsteel.in', '+91-9822011223', 'Mumbai', 'Maharashtra',
             '27AABCB1234C1Z5', 'AABCB1234C', 'INR', 'Net 30', 7,
             [('MAT-0004', 'EN8', 85.0, True), ('MAT-0010', 'SAE 1018', 70.0, False)]),
            ('SUP-0002', 'Sterling Metals Pvt Ltd', 'SUPT-001', 'Meena Iyer',
             'orders@sterlingmetals.com', '+91-9845567788', 'Bengaluru', 'Karnataka',
             '29AACCS5678D1Z2', 'AACCS5678D', 'INR', 'Net 45', 10,
             [('MAT-0001', 'AISI 304', 210.0, True), ('MAT-0002', 'AISI 316', 305.0, False)]),
            ('SUP-0003', 'ToolMart Industrial Supplies', 'SUPT-002', 'Anil Desai',
             'info@toolmart.co.in', '+91-9730099001', 'Pune', 'Maharashtra',
             '27AAGCT9012E1Z9', 'AAGCT9012E', 'INR', 'Net 15', 4,
             [('MAT-0005', '230M07', 92.0, True)]),
            ('SUP-0004', 'Global Brass Alloys', 'SUPT-001', 'Fatima Sheikh',
             'export@globalbrass.ae', '+971-501234567', 'Dubai', 'Dubai',
             None, None, 'AED', 'Net 60', 21,
             [('MAT-0006', 'CW614N', 24.5, True)]),
            ('SUP-0005', 'Pinnacle Surface Tech', 'SUPT-004', 'Suresh Nair',
             'service@pinnaclesurface.in', '+91-9445566778', 'Chennai', 'Tamil Nadu',
             '33AAECP3456F1Z1', 'AAECP3456F', 'INR', 'Net 30', 12,
             [('MAT-0003', 'AISI 303', 258.0, False)]),
        ]

        for (code, company, tcode, contact, email, phone, city, state,
             gst, pan, currency, terms, lead, mats) in suppliers_data:
            supplier = Supplier(
                supplier_code=code,
                company_name=company,
                supplier_type_id=supplier_type_objs[tcode].id,
                contact_person=contact,
                email=email,
                phone=phone,
                city=city,
                state=state,
                country='UAE' if currency == 'AED' else 'India',
                gst_vat=gst,
                pan=pan,
                default_currency=currency,
                payment_terms=terms,
                lead_time=lead,
                is_active=True,
            )
            db.session.add(supplier)
            db.session.flush()

            for mat_code, grade, rate, preferred in mats:
                material = material_by_code.get(mat_code)
                if material is None:
                    continue
                db.session.add(SupplierMaterial(
                    supplier_id=supplier.id,
                    material_id=material.id,
                    grade=grade,
                    rate=rate,
                    rate_unit='₹/kg',
                    lead_time=lead,
                    is_preferred=preferred,
                ))
                # Backfill the material's default supplier link where empty
                if material.supplier_id is None:
                    material.supplier_id = supplier.id
                    if not material.default_supplier:
                        material.default_supplier = company

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
        

        # ============ Procurement demo data (purchase orders, receipts, payables) ============
        # A single illustrative PO with a partial receipt (batch/heat/inspection),
        # plus one invoice and one part-paid payment, so the supplier and vendor
        # procurement panels show real records on a freshly seeded database.
        _seed_procurement()

        # ============ Orders & tracking demo data ============
        # Illustrative orders with a manufacturing route, material lots,
        # an inspection and packaging, so the Live Operations screens are
        # populated on a freshly seeded database.
        _seed_orders()

        # ============ Inventory, costing & supplier RFQ demo data ============
        # Stock locations, a transfer, an adjustment, actual cost entries and a
        # supplier RFQ with comparable quotes, so the new screens are populated.
        _seed_inventory_costing()

        # ============ Documents, dispatch & notifications demo data ============
        _seed_documents_dispatch()

        # Commit all
        db.session.commit()
        
        print("Database seeded successfully!")
        print("\nDefault login credentials:")
        print("  Admin: admin / admin123")
        print("  Sales: sales / sales123")
        print("  Production: production / prod123")


if __name__ == '__main__':
    seed_database()
