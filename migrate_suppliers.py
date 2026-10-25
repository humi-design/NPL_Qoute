"""Data migration: backfill the Supplier master from existing free-text data.

Existing installs stored a supplier as free text on ``materials.default_supplier``
(and ``material_rates.supplier``). This script promotes those strings into real
``suppliers`` rows and links the materials to them, so no historical data is
lost and referential integrity is established going forward.

Safe to run repeatedly (idempotent): existing suppliers are matched by name and
reused; materials already linked are left untouched. Nothing is deleted.

Usage:
    python migrate_suppliers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.extensions import create_app, db
from app.models import Material, Supplier, SupplierType, SupplierMaterial
from app.utils import generate_supplier_code


# Seed supplier types used when the table is empty (kept in sync with seed_data).
DEFAULT_TYPES = [
    ('SUPT-001', 'Raw Material', 'Ferrous and non-ferrous raw material suppliers', 10),
    ('SUPT-002', 'Consumables', 'Cutting tools, coolant, packaging consumables', 20),
    ('SUPT-003', 'Hardware & Fasteners', 'Standard hardware and fastener components', 30),
    ('SUPT-004', 'Services', 'Job work and outsourced service providers', 40),
]


def _ensure_types():
    """Create the default supplier types if none exist yet."""
    if SupplierType.query.first():
        return
    for code, name, desc, order in DEFAULT_TYPES:
        db.session.add(SupplierType(type_code=code, type_name=name,
                                    description=desc, sort_order=order, is_active=True))
    db.session.flush()
    print(f"  Created {len(DEFAULT_TYPES)} default supplier types.")


def _default_type_id():
    """Prefer the 'Raw Material' type for backfilled suppliers."""
    raw = SupplierType.query.filter_by(type_code='SUPT-001').first()
    if raw:
        return raw.id
    first = SupplierType.query.order_by(SupplierType.sort_order).first()
    return first.id if first else None


def migrate():
    app = create_app('development')
    with app.app_context():
        # Tables must exist (schema.sql / migrate.sql creates them additively).
        db.create_all()

        print("Backfilling suppliers from existing material data...")
        _ensure_types()
        default_type_id = _default_type_id()

        # Collect every distinct non-empty supplier name already referenced.
        names = set()
        for (value,) in db.session.query(Material.default_supplier).distinct().all():
            if value and value.strip():
                names.add(value.strip())

        created = 0
        linked = 0
        cache = {}

        for name in sorted(names):
            supplier = Supplier.query.filter_by(company_name=name).first()
            if supplier is None:
                supplier = Supplier(
                    supplier_code=generate_supplier_code(),
                    company_name=name,
                    supplier_type_id=default_type_id,
                    is_active=True,
                    notes='Backfilled from existing material default_supplier values.',
                )
                db.session.add(supplier)
                db.session.flush()
                created += 1
                print(f"  + {supplier.supplier_code}  {name}")
            cache[name] = supplier

        # Link materials that carry a free-text supplier but no master link.
        for material in Material.query.filter(Material.supplier_id.is_(None)).all():
            name = (material.default_supplier or '').strip()
            if not name or name not in cache:
                continue
            supplier = cache[name]
            material.supplier_id = supplier.id
            linked += 1

            # Mirror the material/grade association so the supplier detail page
            # reflects what was already implied by the free text.
            if not SupplierMaterial.query.filter_by(supplier_id=supplier.id,
                                                    material_id=material.id).first():
                db.session.add(SupplierMaterial(
                    supplier_id=supplier.id,
                    material_id=material.id,
                    grade=material.grade,
                    rate=material.default_rate,
                    rate_unit=material.rate_unit or '₹/kg',
                    is_preferred=True,
                ))

        db.session.commit()
        print(f"Done. Suppliers created: {created}, materials linked: {linked}.")


if __name__ == '__main__':
    migrate()
