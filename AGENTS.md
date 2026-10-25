# AGENTS.md — Nirmal Precision Fasteners ERP

## Stack
- Flask 3.x app-factory (`app/extensions.py: create_app`), Flask-SQLAlchemy, Flask-Login,
  Flask-WTF (CSRF on by default), Flask-Bcrypt, Flask-Migrate, Flask-CORS.
- DB: MySQL via PyMySQL in production; SQLite (`:memory:` / file) for `testing`.
- Exports: openpyxl (Excel), reportlab (PDF). Python 3.13.
- Blueprints per module under `app/<module>/routes.py`; templates in `app/templates/<module>/`.

## Commands
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                 # run dev server
.venv/bin/python seed_data.py           # seed demo data (idempotent-ish)
.venv/bin/python -m pytest -q           # run tests (tests/ dir)
mysql < schema.sql                       # full schema (fresh DB)
mysql < migrate.sql                      # additive migration (existing DB)
```
Configs: `app/config.py` — `development`, `testing`, `production`.

## Testing notes
- Tests live in `tests/`. They use `create_app('testing')` + `db.create_all()`.
- **In-memory SQLite is per-connection** — for multi-request flows use a file DB
  (`sqlite:////tmp/foo.db`), not `sqlite:///:memory:`.
- `LoginForm` requires `username` Length(min=3); short usernames silently fail login.
- `Flask-WTF` CSRF is enabled in `development`/`production`. In tests, either set
  `WTF_CSRF_ENABLED=False` or scrape the `csrf_token` from the rendered form.

## Conventions / gotchas
- `migrate.sql` uses MariaDB-style `ADD COLUMN IF NOT EXISTS`; MySQL lacks
  `CREATE INDEX IF NOT EXISTS`, so indexes are declared only in `schema.sql`.
- **Schema drift guard**: `schema.sql` (fresh install) and `migrate.sql` (existing
  DB upgrade) must both cover every model column. `tests/test_schema_consistency.py`
  fails if a model column is missing from both. Add new columns to BOTH files.
- SQLAlchemy: MySQL has no `NULLS LAST`; use `case((col.is_(None), 1), else_=0)` for null-last ordering.
- `base.html` template blocks: `extra_css`, `breadcrumb`, `content`, `extra_js`.
- Do not reset the DB or duplicate modules; extend existing blueprints/templates.

## Referential integrity on master-data deletes
- `app/utils.py`: `dependent_counts(record)` and `referential_delete_block(record)`
  report relationships that still reference a record. Relationships with
  `delete-orphan` cascade are skipped (owned children go with the parent).
- Customer / product / material / machine / process / vendor delete routes now
  refuse while referenced, and flash which relations block the delete.
- Why: before this, deleting a customer with orders raised an unhandled
  `IntegrityError` (500), and deleting a material referenced by
  `supplier_materials` silently nulled `material_id`, corrupting history.
- `tests/test_referential_integrity.py` pins the guarded behaviour.

## Error pages
- `errors/error.html` is standalone (does not extend `base.html`) so it renders
  even when the DB is the thing that failed. Handlers for 403/404/405/500 live in
  `app/extensions.py`; the 500 handler logs the traceback and never leaks it.
- `tests/test_error_handlers.py` pins the pages and the no-leak guarantee.

## Settings: SystemSetting vs app.config
- Company identity and costing defaults live in `SystemSetting`, edited at
  `/settings/general`. Read them through `app/utils.py: get_company_settings()`
  and `get_costing_defaults()` — each falls back to `app.config` when the row is
  absent or unparseable.
- Why: the page used to write values nothing read (the document header, footer
  and costing calculator read `app.config`, i.e. the environment), so saving had
  no effect. `setting.general` was also GET-only while handling POST, so the form
  returned 405 and never saved.

## CSRF coverage
- Every POST form must carry `csrf_token`. WTForms-based pages get it from
  `{{ form.hidden_tag() }}`; hand-written forms need
  `<input type="hidden" name="csrf_token" value="{{ csrf_token() }}">`.
- CSRF is enabled in development and production (only `testing` disables it), so
  a missing token returns 400 in real use.
- `tests/test_form_csrf.py` renders every GET page with CSRF on and asserts each
  POST form has a token — this also catches forms assembled from includes.

## Security: redirects and user deletion
- `app/utils.py`: `safe_redirect_target(target, fallback)` allows only same-site
  absolute paths. The post-login `next` param and the notification `next` form
  field both go through it. Before this, `/auth/login?next=https://evil.example.com`
  was an open redirect.
- `auth.delete_user` now refuses to delete a user who authored records (orders,
  quotations, GRNs...) and clears the user's own notifications and alert
  acknowledgements. Previously it left dangling `created_by` ids (and would 500
  once FKs are enforced).
- `tests/test_auth_security.py` pins both.

## Write authorization (central guard)
- `app/extensions.py` installs a single `before_request` guard: any non-GET
  request whose endpoint is not in the exempt set requires `write` permission.
- Legacy master-data modules (customer, vendor, material, product, process,
  machine, quotation, rfq, setting) previously had NO write check — a `read_only`
  account could create/edit/delete. The guard closes that gap centrally.
- Exempt set (per-user state or pure calculators, no shared-record mutation):
  auth (login/logout/profile), `api.*` (separately `login_required`),
  notifications (own), alerts acknowledge/unacknowledge (own),
  `quotation.cost_item` and the material/costing calculators.
- `tests/test_write_authorization.py` proves read_only is blocked while roles with
  the `write` permission (admin/sales/production) still write.

## Vendors module (upgraded)
- Models: `Vendor`, `VendorRate`, `VendorWorkOrder`, `VendorQualityIssue`.
- `VendorWorkOrder`: `status` in {pending, in_progress, completed, cancelled, on_hold};
  helpers `is_open`, `is_completed`, `is_cancelled`, `is_overdue`, `days_overdue`,
  `pending_quantity`; class constants `OPEN_STATUSES`, `COMPLETE_STATUSES`.
- Utils: `generate_work_order_number`, `compute_vendor_summary`, `export_vendors_to_excel`.
- Routes (`app/vendor/routes.py`): `/vendors/` (search/filter/sort/pagination),
  `/vendors/<id>` (Current/Past/All work tabs), `/vendors/work-orders` register,
  work-order CRUD + `receive` + `status/<s>`, quality-issue CRUD, exports, api search.
- API (`app/api/routes.py`): `/api/vendors?summary=1`, `/api/vendors/<id>`,
  `/api/vendors/work-orders?view=current|past|all`.

## Suppliers module (Master Data)
- Suppliers = procurement master (companies we buy raw material/consumables from),
  distinct from Vendors (subcontracting/services). Do not merge the two.
- Models: `SupplierType`, `Supplier`, `SupplierMaterial`; `Material.supplier_id` FK
  (optional) coexists with the legacy free-text `Material.default_supplier`.
- `SupplierType` is configurable master data (code/name/sort_order/is_active) — never
  hardcode supplier categories in Python; read them from the table.
- Utils: `generate_supplier_code` (`SUP-`), `generate_supplier_type_code` (`SUPT-`),
  `compute_supplier_summary`, `export_suppliers_to_excel`.
- Routes (`app/supplier/routes.py`, prefix `/suppliers`): list grid (search/type/status
  filters, whitelisted sort via `SUPPLIER_SORT_FIELDS`, pagination, per_page), view
  (summary tiles + supplied materials + attachments), new/edit, `deactivate`/`activate`,
  guarded `delete`, `material` add/remove, `types` CRUD (admin-only), `/export`,
  `/api/search`.
- API: `/api/suppliers?summary=1&type=&status=`, `/api/suppliers/<id>`, `/api/supplier-types`.
- Delete is guarded: blocked when materials/documents are linked (deactivate instead).
- `migrate_suppliers.py` backfills the Supplier master from legacy free-text
  `materials.default_supplier` values; idempotent, non-destructive.
- Nav: `base.html` groups Customers/Suppliers/Vendors/Supplier Types under "Master Data".

## Procurement & payables module (Purchasing)
- New blueprint `app/procurement/` (prefix `/procurement`) captures the procurement
  data the supplier/vendor profiles previously had to mark "not tracked".
- Models (all additive, no existing table changed):
  - PO chain: `SupplierPurchaseOrder` -> `SupplierPurchaseOrderLine` ->
    `SupplierOrderReceipt` -> `SupplierOrderReceiptLine` (batch/heat/lot/certificate
    + `inspection_status`).
  - Supplier payables: `SupplierInvoice`, `SupplierPayment`,
    `SupplierPaymentAllocation`.
  - Vendor payables: `VendorInvoice`, `VendorPayment`, `VendorPaymentAllocation`.
- Integrity rules enforced in code and DB:
  - A PO line's received qty is *derived* from its receipt lines, so ordered /
    received / accepted / rejected / pending always reconcile (never stored twice).
  - A receipt line's rejected qty cannot exceed its received qty (route guard).
  - `UNIQUE(payment_id, invoice_id)` on both allocation tables means a payment
    can be applied to an invoice only once, so payments are never double counted.
  - A payment allocation cannot exceed the payment's unallocated balance (route guard).
  - Unallocated payment remainder = advance/credit (`unallocated_amount`).
- Utils: `generate_po_number` (`PO-YYYY-NNNN`), `generate_receipt_number`
  (`GRN-YYYY-NNNN`), `generate_payment_number` (`PAY-`/`VPAY-`),
  `compute_supplier_procurement_summary`, `compute_vendor_financial_summary`,
  `_ageing_from_invoices`, `AGEING_BUCKETS` (not_due/0-30/31-60/61-90/90+).
- Supplier view (`/suppliers/<id>`) now shows real POs, batches/inspection,
  invoices, payments and ageing; a payment can be allocated inline. Vendor view
  (`/vendors/<id>`) shows job-work invoices, payments and ageing.
- "Not tracked vs zero": `has_procurement_data` / `has_financial_data` flags
  distinguish "feature unused" (show explanatory note) from a genuine 0.
  `on_time_percent` is `None` (not measurable) when no receipt/expected dates exist.
- Nav: `base.html` adds a "Purchasing" section (Purchase Orders, Supplier
  Invoices, Supplier Payments).
- Schema: new tables appended to `schema.sql`; `migrate.sql` adds them with
  `CREATE TABLE IF NOT EXISTS` so existing databases are upgraded without data loss.
- `seed_data.py` calls `_seed_procurement()` to add one demo PO + partial receipt
  (with batch/heat/inspection), a supplier invoice + payment and a vendor invoice +
  payment, so the new panels are populated on a freshly seeded dev database.
- Models expose `supplier`/`vendor` relationships (with `purchase_orders`,
  `order_receipts`, `invoices`, `payments` backrefs) so templates and exports can
  reach the parent party without manual joins.
- List conventions (same as supplier/vendor modules): whitelisted `sort` keys,
  `dir`, `per_page` (10/25/50/100, invalid falls back to 25), `_nulls_last`
  ordering, and an `/export` route returning an XLSX via `send_file`. Sort columns
  come only from the `*_SORT_FIELDS` dicts, never from raw request input.
- Invoices and payments are editable (`/supplier-invoices/<id>/edit` etc.) with
  guards: an invoice total cannot drop below what is already paid, and a payment
  amount cannot drop below what is already allocated.

## Orders & manufacturing tracking module (Live Operations)
- New blueprint `app/orders/` (prefix `/orders`) upgrades the order and
  production-tracking workflows. No existing module was rebuilt or duplicated:
  it reuses `Customer`, `Product`, `Process`, `Machine`, `Vendor`, `Supplier`,
  `Material`, the procurement receipt chain and the quotation chain.
- Anchor model `Order` (table `orders`) is a *confirmed customer order*. It is
  self-contained (part/drawing/revision/spec/material grade copied in) and links
  optionally back to the won `Quotation`/`QuotationItem`/`Product`. A won
  quotation can be promoted to an order via `/orders/new?quotation_id=<id>`
  (fields are prefilled, nothing is auto-created).
- The part's route lives on the order, not on a fixed template:
  `OrderOperation` (ordered by `sequence`, in-house or `is_vendor` + `vendor_id`,
  `is_final` marks the step feeding finished stock). This supports a different
  sequence per fastener and multiple subcontractors.
- Ledgers (all additive; nothing existing was altered):
  - `OperationTransaction` — `issue` / `return` / `accept` / `reject` / `rework` /
    `scrap` per operation. Partial deliveries are multiple rows.
  - `MaterialLot` — received batch with heat/batch/lot/certificate, optionally
    linked to the procurement `SupplierOrderReceipt`/receipt line.
  - `MaterialMovement` — `receipt` / `allocation` / `issue` / `return` / `scrap` /
    `shortage` / `adjustment` per lot.
  - `Inspection` — incoming / in-process / final, with `acceptance_criteria`,
    `tolerance`, `measurements`, sample size and accepted/rejected/rework qty.
  - `PackagingRecord` — packaging + final accepted qty + warehouse receipt.
- Quantities are always *derived* from the ledgers, never stored twice, so
  ordered / produced / rejected / rework / dispatched / pending reconcile:
  - `OrderOperation.issued/returned/accepted/rejected/rework/scrapped` sum the
    operation transactions; `pending_quantity` = issued − returned − rejected −
    scrapped.
  - `Order.quantity_produced` = accepted qty of the final operation(s);
    `quantity_warehoused` = accepted qty of warehoused/dispatched packaging
    (a stock position); `quantity_shipped` = qty actually on non-cancelled
    dispatches. `quantity_dispatched` is a deprecated alias of
    `quantity_warehoused`; the order-summary "Dispatched" tile uses
    `quantity_shipped` so it matches the shipments card on the same page.
  - `MaterialLot.quantity_available` = received − issued − scrapped;
    `quantity_remaining` = available − allocated.
- Integrity rules enforced in code and DB (mirroring the procurement module):
  - A lot can never be over-allocated or over-issued (guarded against the derived
    available/remaining balances; the attempt is rejected with a clear message).
  - An operation can never return/accept/reject/rework/scrap more than it issued
    and has not yet accounted for.
  - `UNIQUE(order_operation_id, transaction_type, reference)` and
    `UNIQUE(lot_id, movement_type, reference)` stop the same movement being posted
    twice (both a route guard for a clear message and a DB constraint).
  - `UNIQUE(receipt_line_id)` on `material_lots` means one receipt line yields at
    most one lot, so received stock is never duplicated.
  - Accepted + rejected cannot exceed the quantity inspected.
  - Deleting an order is blocked once it has a route/lots/inspections/packaging;
    deleting an operation is blocked once it has movements (cancel instead).
- Status flow: `draft → confirmed → in_production → completed`, with `on_hold`
  and `cancelled`. `CURRENT_STATUSES` = open (draft/confirmed/in_production/
  on_hold); `PAST_STATUSES` = completed/cancelled. Operations auto-advance to
  in_progress/completed from their movements; an order auto-advances to
  in_production, and to completed only when the full ordered qty is packed and
  warehoused (not merely produced).
- Utils: `generate_order_number` (`SO-YYYY-NNNN`), `generate_inspection_number`
  (`INS-YYYY-NNNN`), `generate_lot_number` (`LOT-YYYY-NNNN`),
  `compute_order_summary`, `compute_material_requirement`, `build_order_timeline`,
  `export_orders_to_excel`, `export_material_lots_to_excel`,
  `export_vendor_process_to_excel`.
- Screens (all under the "Live Operations" nav section in `base.html`):
  - Orders register with Current / Past / All views, search (order/part/drawing/
    customer), customer + status + overdue filters, status summary tiles, sort,
    pagination and XLSX export.
  - Order detail: summary tiles, planned-vs-actual details, material requirement,
    the manufacturing route with an inline movement form per operation, material
    lots, inspections, packaging and a merged traceable timeline.
  - Raw Material Tracker & Allocation (`/orders/materials`) with lot balances.
  - Vendor Process Tracking (`/orders/vendor-processes`) across all operations.
  - Quality Inspection (`/orders/inspections`).
  - Final Inspection & Packaging (`/orders/packaging`).
- List conventions match the rest of the app: whitelisted `*_SORT_FIELDS`, `dir`,
  `per_page` (10/25/50/100, invalid falls back to 25), `_nulls_last` ordering and
  an `/export` route returning XLSX via `send_file`.
- Schema: the 7 new tables are appended to `schema.sql` and added to `migrate.sql`
  with `CREATE TABLE IF NOT EXISTS` so existing databases upgrade without data
  loss. `seed_data.py` calls `_seed_orders()` (non-fatal) to add two demo orders
  (one in production with a vendor step + lot + incoming inspection, one completed
  and warehoused) plus a demo customer when none exists.


## Inventory module (Stock Position, Ledger, Locations, Transfers, Adjustments)
- Blueprint `inventory_bp` at `/inventory` (`app/inventory/routes.py`), registered
  in `app/extensions.py`. Nav section "Inventory & Costing" in `base.html`.
- Models: `StockLocation`, `StockTransfer`, `StockAdjustment` (`app/models.py`).
  `StockAdjustment.reverses_movement_id` / `reverses_transfer_id` make a reversal a
  *new compensating row* — existing movement history is never edited or deleted.
- Stock is **derived, never stored twice**: `lot_position()` and
  `compute_inventory_position()` read the existing `MaterialMovement`,
  `OperationTransaction`, `PackagingRecord` and `StockAdjustment` ledgers.
  Categories: raw material (lot receipts), WIP (issued minus accepted on in-house
  ops), subcontractor (issued minus returned on vendor ops), finished goods
  (warehoused packaging), rejected / scrap.
- `build_stock_ledger()` merges movements, operation transactions, transfers and
  adjustments into one traceable event list with references.
- Guards (see `get_stock_policy()`): over-allocation and over-issue are blocked by
  default; a negative-stock adjustment is refused unless `_is_admin()` authorises
  it (then `StockAdjustment.authorized_by` is stamped). Transfers cannot exceed the
  lot's available quantity and cannot be from == to a location. Locations with
  history cannot be deleted. A movement can be reversed only once, and receipts
  cannot be reversed (reverse the source GRN instead).
- Screens: Stock Position (`/inventory/`, XLSX export), Stock Ledger
  (`/inventory/ledger`, per-lot detail `/inventory/lots/<id>`), Stock Locations,
  Stock Transfers, Stock Adjustments. Lists use the app's `*_SORT_FIELDS` +
  `dir` + `per_page` conventions.

## Costing module (order cost entries, estimate vs actual)
- Blueprint `costing_bp` at `/costing` (`app/costing/routes.py`).
- Model: `OrderCostEntry` with `UNIQUE(source_type, source_id, category)` so the
  same source document is never counted twice. `Order.cost_entries` is a
  `lazy='dynamic'` relationship; `OrderOperation.cost_entries` is the backref used
  by `_refresh_operation_actuals()` to roll actual per-piece cost onto a route step.
- `compute_order_cost()` compares the quotation/product-cost-history **estimate**
  against recorded **actuals**, per category, with variance and unit cost. When an
  input is unknown it is reported in `cost['missing']` — never guessed.
  `/costing/missing` lists every order with an incomplete cost picture.
- Entries generated from other documents (`source_type != 'manual'`) cannot be
  deleted on the costing screen; reverse the source document instead.

## Supplier RFQ & quote comparison (Purchasing)
- Models: `SupplierRFQ`, `SupplierQuote`. `SupplierRFQ.awarded_quote_id` is a plain
  `Integer` (no FK) with an explicit `awarded_quote` relationship
  (`foreign_keys=[awarded_quote_id]`) to avoid a circular FK with `SupplierQuote`.
- Routes live in the existing `app/procurement/routes.py` (no new blueprint):
  list/create/edit/view RFQ, add/delete quote, compare, export, award, create PO.
- `compare_supplier_quotes()` scores quotes on weighted price / lead time /
  quality / payment / delivery terms, converts to a target currency via `Currency`,
  ranks them and returns a recommendation plus the reasons. Weights and rationale
  are shown on the comparison screen.
- **Nothing is awarded automatically.** `award_supplier_quote` requires
  `current_user.has_permission('admin')`; only after an award can
  `create_po_from_rfq` raise the PO (and then the RFQ/PO link is protected from
  deletion). Quality defaults to `supplier_quality_history()` (derived from recorded
  receipt rejections) unless a manual `quality_rating` is given.
- Schema: the 6 new tables are appended to `schema.sql` and added to `migrate.sql`
  with `CREATE TABLE IF NOT EXISTS` so existing databases upgrade without data
  loss. `seed_data.py` calls `_seed_inventory_costing()` (non-fatal) to add demo
  stock locations, a transfer, an adjustment, actual cost entries and a supplier
  RFQ with three comparable quotes.

## Documents, Dispatch, Alerts & Notifications (this phase)
- **Documents** (`app/documents/`, `/documents/`): built on the existing
  `Attachment` model (no new table) — added document-control columns (`title`,
  `category`, `revision`, `issue_date`, `notes`, `is_current`, `supersedes_id`,
  `file_type`, `is_primary`, `uploaded_at`) to the model, `schema.sql` and
  `migrate.sql`. The central register filters by category/entity/current-only,
  and `for/<entity_type>/<id>` shows one record's documents. New revision with the
  same title+category marks the previous one superseded (history is kept, never
  deleted). Download goes through `resolve_upload_path()` (path-traversal guard)
  and `inline_ok()`; non-renderable types are always force-downloaded.
- **Dispatch** (`Dispatch`, `DispatchLine`, in `app/orders/routes.py` — no new
  blueprint): ships already-warehoused `PackagingRecord` quantities. A line's
  shipped quantity cannot exceed `packaging_available()` (accepted minus already
  shipped); deleting a line releases the stock back to `warehoused`.
  `compute_order_dispatch_summary()` derives ordered/produced/warehoused/shipped/
  remaining so a partial dispatch never marks the order complete.
- **Alerts** (`app/alerts/`): alerts are DERIVED by `compute_alerts()` from the
  live ledgers with a deterministic key, so they cannot go stale or duplicate.
  The only stored state is per-user acknowledgement (`AlertAcknowledgement`).
  Payment alerts are hidden unless `_may_see_financials(user)` is true.
- **Notifications** (`app/notifications/`): the `Notification` model already
  existed but had no routes; this adds the centre (all/unread/read, mark read,
  mark all read, delete, pagination) and `resolve_notification_link()` so the
  topbar dropdown links to the source record. `notify_roles()` in `utils.py`
  fans a workflow event out to users by role; wired to order/quotation/dispatch/
  PO creation. Notification rows are per-user and never visible across users.
- Dashboard gains an "Operational Metrics" section via
  `compute_operational_metrics(user, include_financials=...)`; financial figures
  are only included when `_may_see_financials(user)` is true.
- `seed_data.py` calls `_seed_documents_dispatch()` (non-fatal) for one document,
  one dispatch with a line, and one notification on a fresh database.
- Tests: `tests/test_documents_dispatch_alerts.py` (documents, dispatch, alerts,
  notifications, dashboard metrics). Full suite: 193 passed.

## Schema drift: schema.sql vs migrate.sql (Phase 7 audit)
- **Both files must cover every model column.** A live-MariaDB run proved that
  `schema.sql` alone leaves 32 model columns undefined; `migrate.sql` supplies
  them via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. The documented install is
  therefore `schema.sql` (base tables + seed rows) **then** `migrate.sql`.
- Two columns were in *neither* file and broke a fresh install at runtime with
  `OperationalError: Unknown column`:
  `process_template_steps.sequence` (model: `NOT NULL`) and
  `quotation_operations.process_id`. Both are now defined in `schema.sql` and
  added idempotently by `migrate.sql` (the latter also repairs legacy installs
  created from the old `step_number`-only definition).
- `tests/test_schema_consistency.py` used to search `migrate.sql` as one blob, so
  an unrelated `ALTER TABLE rfq_items ... ADD COLUMN sequence` masked the missing
  `process_template_steps.sequence`. The check is now **scoped per table**
  (`_migrate_added_columns()`), and a new test asserts that any model table absent
  from `schema.sql` is at least created by `migrate.sql`.
- `schema.sql` seed `INSERT`s were not idempotent; re-running the file aborted on
  `Duplicate entry 'SS303'`. Materials, machines, processes and currencies now end
  with `ON DUPLICATE KEY UPDATE <unique_col>=VALUES(<unique_col>)`.
- `raw_material_calculations` is a legacy orphan table (created by both SQL files,
  no model, no routes). It is harmless and left in place — do not build on it.
- Live verification (opt-in, skipped by default):
  ```bash
  docker run -d --name npl-mariadb -e MARIADB_ROOT_PASSWORD=rootpw \
      -p 13306:3306 mariadb:11.4
  NPL_TEST_MYSQL_URL='mysql+pymysql://root:rootpw@127.0.0.1:13306' \
      .venv/bin/python -m pytest tests/test_migration_mysql.py
  ```
  `tests/test_migration_mysql.py` runs the real SQL against a throwaway database,
  asserts `schema.sql` is repeatable, `migrate.sql` is idempotent, there is no
  drift, and every mapped table is selectable. It creates/drops only its own
  `npl_*_check` databases.

## Financial visibility (supplier / vendor profiles)
- `_may_see_financials(user)` (admin or `accounts` role) already gated the
  dashboard and alerts. Supplier and vendor **profiles** previously exposed
  invoice totals, outstanding balances and ageing to every logged-in role.
- Both view routes now pass `can_view_financials=_may_see_financials(current_user)`
  and the templates wrap money/ageing blocks in `{% if can_view_financials %}`.
  Quantities, statuses and delivery performance stay visible to all roles.
- Tests: `tests/test_financial_visibility.py` asserts a `read_only` user sees no
  amounts, and `admin`/`accounts` do. Full suite: **232 passed, 3 skipped**.
