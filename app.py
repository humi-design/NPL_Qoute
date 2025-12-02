# app.py
import streamlit as st
import pandas as pd
import datetime
from pathlib import Path

from utils import *
from data import *

# -------------------- Paths --------------------
DATA_DIR = Path("fastener_data"); DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"

# -------------------- Session State --------------------
if "materials_df" not in st.session_state:
    st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, DEFAULT_MATERIALS.columns.tolist(), DEFAULT_MATERIALS)

if "din_db" not in st.session_state:
    st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DEFAULTS.columns.tolist(), DIN_DEFAULTS)

if "cost_history" not in st.session_state:
    st.session_state.cost_history = load_or_init_csv(HISTORY_CSV, ["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])

# -------------------- Sidebar --------------------
st.sidebar.header("Rates & Settings")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=float(83.0), min_value=float(0.01), step=float(0.01), format="%.2f", key="usd_rate")
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=float(90.0), min_value=float(0.01), step=float(0.01), format="%.2f", key="eur_rate")
default_scrap = st.sidebar.number_input("Scrap (%)", value=float(2.0), min_value=float(0.0), step=float(0.1), format="%.2f", key="scrap")
default_overhead = st.sidebar.number_input("Overhead (%)", value=float(10.0), min_value=float(0.0), step=float(0.1), format="%.2f", key="overhead")
default_profit = st.sidebar.number_input("Profit (%)", value=float(8.0), min_value=float(0.0), step=float(0.1), format="%.2f", key="profit")

# -------------------- Tabs --------------------
tabs = st.tabs(["Bulk / DIN","Single Calculator","DIN DB","Materials","History"])

# -------------------- Single Calculator --------------------
with tabs[1]:
    st.header("Single Item Calculator")
    stock_type = st.selectbox("Stock Type (Single)", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"], key="stock_single")
    diameter = st.number_input("Diameter / AF / Side / OD (mm) (Single)", value=float(30.0), min_value=float(0.1), step=float(0.1), format="%.2f", key="dia_single")
    length = st.number_input("Length (mm) (Single)", value=float(50.0), min_value=float(0.1), step=float(0.1), format="%.2f", key="len_single")
    auto_parting = st.checkbox("Auto parting (Single)", True, key="auto_part_single")
    parting = compute_auto_parting(length) if auto_parting else st.number_input("Parting (mm) (Single)", value=float(5.0), min_value=float(0.1), step=float(0.1), format="%.2f", key="part_single")
    qty = st.number_input("Quantity (Single)", value=int(100), min_value=int(1), step=int(1), format="%d", key="qty_single")
    material = st.selectbox("Material (Single)", st.session_state.materials_df["Material"].tolist(), key="mat_single")
    mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
    density = st.number_input("Density (kg/m3) (Single)", value=float(mrow["Density (kg/m3)"].item()), min_value=float(0.1), step=float(0.1), format="%.2f", key="density_single")
    mat_price = st.number_input("Material price (₹/kg) (Single)", value=float(mrow["Default Price (₹/kg)"].item()), min_value=float(0.01), step=float(0.01), format="%.2f", key="price_single")

    mass_kg = volume_by_stock(stock_type, diameter, length+parting)*density/1e9
    material_cost = mass_kg*mat_price
    traub_cost = material_cost*0.1
    milling_cost = material_cost*0.05
    threading_cost = material_cost*0.05
    punching_cost = material_cost*0.02
    tooling_cost = material_cost*0.01

    subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_cost
    subtotal *= (1 + default_scrap/100)*(1 + default_overhead/100)*(1 + default_profit/100)

    st.metric("Material kg/pc", f"{mass_kg:.6f}")
    st.metric("Total per piece (INR)", f"₹ {subtotal:.2f}")
    st.metric("Total batch cost (INR)", f"₹ {subtotal*qty:.2f}")

# -------------------- Bulk / DIN Batch Costing --------------------
with tabs[0]:
    st.header("Bulk / DIN Batch Costing")
    bulk_file = st.file_uploader("Upload CSV/XLSX with 'DIN', optionally 'Size' & 'Qty'", type=["csv","xlsx"], key="bulk_file")
    openai_key = st.text_input("OpenAI API Key (for GPT lookup)", type="password", key="openai_key")
    stock_type_bulk = st.selectbox("Stock Type (Bulk)", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"], key="stock_bulk")
    material_bulk = st.selectbox("Material (Bulk)", st.session_state.materials_df["Material"].tolist(), key="mat_bulk")
    mrow_bulk = st.session_state.materials_df[st.session_state.materials_df["Material"]==material_bulk].iloc[0]
    density_bulk = st.number_input("Density (kg/m3) (Bulk)", value=float(mrow_bulk["Density (kg/m3)"].item()), min_value=float(0.1), step=float(0.1), format="%.2f", key="density_bulk")
    mat_price_bulk = st.number_input("Material price (₹/kg) (Bulk)", value=float(mrow_bulk["Default Price (₹/kg)"].item()), min_value=float(0.01), step=float(0.01), format="%.2f", key="price_bulk")
    auto_parting_bulk = st.checkbox("Auto parting for bulk", True, key="auto_part_bulk")

    if bulk_file:
        if bulk_file.name.lower().endswith(".csv"):
            bulk_df = pd.read_csv(bulk_file)
        else:
            bulk_df = pd.read_excel(bulk_file)

        if 'DIN' not in bulk_df.columns:
            st.error("CSV must have 'DIN' column")
        else:
            if 'Qty' not in bulk_df.columns:
                bulk_df['Qty'] = int(100)

            # Auto-fill dimensions if OpenAI API key provided
            if openai_key:
                for idx, row in bulk_df.iterrows():
                    if pd.isna(row.get('Diameter(mm)', None)):
                        diameter_val, head_dia_val = query_gpt_for_din(row['DIN'], row.get('Size', ''), stock_type_bulk, openai_key)
                        bulk_df.at[idx, 'Diameter(mm)'] = diameter_val
                        bulk_df.at[idx, 'HeadDia(mm)'] = head_dia_val

            # Auto parting
            if auto_parting_bulk:
                bulk_df['Parting(mm)'] = bulk_df['Length(mm)'].apply(compute_auto_parting)
            else:
                if 'Parting(mm)' not in bulk_df.columns:
                    bulk_df['Parting(mm)'] = float(5.0)

            # Compute costing per item
            bulk_df['Material_kg'] = bulk_df.apply(
                lambda x: volume_by_stock(stock_type_bulk, float(x['Diameter(mm)']), float(x['Length(mm)'])+float(x['Parting(mm)'])) * float(density_bulk)/1e9,
                axis=1
            )
            bulk_df['MaterialCost'] = bulk_df['Material_kg'] * float(mat_price_bulk)
            bulk_df['TraubCost'] = bulk_df['MaterialCost'] * 0.1
            bulk_df['MillingCost'] = bulk_df['MaterialCost'] * 0.05
            bulk_df['ThreadingCost'] = bulk_df['MaterialCost'] * 0.05
            bulk_df['PunchingCost'] = bulk_df['MaterialCost'] * 0.02
            bulk_df['ToolingCost'] = bulk_df['MaterialCost'] * 0.01
            bulk_df['Subtotal_INR'] = (bulk_df['MaterialCost'] + bulk_df['TraubCost'] + bulk_df['MillingCost'] +
                                       bulk_df['ThreadingCost'] + bulk_df['PunchingCost'] + bulk_df['ToolingCost'])
            bulk_df['TotalUnitPrice_INR'] = bulk_df['Subtotal_INR'] * (1 + default_scrap/100)*(1 + default_overhead/100)*(1 + default_profit/100)
            bulk_df['TotalBatch_INR'] = bulk_df['TotalUnitPrice_INR'] * bulk_df['Qty']

            if 'TargetPrice' not in bulk_df.columns:
                bulk_df['TargetPrice'] = None

            st.subheader("Bulk Costing Preview")
            st.data_editor(bulk_df, num_rows="dynamic", key="bulk_calc_editor")

            # Editable Target Price
            st.subheader("Enter Target Prices (Optional)")
            for idx in bulk_df.index:
                bulk_df.at[idx, 'TargetPrice'] = st.number_input(
                    f"Target Price for {bulk_df.at[idx,'DIN']} (INR)", 
                    value=float(bulk_df.at[idx,'TargetPrice']) if bulk_df.at[idx,'TargetPrice'] else float(0),
                    min_value=float(0),
                    step=float(0.01),
                    format="%.2f",
                    key=f"target_{idx}"
                )

            # Currency conversion
            bulk_df['TotalUnitPrice_USD'] = bulk_df['TotalUnitPrice_INR'] / float(usd_rate)
            bulk_df['TotalUnitPrice_EUR'] = bulk_df['TotalUnitPrice_INR'] / float(eur_rate)

            # Save to history
            if st.button("Save Bulk Costing to History", key="save_bulk_history"):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for idx, row in bulk_df.iterrows():
                    st.session_state.cost_history = st.session_state.cost_history.append({
                        "Timestamp": timestamp,
                        "PartDesc": row['DIN'],
                        "Qty": row['Qty'],
                        "Material": material_bulk,
                        "Diameter(mm)": row['Diameter(mm)'],
                        "Length(mm)": row['Length(mm)'],
                        "Parting(mm)": row['Parting(mm)'],
                        "UnitPrice_INR": row['TotalUnitPrice_INR'],
                        "Total_INR": row['TotalBatch_INR'],
                        "TargetPrice": row['TargetPrice'],
                        "Notes": ""
                    }, ignore_index=True)
                st.session_state.cost_history.to_csv(HISTORY_CSV, index=False)
                st.success("Saved to history!")

            # PDF Export
            if st.button("Generate Quotation PDF", key="pdf_export"):
                pdf_bytes = pdf_quote_bytes(bulk_df.to_dict(orient="records"))
                st.download_button("Download Quotation PDF", pdf_bytes, file_name="quotation.pdf", mime="application/pdf")

# -------------------- DIN DB Tab --------------------
with tabs[2]:
    st.header("DIN / ISO DB")
    st.data_editor(st.session_state.din_db, num_rows="dynamic", key="din_db_editor")

# -------------------- Materials Tab --------------------
with tabs[3]:
    st.header("Materials Database")
    st.data_editor(st.session_state.materials_df, num_rows="dynamic", key="materials_editor")

# -------------------- History Tab --------------------
with tabs[4]:
    st.header("Costing History")
    st.data_editor(st.session_state.cost_history, num_rows="dynamic", key="history_editor")
