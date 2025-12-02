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
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=float(83.0), min_value=float(0.01), step=float(0.01), format="%.2f")
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=float(90.0), min_value=float(0.01), step=float(0.01), format="%.2f")
default_scrap = st.sidebar.number_input("Scrap (%)", value=float(2.0), min_value=float(0.0), step=float(0.1), format="%.2f")
default_overhead = st.sidebar.number_input("Overhead (%)", value=float(10.0), min_value=float(0.0), step=float(0.1), format="%.2f")
default_profit = st.sidebar.number_input("Profit (%)", value=float(8.0), min_value=float(0.0), step=float(0.1), format="%.2f")

# -------------------- Tabs --------------------
tabs = st.tabs(["Bulk / DIN","Single Calculator","DIN DB","Materials","History"])

# -------------------- Single Calculator --------------------
with tabs[1]:
    st.header("Single Item Calculator")
    stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    diameter = st.number_input("Diameter / AF / Side / OD (mm)", value=float(30.0), min_value=float(0.1), step=float(0.1), format="%.2f")
    length = st.number_input("Length (mm)", value=float(50.0), min_value=float(0.1), step=float(0.1), format="%.2f")
    auto_parting = st.checkbox("Auto parting", True)
    parting = compute_auto_parting(length) if auto_parting else st.number_input("Parting (mm)", value=float(5.0), min_value=float(0.1), step=float(0.1), format="%.2f")
    qty = st.number_input("Quantity", value=int(100), min_value=int(1), step=int(1), format="%d")
    material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]

    # --- FIXED: convert to native Python types using .item() ---
    density = st.number_input("Density (kg/m3)", value=float(mrow["Density (kg/m3)"].item()), min_value=float(0.1), step=float(0.1), format="%.2f")
    mat_price = st.number_input("Material price (₹/kg)", value=float(mrow["Default Price (₹/kg)"].item()), min_value=float(0.01), step=float(0.01), format="%.2f")

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
    
    st.subheader("Bulk Input by DIN / Article Number")
    bulk_file = st.file_uploader("Upload CSV/XLSX with 'DIN', optionally 'Size' & 'Qty'", type=["csv","xlsx"])
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
            st.data_editor(bulk_df, num_rows="dynamic")

    openai_key = st.text_input("OpenAI API Key (for GPT lookup)", type="password")
    stock_type_bulk = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    material_bulk = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow_bulk = st.session_state.materials_df[st.session_state.materials_df["Material"]==material_bulk].iloc[0]

    density_bulk = st.number_input("Density (kg/m3)", value=float(mrow_bulk["Density (kg/m3)"].item()), min_value=float(0.1), step=float(0.1), format="%.2f")
    mat_price_bulk = st.number_input("Material price (₹/kg)", value=float(mrow_bulk["Default Price (₹/kg)"].item()), min_value=float(0.01), step=float(0.01), format="%.2f")
    auto_parting_bulk = st.checkbox("Auto parting for bulk", True)

    # ... rest of bulk processing as in previous code ...

# -------------------- DIN DB Tab --------------------
with tabs[2]:
    st.header("DIN / ISO Database")
    st.data_editor(st.session_state.din_db, num_rows="dynamic")

# -------------------- Materials Tab --------------------
with tabs[3]:
    st.header("Materials Database")
    st.data_editor(st.session_state.materials_df, num_rows="dynamic")

# -------------------- History Tab --------------------
with tabs[4]:
    st.header("Costing History")
    st.data_editor(st.session_state.cost_history, num_rows="dynamic")
