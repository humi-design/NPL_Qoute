# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import datetime

st.set_page_config(layout="wide", page_title="Fastener Costing App")

# ---------- Paths ----------
DATA_DIR = Path("fastener_data")
DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
VENDOR_CSV = DATA_DIR / "vendor_db.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"

# ---------- Helper Functions ----------
def area_round(d): return np.pi*(d/2)**2
def area_hex(af): return (3*np.sqrt(3)/2)*(af/2)**2
def area_square(s): return s**2

def volume_by_stock(stock_type, dim, total_length, inner_dim=None, thickness=None, width=None):
    if stock_type=="Round Bar": return area_round(dim)*total_length
    if stock_type=="Hex Bar": return area_hex(dim)*total_length
    if stock_type=="Square Bar": return area_square(dim)*total_length
    if stock_type=="Tube":
        if inner_dim: return (area_round(dim)-area_round(inner_dim))*total_length
        return area_round(dim)*total_length
    if stock_type=="Sheet/Cold Formed":
        if thickness and width: return thickness*width*total_length
        return dim*total_length
    return area_round(dim)*total_length

def load_or_init_csv(path, cols, default_df=None):
    if path.exists():
        try:
            return pd.read_csv(path)
        except:
            pass
    if default_df is not None:
        default_df.to_csv(path,index=False)
        return default_df.copy()
    return pd.DataFrame(columns=cols)

# ---------- Default Data ----------
DEFAULT_MATERIALS = pd.DataFrame([
    ["A2 Stainless (304)", 8000.0, 220.0],
    ["A4 Stainless (316)", 8020.0, 300.0],
    ["Steel (C45 / EN8)", 7850.0, 80.0],
    ["Brass (CW614N)", 8530.0, 450.0],
    ["Aluminium 6061", 2700.0, 300.0],
    ["Copper", 8960.0, 700.0],
], columns=["Material","Density (kg/m3)","Default Price (₹/kg)"])

DIN_DEFAULTS = pd.DataFrame([
    ["DIN 933","M8","hex_bolt",8,13,5.3,13,1.25,"Example M8 DIN933"],
    ["DIN 933","M10","hex_bolt",10,16,6.4,17,1.5,"Example M10 DIN933"],
    ["DIN 935","M30","castle_nut",30,48,12,46,2.5,"Example M30 DIN935"],
], columns=["Standard","Size","HeadType","d","dk","k","s","pitch","Notes"])

# ---------- Session State ----------
for key, val in {
    "materials_df": (MATERIALS_CSV, DEFAULT_MATERIALS),
    "vendor_db": (VENDOR_CSV, pd.DataFrame(columns=["Vendor","Item/Spec","Unit Price (₹)","Lead Time (days)","Notes"])),
    "cost_history": (HISTORY_CSV, pd.DataFrame(columns=["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])),
    "din_db": (DIN_DB_CSV, DIN_DEFAULTS)
}.items():
    if key not in st.session_state:
        st.session_state[key] = load_or_init_csv(val[0], val[1].columns.tolist(), default_df=val[1])

# ---------- Sidebar ----------
st.sidebar.header("Settings & Rates")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0, format="%.2f")
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0, format="%.2f")
default_scrap = st.sidebar.number_input("Scrap (%)", value=2.0, format="%.2f")
default_overhead = st.sidebar.number_input("Overhead (%)", value=10.0, format="%.2f")
default_profit = st.sidebar.number_input("Profit (%)", value=8.0, format="%.2f")

# ---------- Tabs ----------
tabs = st.tabs(["Bulk / DIN","Single Calculator","DIN DB","Materials","Vendor DB","History","Trusted Source"])

# ---------- DIN DB tab ----------
with tabs[2]:
    st.header("Local DIN / ISO DB")
    din_df = st.session_state.din_db.copy()
    edited = st.data_editor(din_df, num_rows="dynamic")
    st.session_state.din_db = edited
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Save DIN DB"):
            st.session_state.din_db.to_csv(DIN_DB_CSV,index=False); st.success("Saved DIN DB")
    with c2:
        if st.button("Reload DIN DB"):
            st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DEFAULTS.columns.tolist(), default_df=DIN_DEFAULTS)
            st.success("Reloaded DIN DB")

# ---------- Materials tab ----------
with tabs[3]:
    st.header("Materials")
    mat_df = st.session_state.materials_df.copy()
    edited_mat = st.data_editor(mat_df, num_rows="dynamic")
    st.session_state.materials_df = edited_mat
    m1,m2 = st.columns(2)
    with m1:
        if st.button("Save Materials"): st.session_state.materials_df.to_csv(MATERIALS_CSV,index=False); st.success("Materials saved")
    with m2:
        if st.button("Reload Materials"): st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, DEFAULT_MATERIALS.columns.tolist(), default_df=DEFAULT_MATERIALS); st.success("Materials reloaded")

# ---------- Vendor DB tab ----------
with tabs[4]:
    st.header("Vendor DB")
    vdf = st.session_state.vendor_db.copy()
    edited_v = st.data_editor(vdf,num_rows="dynamic")
    st.session_state.vendor_db = edited_v
    if st.button("Save Vendor DB"): st.session_state.vendor_db.to_csv(VENDOR_CSV,index=False); st.success("Saved Vendor DB")

# ---------- History tab ----------
with tabs[5]:
    st.header("Costing History")
    hdf = st.session_state.cost_history.copy()
    edited_h = st.data_editor(hdf,num_rows="dynamic")
    st.session_state.cost_history = edited_h
    if st.button("Save History"): st.session_state.cost_history.to_csv(HISTORY_CSV,index=False); st.success("Saved history")

# ---------- Trusted source tab ----------
with tabs[6]:
    st.header("Trusted Source Import")
    trusted_file = st.file_uploader("Upload trusted CSV/XLSX", type=["csv","xlsx"])
    if trusted_file:
        if trusted_file.name.lower().endswith(".csv"): trusted_df = pd.read_csv(trusted_file)
        else: trusted_df = pd.read_excel(trusted_file)
        st.data_editor(trusted_df)
        if st.button("Merge into local DIN DB"):
            local = st.session_state.din_db.copy(); added=0
            for _, r in trusted_df.iterrows():
                std = str(r.get("Standard","")).strip()
                sz = str(r.get("Size","")).strip()
                if std=="" or sz=="": continue
                exists = local[(local["Standard"].str.upper()==std.upper())&(local["Size"].str.upper()==sz.upper())]
                if exists.empty:
                    local = local.append({c:r.get(c,None) for c in local.columns}, ignore_index=True)
                    added += 1
            st.session_state.din_db = local; st.session_state.din_db.to_csv(DIN_DB_CSV,index=False)
            st.success(f"Merged {added} new rows")

# ---------- Single Calculator tab ----------
with tabs[1]:
    st.header("Single Item Calculator")
    stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    diameter = st.number_input("Diameter / AF / Side / OD (mm)", value=30.0, format="%.2f")
    length   = st.number_input("Length (mm)", value=50.0, format="%.2f")
    auto_parting = st.checkbox("Auto parting", True)
    def compute_auto_parting(L):
        if L<=25: return 3.0
        if L<=35: return 4.0
        if L<=50: return 5.0
        if L<=65: return 6.0
        return 7.0
    parting = compute_auto_parting(length) if auto_parting else st.number_input("Parting (mm)", value=5.0, format="%.2f")
    qty = st.number_input("Quantity", value=100, min_value=1, step=1)
    material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
    density = st.number_input("Density", value=mrow["Density (kg/m3)"], format="%.2f")
    mat_price = st.number_input("Material price (₹/kg)", value=mrow["Default Price (₹/kg)"], format="%.2f")
    mass_kg = volume_by_stock(stock_type,diameter,length+parting)*density/1e9
    material_cost = mass_kg*mat_price
    final_price_inr = material_cost
    st.metric("Material kg/pc", f"{mass_kg:.6f}")
    st.metric("Final price / pc (INR)", f"₹ {final_price_inr:.4f}")
