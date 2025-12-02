# app.py
# Full End-to-End Fastener Costing App
# Streamlit >=1.28 required

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import datetime

# Optional GPT
try:
    import openai
except:
    openai = None

st.set_page_config(layout="wide", page_title="Fastener Costing App")

# -------------------- Paths --------------------
DATA_DIR = Path("fastener_data")
DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
VENDOR_CSV = DATA_DIR / "vendor_db.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"

# -------------------- Helper Functions --------------------
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

def compute_auto_parting(L):
    if L<=25: return 3.0
    if L<=35: return 4.0
    if L<=50: return 5.0
    if L<=65: return 6.0
    return 7.0

def load_or_init_csv(path, cols, default_df=None):
    if path.exists():
        try: return pd.read_csv(path)
        except: pass
    if default_df is not None:
        default_df.to_csv(path,index=False)
        return default_df.copy()
    return pd.DataFrame(columns=cols)

def pdf_quote_bytes(cost_summary):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    margin = 40
    y = h - margin
    def write(txt, shift=14, bold=False):
        nonlocal y
        if y < margin+60: c.showPage(); y=h-margin
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        c.drawString(margin, y, txt)
        y -= shift
    write("Quotation / Costing Summary",18,True)
    write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write("")
    for k,v in cost_summary.items(): write(f"{k}: {v}")
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

def query_gpt_for_din(standard, size, head_type, key):
    if not openai or key.strip()=="":
        return None
    openai.api_key = key.strip()
    prompt = f"""
    Provide the standard dimensions for {standard} {size} {head_type} fastener.
    Return as CSV: d,dk,k,s,pitch. Only numbers, no units or extra text.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-5-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        text = response['choices'][0]['message']['content']
        nums = [float(x.strip()) for x in text.replace("\n",",").split(",") if x.strip()]
        if len(nums)==5: return nums
    except Exception as e:
        st.warning(f"GPT lookup failed: {e}")
    return None

# -------------------- Default Data --------------------
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

# -------------------- Session State --------------------
for key, val in {
    "materials_df": (MATERIALS_CSV, DEFAULT_MATERIALS),
    "vendor_db": (VENDOR_CSV, pd.DataFrame(columns=["Vendor","Item/Spec","Unit Price (₹)","Lead Time (days)","Notes"])),
    "cost_history": (HISTORY_CSV, pd.DataFrame(columns=["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])),
    "din_db": (DIN_DB_CSV, DIN_DEFAULTS)
}.items():
    if key not in st.session_state:
        st.session_state[key] = load_or_init_csv(val[0], val[1].columns.tolist(), default_df=val[1])

# -------------------- Sidebar --------------------
st.sidebar.header("Settings & Rates")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0, format="%.2f")
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0, format="%.2f")
default_scrap = st.sidebar.number_input("Scrap (%)", value=2.0, format="%.2f")
default_overhead = st.sidebar.number_input("Overhead (%)", value=10.0, format="%.2f")
default_profit = st.sidebar.number_input("Profit (%)", value=8.0, format="%.2f")

# Machining rates
operation_rates = {}
for op in ["Traub","Milling","Threading","Punching","Tooling"]:
    operation_rates[op] = st.sidebar.number_input(f"{op} rate (₹/hr)", value=500.0, format="%.2f")

# -------------------- Tabs --------------------
tabs = st.tabs(["Bulk / DIN","Single Calculator","DIN DB","Materials","Vendor DB","History","Trusted Source"])

# -------------------- DIN DB --------------------
with tabs[2]:
    st.header("Local DIN / ISO DB")
    din_df = st.session_state.din_db.copy()
    edited = st.data_editor(din_df, num_rows="dynamic")
    st.session_state.din_db = edited
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Save DIN DB"): st.session_state.din_db.to_csv(DIN_DB_CSV,index=False); st.success("Saved DIN DB")
    with c2:
        if st.button("Reload DIN DB"): st.session_state.din_db = load_or_init_csv(DIN_DB_CSV,DIN_DEFAULTS.columns.tolist(),default_df=DIN_DEFAULTS); st.success("Reloaded DIN DB")

# -------------------- Materials --------------------
with tabs[3]:
    st.header("Materials")
    mat_df = st.session_state.materials_df.copy()
    edited_mat = st.data_editor(mat_df, num_rows="dynamic")
    st.session_state.materials_df = edited_mat
    m1,m2 = st.columns(2)
    with m1:
        if st.button("Save Materials"): st.session_state.materials_df.to_csv(MATERIALS_CSV,index=False); st.success("Materials saved")
    with m2:
        if st.button("Reload Materials"): st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV,DEFAULT_MATERIALS.columns.tolist(),default_df=DEFAULT_MATERIALS); st.success("Materials reloaded")

# -------------------- Vendor DB --------------------
with tabs[4]:
    st.header("Vendor DB")
    vdf = st.session_state.vendor_db.copy()
    edited_v = st.data_editor(vdf,num_rows="dynamic")
    st.session_state.vendor_db = edited_v
    if st.button("Save Vendor DB"): st.session_state.vendor_db.to_csv(VENDOR_CSV,index=False); st.success("Saved Vendor DB")

# -------------------- History --------------------
with tabs[5]:
    st.header("Costing History")
    hdf = st.session_state.cost_history.copy()
    edited_h = st.data_editor(hdf,num_rows="dynamic")
    st.session_state.cost_history = edited_h
    if st.button("Save History"): st.session_state.cost_history.to_csv(HISTORY_CSV,index=False); st.success("Saved history")

# -------------------- Trusted Source --------------------
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

# -------------------- Single Calculator --------------------
with tabs[1]:
    st.header("Single Item Calculator")
    stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    diameter = st.number_input("Diameter / AF / Side / OD (mm)", value=30.0, format="%.2f")
    length = st.number_input("Length (mm)", value=50.0, format="%.2f")
    auto_parting = st.checkbox("Auto parting", True)
    parting = compute_auto_parting(length) if auto_parting else st.number_input("Parting (mm)", value=5.0, format="%.2f")
    qty = st.number_input("Quantity", value=100, min_value=1, step=1)
    material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
    density = st.number_input("Density (kg/m3)", value=mrow["Density (kg/m3)"], format="%.2f")
    mat_price = st.number_input("Material price (₹/kg)", value=mrow["Default Price (₹/kg)"], format="%.2f")
    
    # Material calculation
    mass_kg = volume_by_stock(stock_type, diameter, length+parting) * density/1e9
    material_cost = mass_kg * mat_price
    
    # Operations (editable)
    traub_cost = st.number_input("Traub cost per piece (₹)", value=material_cost*0.1, format="%.2f")
    milling_cost = st.number_input("Milling cost per piece (₹)", value=material_cost*0.05, format="%.2f")
    threading_cost = st.number_input("Threading cost per piece (₹)", value=material_cost*0.05, format="%.2f")
    punching_cost = st.number_input("Punching cost per piece (₹)", value=material_cost*0.02, format="%.2f")
    tooling_cost = st.number_input("Tooling cost per piece (₹)", value=material_cost*0.01, format="%.2f")
    
    subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_cost
    subtotal *= (1 + default_scrap/100)
    subtotal *= (1 + default_overhead/100)
    subtotal *= (1 + default_profit/100)
    
    st.metric("Material kg/pc", f"{mass_kg:.6f}")
    st.metric("Total per piece (INR)", f"₹ {subtotal:.2f}")
    st.metric("Total batch cost (INR)", f"₹ {subtotal*qty:.2f}")
    
    # PDF quotation
    if st.button("Generate Quotation PDF"):
        summary = {
            "Stock Type": stock_type,
            "Diameter": diameter,
            "Length": length,
            "Parting": parting,
            "Quantity": qty,
            "Material": material,
            "Material Cost/pc (₹)": f"{material_cost:.2f}",
            "Traub (₹)": f"{traub_cost:.2f}",
            "Milling (₹)": f"{milling_cost:.2f}",
            "Threading (₹)": f"{threading_cost:.2f}",
            "Punching (₹)": f"{punching_cost:.2f}",
            "Tooling (₹)": f"{tooling_cost:.2f}",
            "Total per piece (₹)": f"{subtotal:.2f}",
            "Total batch cost (₹)": f"{subtotal*qty:.2f}",
            "USD Rate": usd_rate,
            "EUR Rate": eur_rate
        }
        pdf_bytes = pdf_quote_bytes(summary)
        st.download_button("Download Quotation PDF", data=pdf_bytes, file_name="quotation.pdf", mime="application/pdf")

# -------------------- Bulk / DIN Batch Costing --------------------
with tabs[0]:
    st.header("Bulk / DIN Batch Costing")
    din_df = st.session_state.din_db.copy()
    edited = st.data_editor(din_df, num_rows="dynamic")
    st.session_state.din_db = edited

    openai_key = st.text_input("OpenAI API Key (for GPT lookup)", type="password")
    
    stock_type_bulk = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    material_bulk = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow_bulk = st.session_state.materials_df[st.session_state.materials_df["Material"]==material_bulk].iloc[0]
    density_bulk = st.number_input("Density (kg/m3)", value=mrow_bulk["Density (kg/m3)"], format="%.2f")
    mat_price_bulk = st.number_input("Material price (₹/kg)", value=mrow_bulk["Default Price (₹/kg)"], format="%.2f")
    qty_bulk = st.number_input("Quantity per item", value=100, min_value=1, step=1)
    
    auto_parting_bulk = st.checkbox("Auto parting", True)
    
    if st.button("Auto-fill missing dimensions & calculate costs"):
        results = []
        updated_rows = 0
        for idx, row in st.session_state.din_db.iterrows():
            # GPT auto-fill missing dimensions
            missing = any(pd.isna(row[c]) for c in ["d","dk","k","s","pitch"])
            if missing and openai_key.strip()!="":
                dims = query_gpt_for_din(row["Standard"], row["Size"], row["HeadType"], openai_key)
                if dims:
                    st.session_state.din_db.at[idx,"d"] = dims[0]
                    st.session_state.din_db.at[idx,"dk"] = dims[1]
                    st.session_state.din_db.at[idx,"k"] = dims[2]
                    st.session_state.din_db.at[idx,"s"] = dims[3]
                    st.session_state.din_db.at[idx,"pitch"] = dims[4]
                    updated_rows += 1

            # Parting
            L = row.get("Length",50.0)
            parting = compute_auto_parting(L) if auto_parting_bulk else row.get("Parting",5.0)
            
            # Material cost
            diameter = row.get("d", 10.0)
            length = row.get("Length",50.0)
            mass_kg = volume_by_stock(stock_type_bulk, diameter, length+parting)*density_bulk/1e9
            material_cost = mass_kg * mat_price_bulk
            
            # Default operations cost (editable)
            traub_cost = material_cost*0.1
            milling_cost = material_cost*0.05
            threading_cost = material_cost*0.05
            punching_cost = material_cost*0.02
            tooling_cost = material_cost*0.01
            
            subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_cost
            subtotal *= (1 + default_scrap/100)*(1 + default_overhead/100)*(1 + default_profit/100)
            
            results.append({
                "PartDesc": f"{row['Standard']} {row['Size']}",
                "Material": material_bulk,
                "Diameter(mm)": diameter,
                "Length(mm)": length,
                "Parting(mm)": parting,
                "MaterialCost_INR": round(material_cost,2),
                "Traub": round(traub_cost,2),
                "Milling": round(milling_cost,2),
                "Threading": round(threading_cost,2),
                "Punching": round(punching_cost,2),
                "Tooling": round(tooling_cost,2),
                "UnitPrice_INR": round(subtotal,2),
                "
