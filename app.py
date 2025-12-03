# app.py
# Fastener Costing App — Dual AI (OpenAI + Gemini) + DB-first fallback
# Requires: streamlit>=1.28.0, pandas, numpy, openpyxl, reportlab, openai, requests
# Install: pip install streamlit pandas numpy openpyxl reportlab openai requests

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os, re, json, datetime
from pathlib import Path

# Optional OpenAI
try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

import requests

st.set_page_config(layout="wide", page_title="Fastener Costing App (Dual AI)")

# ---------- Persistence ----------
DATA_DIR = Path("fastener_data")
DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
VENDOR_CSV = DATA_DIR / "vendor_db.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"

# ---------- Utilities ----------
def area_round(d): return np.pi*(d/2)**2
def area_hex(af): return (3*np.sqrt(3)/2)*(af/2)**2
def area_square(s): return s**2

def volume_by_stock(stock_type, dim, total_length, inner_dim=None, thickness=None, width=None):
    if stock_type=="Round Bar": return area_round(dim)*total_length
    if stock_type=="Hex Bar": return area_hex(dim)*total_length
    if stock_type=="Square Bar": return area_square(dim)*total_length
    if stock_type=="Tube":
        if inner_dim is None or inner_dim==0: return area_round(dim)*total_length
        return (area_round(dim)-area_round(inner_dim))*total_length
    if stock_type=="Sheet/Cold Formed":
        if thickness is None or width is None: return dim*total_length
        return thickness*width*total_length
    return area_round(dim)*total_length

def bytes_to_excel_bytes(dfs):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for n, df in dfs.items():
            df.to_excel(writer, sheet_name=n[:31], index=False)
    return out.getvalue()

def pdf_quote_bytes(costing_summary):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    margin = 40
    y = h - margin
    def write(txt, shift=14, bold=False):
        nonlocal y
        if y < margin+60:
            c.showPage(); y = h - margin
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        c.drawString(margin, y, txt); y -= shift
    write("Quotation / Costing Summary", 18, bold=True)
    write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write("")
    for k,v in costing_summary.items(): write(f"{k}: {v}")
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

def load_or_init_csv(path, cols, default_df=None):
    if path.exists():
        try: return pd.read_csv(path)
        except: pass
    if default_df is not None:
        default_df.to_csv(path,index=False)
        return default_df.copy()
    return pd.DataFrame(columns=cols)

# ---------- Default data ----------
DEFAULT_MATERIALS = pd.DataFrame([
    ["A2 Stainless (304)", 8000, 220],
    ["A4 Stainless (316)", 8020, 300],
    ["Steel (C45 / EN8)", 7850, 80],
    ["Brass (CW614N)", 8530, 450],
    ["Aluminium 6061", 2700, 300],
    ["Copper", 8960, 700],
], columns=["Material","Density (kg/m3)","Default Price (₹/kg)"])

DIN_DEFAULTS = pd.DataFrame([
    ["DIN 933","M8","hex_bolt",8,13,5.3,13,1.25,"Example M8 DIN933"],
    ["DIN 933","M10","hex_bolt",10,16,6.4,17,1.5,"Example M10 DIN933"],
    ["DIN 935","M30","castle_nut",30,48,12,46,2.5,"Example M30 DIN935"],
], columns=["Standard","Size","HeadType","d","dk","k","s","pitch","Notes"])

# ---------- Session State ----------
if "materials_df" not in st.session_state: st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, DEFAULT_MATERIALS.columns.tolist(), default_df=DEFAULT_MATERIALS)
if "vendor_db" not in st.session_state: st.session_state.vendor_db = load_or_init_csv(VENDOR_CSV, ["Vendor","Item/Spec","Unit Price (₹)","Lead Time (days)","Notes"])
if "cost_history" not in st.session_state: st.session_state.cost_history = load_or_init_csv(HISTORY_CSV, ["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])
if "din_db" not in st.session_state: st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DEFAULTS.columns.tolist(), default_df=DIN_DEFAULTS)

# ---------- Helpers ----------
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    cols_lower = {col.lower():col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower: return cols_lower[c.lower()]
    return None

def parse_item_text(item_text):
    s = str(item_text).strip()
    m = re.search(r'((DIN|ISO)\s*\d{2,5})', s, re.IGNORECASE)
    standard = m.group(1).upper().replace(" ","") if m else None
    m2 = re.search(r'(M\d{1,3})(?:x(\d{1,4}))?', s, re.IGNORECASE)
    size = None; length = None
    if m2:
        size = m2.group(1).upper()
        if m2.group(2): length = float(m2.group(2))
    else:
        m3 = re.search(r'(\d{1,3})x(\d{1,4})', s)
        if m3:
            size = f"M{m3.group(1)}"; length=float(m3.group(2))
    return standard, size, length

def build_gpt_prompt(standard,size):
    return f"""
You are a precise engineering assistant. Return EXACTLY one JSON object (no extra text).
Input: standard='{standard}', size='{size}'.
Return keys: standard,size,head_type,d,dk,k,s,pitch,notes.
Numeric keys should be numbers (no units). If unknown, use null.
Example:
{{"standard":"DIN 933","size":"M8","head_type":"hex_bolt","d":8,"dk":13,"k":5.3,"s":13,"pitch":1.25,"notes":"common dims"}}
"""

def get_ai_provider():
    if st.session_state.get("openai_key"): return "openai"
    elif st.session_state.get("gemini_key"): return "gemini"
    return None

def fetch_dimensions_via_gpt(api_key_openai, api_key_gemini, standard, size, model_openai="gpt-4o-mini", model_gemini="gemini-1.5", provider=None):
    safe_dict = {"d":None,"dk":None,"k":None,"s":None,"pitch":None,"HeadType":"","notes":""}
    if provider is None: provider = get_ai_provider()
    if provider is None: return safe_dict
    prompt = build_gpt_prompt(standard,size)
    try:
        text = ""
        if provider=="openai":
            if OpenAI is None or not api_key_openai: return safe_dict
            client = OpenAI(api_key=api_key_openai)
            resp = client.chat.completions.create(model=model_openai, messages=[{"role":"system","content":"You are a precise dimensional assistant. Reply ONLY with JSON."},{"role":"user","content":prompt}], temperature=0.0,max_tokens=400)
            try: text = resp.choices[0].message.content.strip()
            except: text=str(resp.choices[0].message).strip()
        elif provider=="gemini":
            if not api_key_gemini: return safe_dict
            url="https://api.gemini.ai/v1/chat/completions"
            headers={"Authorization":f"Bearer {api_key_gemini}","Content-Type":"application/json"}
            payload={"model":model_gemini,"messages":[{"role":"system","content":"You are a precise dimensional assistant. Reply ONLY with JSON."},{"role":"user","content":prompt}],"temperature":0.0,"max_tokens":400}
            r=requests.post(url,headers=headers,json=payload,timeout=20); r.raise_for_status()
            text=r.json().get("choices",[{}])[0].get("message",{}).get("content","").strip()
        if not text: return safe_dict
        text=re.sub(r"^```json|```$","",text,flags=re.IGNORECASE).strip()
        match=re.search(r'(\{.*\})',text,re.DOTALL)
        if match: text=match.group(1)
        data=json.loads(text)
        for k in ["d","dk","k","s","pitch"]:
            if k in data:
                try: data[k]=float(data[k]) if data[k] not in (None,"") else None
                except: data[k]=None
        for k in safe_dict.keys():
            if k not in data: data[k]=safe_dict[k]
        return data
    except Exception as e:
        st.warning(f"{provider.upper()} GPT fetch error: {e}")
        return safe_dict

# ---------- Sidebar ----------
st.sidebar.header("AI Settings")
st.session_state["openai_key"] = st.sidebar.text_input("OpenAI API Key", type="password")
st.session_state["gemini_key"] = st.sidebar.text_input("Gemini AI Key", type="password")
use_gpt_fallback = st.sidebar.checkbox("Enable GPT fallback (DB first)", value=True if st.session_state["openai_key"] or st.session_state["gemini_key"] else False)

# ---------- Machine rates, exchange, defaults ----------
st.sidebar.subheader("Machine Rates (₹/hr)")
if "rates" not in st.session_state:
    st.session_state.rates = {"Traub":700.0,"CNC Turning":650.0,"VMC Milling":600.0,"Drilling":500.0,"Threading":450.0,"Punching":400.0}
for k in list(st.session_state.rates.keys()):
    st.session_state.rates[k] = st.sidebar.number_input(k+" (₹/hr)", value=float(st.session_state.rates[k]), format="%.2f")

st.sidebar.subheader("Exchange Rates")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0)
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0)

st.sidebar.subheader("Defaults")
default_scrap = st.sidebar.number_input("Default scrap (%)", value=2.0)
default_overhead = st.sidebar.number_input("Default overhead (%)", value=10.0)
default_profit = st.sidebar.number_input("Default profit (%)", value=8.0)

# ---------- Tabs ----------
tabs = st.tabs(["Bulk Upload / DIN","Single Calculator","DIN DB","Materials","Vendor DB","History","Trusted Source"])

# ---------- DIN DB tab ----------
with tabs[2]:
    st.header("Local DIN / ISO Dimension DB")
    st.markdown("Edit entries, validate, save.")
    din_df = st.session_state.din_db.copy()
    edited = st.data_editor(din_df,num_rows="dynamic")
    st.session_state.din_db = edited
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Save DIN DB"): st.session_state.din_db.to_csv(DIN_DB_CSV,index=False); st.success("Saved DIN DB")
    with c2:
        if st.button("Reload DIN DB"): st.session_state.din_db = load_or_init_csv(DIN_DB_CSV,DIN_DEFAULTS.columns.tolist(),default_df=DIN_DEFAULTS); st.success("Reloaded DIN DB")

# ---------- Materials tab ----------
with tabs[3]:
    st.header("Materials")
    mat_df = st.session_state.materials_df.copy()
    edited_mat = st.data_editor(mat_df,num_rows="dynamic")
    st.session_state.materials_df = edited_mat
    m1,m2 = st.columns(2)
    with m1:
        if st.button("Save Materials"): st.session_state.materials_df.to_csv(MATERIALS_CSV,index=False); st.success("Materials saved")
    with m2:
        if st.button("Reload Materials"): st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV,DEFAULT_MATERIALS.columns.tolist(),default_df=DEFAULT_MATERIALS); st.success("Reloaded Materials")

# ---------- Vendor DB tab ----------
with tabs[4]:
    st.header("Vendor DB")
    vdf = st.session_state.vendor_db.copy()
    edited_v = st.data_editor(vdf,num_rows="dynamic")
    st.session_state.vendor_db = edited_v
    if st.button("Save Vendor DB"): st.session_state.vendor_db.to_csv(VENDOR_CSV,index=False); st.success("Vendor DB saved")

# ---------- History tab ----------
with tabs[5]:
    st.header("Costing History")
    hdf = st.session_state.cost_history.copy()
    edited_h = st.data_editor(hdf,num_rows="dynamic")
    st.session_state.cost_history = edited_h
    if st.button("Save History"): st.session_state.cost_history.to_csv(HISTORY_CSV,index=False); st.success("History saved")

# ---------- Trusted Source tab ----------
with tabs[6]:
    st.header("Trusted Source Import")
    st.markdown("Upload a CSV of standard dims (columns: Standard,Size,HeadType,d,dk,k,s,pitch,Notes)")
    trusted_file = st.file_uploader("Upload trusted CSV", type=["csv","xlsx"])
    if trusted_file:
        try:
            if trusted_file.name.lower().endswith(".csv"): trusted_df=pd.read_csv(trusted_file)
            else: trusted_df=pd.read_excel(trusted_file)
            st.subheader("Preview uploaded data")
            st.dataframe(trusted_df.head())
            if st.button("Merge into local DIN DB"):
                local = st.session_state.din_db.copy(); added=0
                for _, r in trusted_df.iterrows():
                    std, sz = str(r.get("Standard","")).strip(), str(r.get("Size","")).strip()
                    if std=="" or sz=="": continue
                    exists = local[(local["Standard"].str.upper()==std.upper()) & (local["Size"].str.upper()==sz.upper())]
                    if exists.empty: local = local.append({c:r.get(c,None) for c in local.columns}, ignore_index=True); added+=1
                st.session_state.din_db = local; st.session_state.din_db.to_csv(DIN_DB_CSV,index=False)
                st.success(f"Added {added} new rows.")
        except Exception as e: st.error(f"Failed to read uploaded file: {e}")

# ---------- Bulk Upload / DIN Search tab ----------
with tabs[0]:
    st.header("Bulk Upload / DIN Search")
    st.markdown("Upload CSV/XLSX with 'Item' column like 'DIN 933 M8x20'. Optional: Qty, Material, OverrideDiameter, OverrideLength, Parting.")
    uploaded = st.file_uploader("Upload items file", type=["csv","xlsx"])
    if st.button("Download sample file"):
        sample_df = pd.DataFrame({"Item":["DIN 933 M8x20","DIN 934 M10","DIN 935 M30"], "Qty":[100,200,50], "Material":["A2 Stainless (304)","Steel (C45 / EN8)","A2 Stainless (304)"]})
        st.download_button("Download sample", bytes_to_excel_bytes({"sample":sample_df}), file_name="sample_items.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if uploaded:
        try: df_items = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e: st.error(f"Failed to read file: {e}"); df_items=None
        if df_items is not None:
            st.write("Preview (editable):")
            df_items = st.data_editor(df_items,num_rows="dynamic")
            proposals=[]; unmatched=[]; matched_rows=[]
            for idx,row in df_items.iterrows():
                item_text = str(row.get("Item","")).strip()
                if item_text=="": continue
                qty = int(row.get("Qty",1)) if not pd.isna(row.get("Qty",1)) else 1
                material=row.get("Material",None)
                override_d=row.get("OverrideDiameter",None) if "OverrideDiameter" in df_items.columns else None
                override_l=row.get("OverrideLength",None) if "OverrideLength" in df_items.columns else None
                standard,size,length=parse_item_text(item_text)
                if override_l and not pd.isna(override_l): length=float(override_l)
                found=None
                if standard and size:
                    db=st.session_state.din_db
                    std_col=find_column(db,["Standard","DIN","Std"])
                    size_col=find_column(db,["Size","ThreadSize","S"])
                    if std_col and size_col:
                        match=db[(db[std_col].astype(str).str.upper()==standard.upper()) & (db[size_col].astype(str).str.upper()==size.upper())]
                        if not match.empty:
                            row0=match.iloc[0].to_dict()
                            def get_field(dct,names): 
                                for nm in names:
                                    if nm in dct and pd.notna(dct[nm]): return dct[nm]
                                return None
                            found={"d":get_field(row0,["d","D","nominal","d1"]),"dk":get_field(row0,["dk","HeadDia","Head_Dia"]), "k":get_field(row0,["k","height","thickness"]), "s":get_field(row0,["s","flat","AF"]), "pitch":get_field(row0,["pitch","P"]), "head_type":get_field(row0,["HeadType","head"]), "notes":get_field(row0,["Notes","comments"])}
                if not found and use_gpt_fallback and standard and size:
                    found = fetch_dimensions_via_gpt(st.session_state.openai_key, st.session_state.gemini_key, standard, size)
                if not found: unmatched.append(item_text); continue
                if override_d and not pd.isna(override_d): found["d"]=float(override_d)
                proposal={"Item":item_text,"Qty":qty,"Material":material,"d":found.get("d"),"dk":found.get("dk"),"k":found.get("k"),"s":found.get("s"),"pitch":found.get("pitch"),"HeadType":found.get("head_type"),"Notes":found.get("notes")}
                proposals.append(proposal)
            if proposals:
                out_df = pd.DataFrame(proposals)
                st.subheader("Proposals / Costing-ready")
                st.dataframe(out_df)
                excel_bytes = bytes_to_excel_bytes({"Costing":out_df})
                st.download_button("Download proposals as Excel", excel_bytes, file_name="proposals.xlsx")
            if unmatched:
                st.warning(f"Unmatched items (no DB or GPT data): {unmatched}")

# ---------- Single Calculator tab ----------
with tabs[1]:
    st.header("Single Part Costing Calculator")
    c1,c2 = st.columns(2)
    with c1:
        part_desc = st.text_input("Part Description / Item")
        qty = st.number_input("Quantity", value=1, min_value=1)
        material_sel = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
        stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
        diameter = st.number_input("Diameter / Outer Dimension (mm)", value=10.0)
        length = st.number_input("Length (mm)", value=20.0)
        parting = st.number_input("Parting Loss (mm)", value=1.0)
    with c2:
        inner_diameter = st.number_input("Inner diameter (if Tube, mm)", value=0.0)
        thickness = st.number_input("Thickness (if Sheet, mm)", value=0.0)
        width = st.number_input("Width (if Sheet, mm)", value=0.0)
    if st.button("Compute Cost"):
        mat_row = st.session_state.materials_df[st.session_state.materials_df["Material"]==material_sel].iloc[0]
        density = mat_row["Density (kg/m3)"]
        price_kg = mat_row["Default Price (₹/kg)"]
        volume_m3 = volume_by_stock(stock_type,diameter/1000,length/1000,inner_dim=inner_diameter/1000,thickness=thickness/1000,width=width/1000)
        mass_kg = volume_m3*density
        raw_cost = mass_kg*price_kg
        overhead = raw_cost*default_overhead/100
        profit = raw_cost*default_profit/100
        total_cost = raw_cost + overhead + profit
        st.success(f"Raw Material: ₹{raw_cost:.2f} | Total (with OH+Profit): ₹{total_cost:.2f}")

