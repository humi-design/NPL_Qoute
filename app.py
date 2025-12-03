# app.py
# Fastener Costing App — Patched for new OpenAI client + "Database first then GPT" fallback (Option A)
# Requires: streamlit>=1.28.0, pandas, numpy, openpyxl, reportlab, openai
# Install: pip install streamlit pandas numpy openpyxl reportlab openai

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os, re, json, datetime
from pathlib import Path

# Optional OpenAI library (we will import client only if available)
try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

st.set_page_config(layout="wide", page_title="Fastener Costing App (DB-first + GPT fallback)")

# ---------- Persistence ----------
DATA_DIR = Path("fastener_data")
DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
VENDOR_CSV = DATA_DIR / "vendor_db.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"  # local DIN/ISO dimension DB

# ---------- Utilities & calc helpers ----------
def area_round(d): return np.pi * (d/2)**2
def area_hex(af): return (3*np.sqrt(3)/2) * (af/2)**2
def area_square(s): return s**2

def volume_by_stock(stock_type, dim, total_length, inner_dim=None, thickness=None, width=None):
    if stock_type == "Round Bar":
        return area_round(dim) * total_length
    if stock_type == "Hex Bar":
        return area_hex(dim) * total_length
    if stock_type == "Square Bar":
        return area_square(dim) * total_length
    if stock_type == "Tube":
        if inner_dim is None or inner_dim == 0:
            return area_round(dim) * total_length
        return (area_round(dim) - area_round(inner_dim)) * total_length
    if stock_type == "Sheet/Cold Formed":
        if thickness is None or width is None:
            return dim * total_length
        return thickness * width * total_length
    return area_round(dim) * total_length

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

# ---------- load / init CSV helpers ----------
def load_or_init_csv(path, cols, default_df=None):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    if default_df is not None:
        default_df.to_csv(path, index=False)
        return default_df.copy()
    return pd.DataFrame(columns=cols)

# ---------- default data ----------
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

# ---------- session state init ----------
if "materials_df" not in st.session_state:
    st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, DEFAULT_MATERIALS.columns.tolist(), default_df=DEFAULT_MATERIALS)
if "vendor_db" not in st.session_state:
    st.session_state.vendor_db = load_or_init_csv(VENDOR_CSV, ["Vendor","Item/Spec","Unit Price (₹)","Lead Time (days)","Notes"])
if "cost_history" not in st.session_state:
    st.session_state.cost_history = load_or_init_csv(HISTORY_CSV, ["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])
if "din_db" not in st.session_state:
    st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DEFAULTS.columns.tolist(), default_df=DIN_DEFAULTS)

# ---------- safe column detector ----------
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive search
    cols_lower = {col.lower():col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

# ---------- parse / GPT helpers ----------
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
            size = f"M{m3.group(1)}"; length = float(m3.group(2))
    return standard, size, length

def build_gpt_prompt(standard, size):
    return f"""
You are a precise engineering assistant. Return EXACTLY one JSON object (no extra text).
Input: standard='{standard}', size='{size}'.
Return keys: standard,size,head_type,d,dk,k,s,pitch,notes.
Numeric keys should be numbers (no units). If unknown, use null.
Example:
{{"standard":"DIN 933","size":"M8","head_type":"hex_bolt","d":8,"dk":13,"k":5.3,"s":13,"pitch":1.25,"notes":"common dims"}}
"""

def fetch_dimensions_via_gpt(api_key, standard, size, model="gpt-4o-mini"):
    """
    Uses the new OpenAI client (OpenAI) to request JSON. Returns dict or None.
    """
    if OpenAI is None:
        raise RuntimeError("openai library not installed")
    client = OpenAI(api_key=api_key)
    prompt = build_gpt_prompt(standard, size)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise dimensional assistant. Reply ONLY with JSON."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        # newer client returns choices list; extract safely
        # text extraction supports both .message.content and legacy structure
        text = None
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            # fallback if different shape
            text = str(resp.choices[0].message).strip()
        if not text:
            st.error("GPT returned empty response.")
            return None
        json_text = re.sub(r"^```json|```$", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r'(\{.*\})', json_text, re.DOTALL)
        if match: json_text = match.group(1)
        data = json.loads(json_text)
        for k in ["d","dk","k","s","pitch"]:
            if k in data:
                try: data[k] = float(data[k]) if data[k] is not None else None
                except: data[k] = None
        return data
    except Exception as e:
        st.error(f"GPT fetch error: {e}")
        return None

# ---------- validation routine ----------
def validate_dimensions_row(row):
    warnings = []
    try:
        d = float(row.get("d")) if row.get("d") not in (None,"") else None
    except:
        d = None
    if d is None:
        warnings.append("Missing nominal d (thread diameter) — cannot validate")
        return warnings
    dk = row.get("dk")
    if dk not in (None,"") and pd.notna(dk):
        dk = float(dk)
        if dk < 1.3*d:
            warnings.append(f"dk too small: {dk} mm (expected >= {1.3*d:.2f})")
        if dk > 2.5*d:
            warnings.append(f"dk unusually large: {dk} mm (expected <= {2.5*d:.2f})")
    s = row.get("s")
    if s not in (None,"") and pd.notna(s):
        s = float(s)
        if s < 1.3*d:
            warnings.append(f"across flats (s) too small: {s} mm (expected >= {1.3*d:.2f})")
    k = row.get("k")
    if k not in (None,"") and pd.notna(k):
        k = float(k)
        if k < 0.3*d:
            warnings.append(f"head height k small: {k} mm (expected >= {0.3*d:.2f})")
        if k > 1.5*d:
            warnings.append(f"head height k large: {k} mm (expected <= {1.5*d:.2f})")
    return warnings

# ---------- Sidebar & settings ----------
st.sidebar.header("Settings & OpenAI")
api_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")
use_gpt_fallback = st.sidebar.checkbox("Enable GPT fallback (Option A: DB first)", value=True if api_key else False)

# test API key button
if st.sidebar.button("Test OpenAI API Key"):
    if not api_key:
        st.sidebar.error("No API key provided.")
    elif OpenAI is None:
        st.sidebar.error("openai library not installed in environment.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            # quick chat ping
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are a heartbeat responder. Reply 'pong'."},{"role":"user","content":"ping"}], temperature=0.0, max_tokens=10)
            text = None
            try:
                text = resp.choices[0].message.content.strip()
            except Exception:
                text = str(resp)
            st.sidebar.success(f"API OK — sample response: {text}")
        except Exception as e:
            st.sidebar.error(f"API test failed: {e}")

st.sidebar.subheader("Machine Rates (₹/hr)")
if "rates" not in st.session_state:
    st.session_state.rates = {"Traub":700.0,"CNC Turning":650.0,"VMC Milling":600.0,"Drilling":500.0,"Threading":450.0,"Punching":400.0}
for k in list(st.session_state.rates.keys()):
    st.session_state.rates[k] = st.sidebar.number_input(k+" (₹/hr)", value=float(st.session_state.rates[k]), format="%.2f")

st.sidebar.subheader("Exchange Rates (INR per 1 unit)")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0)
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0)

st.sidebar.subheader("Defaults")
default_scrap = st.sidebar.number_input("Default scrap (%)", value=2.0)
default_overhead = st.sidebar.number_input("Default overhead (%)", value=10.0)
default_profit = st.sidebar.number_input("Default profit (%)", value=8.0)

# ---------- Tabs ----------
tabs = st.tabs(["Bulk Upload / DIN", "Single Calculator", "DIN DB", "Materials", "Vendor DB", "History", "Trusted Source"])

# ---------- DIN DB tab ----------
with tabs[2]:
    st.header("Local DIN / ISO Dimension DB")
    st.markdown("Edit entries, validate, save. This DB is used to auto-fill dimensions for parsed items.")
    din_df = st.session_state.din_db.copy()
    edited = st.data_editor(din_df, num_rows="dynamic")
    st.session_state.din_db = edited
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Save DIN DB"):
            st.session_state.din_db.to_csv(DIN_DB_CSV, index=False)
            st.success(f"Saved DIN DB to {DIN_DB_CSV}")
    with c2:
        if st.button("Reload DIN DB"):
            st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DEFAULTS.columns.tolist(), default_df=DIN_DEFAULTS)
            st.success("Reloaded DIN DB")

# ---------- Materials tab ----------
with tabs[3]:
    st.header("Materials (editable)")
    mat_df = st.session_state.materials_df.copy()
    edited_mat = st.data_editor(mat_df, num_rows="dynamic")
    st.session_state.materials_df = edited_mat
    m1,m2 = st.columns(2)
    with m1:
        if st.button("Save Materials"):
            st.session_state.materials_df.to_csv(MATERIALS_CSV, index=False); st.success("Materials saved")
    with m2:
        if st.button("Reload Materials"):
            st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, DEFAULT_MATERIALS.columns.tolist(), default_df=DEFAULT_MATERIALS)
            st.success("Materials reloaded")

# ---------- Vendor DB tab ----------
with tabs[4]:
    st.header("Vendor DB")
    vdf = st.session_state.vendor_db.copy()
    edited_v = st.data_editor(vdf, num_rows="dynamic")
    st.session_state.vendor_db = edited_v
    if st.button("Save Vendor DB"):
        st.session_state.vendor_db.to_csv(VENDOR_CSV, index=False); st.success("Vendor DB saved")

# ---------- History tab ----------
with tabs[5]:
    st.header("Costing History")
    hdf = st.session_state.cost_history.copy()
    edited_h = st.data_editor(hdf, num_rows="dynamic")
    st.session_state.cost_history = edited_h
    if st.button("Save History"):
        st.session_state.cost_history.to_csv(HISTORY_CSV, index=False); st.success("History saved")

# ---------- Trusted source tab ----------
with tabs[6]:
    st.header("Trusted Source Import (upload your authoritative CSV)")
    st.markdown("Upload a CSV of standard dims (columns: Standard,Size,HeadType,d,dk,k,s,pitch,Notes). The app will preview and allow merging into local DB.")
    trusted_file = st.file_uploader("Upload trusted CSV", type=["csv","xlsx"])
    if trusted_file:
        try:
            if trusted_file.name.lower().endswith(".csv"):
                trusted_df = pd.read_csv(trusted_file)
            else:
                trusted_df = pd.read_excel(trusted_file)
            st.subheader("Preview uploaded data")
            st.dataframe(trusted_df.head())
            merge = st.button("Merge into local DIN DB (preview shows conflicts)")
            if merge:
                local = st.session_state.din_db.copy()
                added = 0
                for _, r in trusted_df.iterrows():
                    std = str(r.get("Standard","")).strip()
                    sz = str(r.get("Size","")).strip()
                    if std=="" or sz=="":
                        continue
                    exists = local[(local["Standard"].str.upper()==std.upper()) & (local["Size"].str.upper()==sz.upper())]
                    if exists.empty:
                        local = local.append({c:r.get(c,None) for c in local.columns}, ignore_index=True)
                        added += 1
                st.session_state.din_db = local
                st.session_state.din_db.to_csv(DIN_DB_CSV, index=False)
                st.success(f"Merged trusted file into DIN DB — added {added} new rows.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# ---------- Bulk Upload / DIN Search tab ----------
with tabs[0]:
    st.header("Bulk Upload / DIN Search (with GPT fallback & batch add)")
    st.markdown("Upload CSV/XLSX with a column named 'Item' containing strings like 'DIN 933 M8x20'. Optional columns: Qty, Material, OverrideDiameter, OverrideLength, Parting.")

    uploaded = st.file_uploader("Upload items file", type=["csv","xlsx"])
    sample = st.button("Download sample file")
    if sample:
        sample_df = pd.DataFrame({
            "Item":["DIN 933 M8x20","DIN 934 M10","DIN 935 M30"],
            "Qty":[100,200,50],
            "Material":["A2 Stainless (304)","Steel (C45 / EN8)","A2 Stainless (304)"]
        })
        st.download_button("Download sample", bytes_to_excel_bytes({"sample":sample_df}), 
                           file_name="sample_items.xlsx", 
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_items = pd.read_csv(uploaded)
            else:
                df_items = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df_items = None

        if df_items is not None:
            st.write("Preview (editable):")
            df_items = st.data_editor(df_items, num_rows="dynamic")

            # ---------- helpers ----------
            def safe_float(val, default=0.0):
                try:
                    if val is None or val=="" or pd.isna(val):
                        return float(default)
                    return float(val)
                except:
                    return float(default)

            def safe_int(val, default=1):
                try:
                    if val is None or val=="" or pd.isna(val):
                        return int(default)
                    return int(val)
                except:
                    return int(default)

            proposals = []
            unmatched = []
            matched_rows = []

            for idx, row in df_items.iterrows():
                item_text = str(row.get("Item","")).strip()
                if item_text == "":
                    continue

                qty = safe_int(row.get("Qty",1))
                material = row.get("Material", None)
                override_d = row.get("OverrideDiameter") if "OverrideDiameter" in df_items.columns else None
                override_l = row.get("OverrideLength") if "OverrideLength" in df_items.columns else None

                standard, size, length = parse_item_text(item_text)
                if override_l not in (None,"") and not pd.isna(override_l):
                    length = safe_float(override_l)

                # ---------- DB-first lookup ----------
                found = None
                if standard and size:
                    db = st.session_state.din_db
                    std_col = find_column(db, ["Standard","DIN","Std"])
                    size_col = find_column(db, ["Size","ThreadSize","S"])
                    if std_col and size_col:
                        match = db[(db[std_col].astype(str).str.upper()==standard.upper()) & 
                                   (db[size_col].astype(str).str.upper()==size.upper())]
                        if not match.empty:
                            row0 = match.iloc[0].to_dict()
                            def get_field(dct, names):
                                for nm in names:
                                    if nm in dct and pd.notna(dct[nm]) and dct[nm]!="":
                                        return safe_float(dct[nm], 0.0)
                                return 0.0
                            found = {
                                "d": get_field(row0, ["d","D","nominal"]),
                                "dk": get_field(row0, ["dk","DK","head_diameter"]),
                                "k": get_field(row0, ["k","K","head_height"]),
                                "s": get_field(row0, ["s","S","across_flats"]),
                                "pitch": get_field(row0, ["pitch","P"]),
                                "HeadType": row0.get("HeadType","")
                            }

                if found:
                    d_final = safe_float(override_d, found["d"])
                    matched_rows.append({
                        "Item":item_text,"Standard":standard,"Size":size,"Qty":qty,"Material":material,
                        "d":d_final,"dk":found["dk"],"k":found["k"],"s":found["s"],"pitch":found["pitch"],"Length":safe_float(length)
                    })
                else:
                    # ---------- GPT fallback ----------
                    if use_gpt_fallback and api_key:
                        st.info(f"Requesting GPT dims for {standard or 'UnknownStd'} {size or ''}")
                        data = fetch_dimensions_via_gpt(api_key, standard, size)
                        if data:
                            warnings = validate_dimensions_row(data)
                            proposals.append({
                                "Item":item_text,"Standard":standard,"Size":size,"Qty":qty,"Material":material,
                                "d":safe_float(data.get("d")), "dk":safe_float(data.get("dk")),
                                "k":safe_float(data.get("k")), "s":safe_float(data.get("s")),
                                "pitch":safe_float(data.get("pitch")), "notes":data.get("notes",""),
                                "warnings":"; ".join(warnings)
                            })
                        else:
                            unmatched.append({"Item":item_text,"Reason":"GPT failed or returned nothing"})
                    else:
                        unmatched.append({"Item":item_text,"Reason":"Missing in local DB"})

            # ---------- display ----------
            if matched_rows:
                st.subheader("Matched items (from local DB)")
                st.dataframe(pd.DataFrame(matched_rows))

            if proposals:
                st.subheader("GPT Proposals (review & bulk-add)")
                prop_df = pd.DataFrame(proposals)
                st.dataframe(prop_df)
                st.markdown("Edit proposals below (if needed), then select rows to add to local DB.")
                editable_props = st.data_editor(prop_df, num_rows="dynamic")
                if "Add?" not in editable_props.columns:
                    editable_props["Add?"] = True
                if st.button("Add selected proposals to local DIN DB"):
                    added = 0
                    local = st.session_state.din_db.copy()
                    for _, prow in editable_props.iterrows():
                        if prow.get("Add?", True) in (True, "True", 1, "1"):
                            std = str(prow.get("Standard","")).strip()
                            sz = str(prow.get("Size","")).strip()
                            if std=="" or sz=="":
                                continue
                            exists = local[(local["Standard"].astype(str).str.upper()==std.upper()) & 
                                           (local["Size"].astype(str).str.upper()==sz.upper())]
                            if exists.empty:
                                newrow = {
                                    "Standard":std,"Size":sz,"HeadType":prow.get("HeadType",""), 
                                    "d":safe_float(prow.get("d")),
                                    "dk":safe_float(prow.get("dk")),
                                    "k":safe_float(prow.get("k")),
                                    "s":safe_float(prow.get("s")),
                                    "pitch":safe_float(prow.get("pitch")),
                                    "Notes":prow.get("notes","")
                                }
                                local = local.append(newrow, ignore_index=True)
                                added += 1
                    st.session_state.din_db = local
                    st.session_state.din_db.to_csv(DIN_DB_CSV, index=False)
                    st.success(f"Added {added} proposals to local DIN DB.")

            if unmatched:
                st.subheader("Unmatched items requiring manual action")
                st.dataframe(pd.DataFrame(unmatched))


# ---------- Single Calculator tab ----------
# ---------- Single Calculator tab ----------
with tabs[1]:
    st.header("Single Item Calculator (full costing, editable)")
    st.markdown("Select a standard from local DB or enter custom dims manually.")

    din_options = st.session_state.din_db.copy()
    # robust detection of standard and size columns
    std_col = find_column(din_options, ["Standard","DIN","Std"])
    size_col = find_column(din_options, ["Size","ThreadSize","S"])
    stds = sorted(din_options[std_col].dropna().astype(str).unique().tolist()) if std_col else []
    chosen = st.selectbox("Choose standard or Custom", ["Custom"] + stds)

    # ---------- Safe numeric extraction ----------
    def safe_float(val, default=0.0):
        try:
            if val is None or val=="" or pd.isna(val):
                return float(default)
            return float(val)
        except:
            return float(default)

    def default_from_size(size_str, fallback=30.0):
        if isinstance(size_str, str) and size_str.upper().startswith("M"):
            try:
                return float(size_str.upper().replace("M",""))
            except:
                return fallback
        return fallback

    if chosen != "Custom" and std_col and size_col:
        sizes = sorted(din_options[din_options[std_col].astype(str)==chosen][size_col].dropna().astype(str).unique().tolist())
        chosen_size = st.selectbox("Size", sizes)
        row = din_options[(din_options[std_col].astype(str)==chosen) & (din_options[size_col].astype(str)==chosen_size)].iloc[0]

        def g(r, names):
            for nm in names:
                if nm in r and pd.notna(r[nm]):
                    return r[nm]
            return None

        pre_d = safe_float(g(row, ["d","D","nominal"]), default=default_from_size(chosen_size))
        pre_dk = safe_float(g(row, ["dk","DK","head_diameter"]), default=0.0)
        pre_k = safe_float(g(row, ["k","K","head_height"]), default=0.0)
        pre_s = safe_float(g(row, ["s","S","across_flats"]), default=0.0)
    else:
        # Custom or fallback values
        chosen_size = "M30"
        pre_d = 30.0
        pre_dk = 0.0
        pre_k = 0.0
        pre_s = 0.0

    # ---------- UI Inputs ----------
    col1,col2 = st.columns([2,1])
    with col1:
        stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
        diameter = st.number_input("Diameter / AF / Side / OD (mm)", value=pre_d)
        if stock_type=="Sheet/Cold Formed":
            thickness = st.number_input("Thickness (mm)", value=2.0)
            width = st.number_input("Width (mm)", value=10.0)
        else:
            thickness=None; width=None
        length = st.number_input("Length (mm)", value=50.0)
        auto_parting = st.checkbox("Use automatic parting rule", value=True)
        def compute_auto_parting(L):
            if L <=25: return 3.0
            if L <=35: return 4.0
            if L <=50: return 5.0
            if L <=65: return 6.0
            return 7.0
        if auto_parting:
            parting = compute_auto_parting(length)
            st.write(f"Auto parting: {parting} mm")
        else:
            parting = st.number_input("Parting (mm)", value=compute_auto_parting(length))
        thread = st.text_input("Thread (e.g. M30)", value=chosen_size if chosen!="Custom" else "M30")
        qty = st.number_input("Quantity", value=100, min_value=1)

    with col2:
        material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
        mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
        density = st.number_input("Density (kg/m3)", value=float(mrow["Density (kg/m3)"]))
        mat_price = st.number_input("Material price (₹/kg)", value=float(mrow["Default Price (₹/kg)"]))
        traub_cycle = st.number_input("Traub (sec)", value=max(12,(length+parting)*0.4+12))
        milling_cycle = st.number_input("Milling (sec)", value=max(8,(length+parting)*0.25+8))
        threading_cycle = st.number_input("Threading (sec)", value=20.0)
        punching_cycle = st.number_input("Punching (sec)", value=max(6,6+(length+parting)*0.03))
        tooling_cost = st.number_input("Tooling cost (₹)", value=2000.0)
        run_qty = st.number_input("Tooling run qty", value=5000, min_value=1)
        heat_treat = st.number_input("Heat treat/passivation (₹/pc)", value=0.3)
        plating = st.number_input("Plating (₹/pc)", value=0.0)
        inspection = st.number_input("Inspection/Deburr (₹/pc)", value=0.1)
        packaging = st.number_input("Packaging (₹/pc)", value=0.05)
        labour_add = st.number_input("Labour add-on (₹/pc)", value=0.1)
        scrap_pct = st.number_input("Scrap (%)", value=default_scrap)

    # ---------- Calculations ----------
    total_length = length + parting
    vol_mm3 = volume_by_stock(stock_type, diameter, total_length, thickness=thickness, width=width)
    mass_kg = vol_mm3/1e9 * density
    material_cost = mass_kg * mat_price

    rates = st.session_state.rates
    traub_cost = (rates["Traub"] * traub_cycle)/3600.0
    milling_cost = (rates["VMC Milling"] * milling_cycle)/3600.0
    threading_cost = (rates["Threading"] * threading_cycle)/3600.0
    punching_cost = (rates["Punching"] * punching_cycle)/3600.0
    tooling_per_piece = tooling_cost / run_qty
    other_direct = heat_treat + plating + inspection + packaging + labour_add

    direct_subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_per_piece + other_direct
    direct_with_scrap = direct_subtotal / (1 - scrap_pct/100.0) if (1 - scrap_pct/100.0)!=0 else direct_subtotal
    overhead = direct_with_scrap * (default_overhead/100.0)
    profit_amount = (direct_with_scrap + overhead) * (default_profit/100.0)
    final_price_inr = direct_with_scrap + overhead + profit_amount

    # ---------- Display ----------
    st.subheader("Result")
    st.metric("Material kg/pc", f"{mass_kg:.6f}")
    st.metric("Final price / pc (INR)", f"₹ {final_price_inr:.4f}")
    st.metric("Final price / pc (USD)", f"$ {final_price_inr/usd_rate:.4f}")
    st.metric("Final price / pc (EUR)", f"€ {final_price_inr/eur_rate:.4f}")
    st.write(f"Total for {qty} pcs: ₹ {final_price_inr*qty:.2f} | $ {final_price_inr*qty/usd_rate:.2f} | € {final_price_inr*qty/eur_rate:.2f}")

    if st.button("Save this costing to history"):
        new_row = {
            "Timestamp": datetime.datetime.now().isoformat(),
            "PartDesc": f"{stock_type} {thread}",
            "Qty": int(qty),
            "Material": material,
            "Diameter(mm)": float(diameter),
            "Length(mm)": float(length),
            "Parting(mm)": float(parting),
            "UnitPrice_INR": float(final_price_inr),
            "Total_INR": float(final_price_inr*qty),
            "TargetPrice": "",
            "Notes": ""
        }
        st.session_state.cost_history = st.session_state.cost_history.append(new_row, ignore_index=True)
        st.session_state.cost_history.to_csv(HISTORY_CSV, index=False)
        st.success("Saved to history and persisted.")


# ---------- Export helpers (global) ----------
with st.sidebar:
    st.markdown("---")
    if st.button("Export whole dataset (Materials, DIN DB, Vendor, History)"):
        out = bytes_to_excel_bytes({"Materials":st.session_state.materials_df, "DIN_DB":st.session_state.din_db, "VendorDB":st.session_state.vendor_db, "History":st.session_state.cost_history})
        st.download_button("Download All Data", out, file_name="fastener_data_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.write("App ready — DB-first fallback implemented. Use GPT fallback cautiously and always review before adding to DB.")
