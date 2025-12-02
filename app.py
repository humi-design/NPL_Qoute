# app.py
# Streamlit Fastener Costing App — Final with GPT fallback for DIN dimensions (Option B)
# Requirements: streamlit, pandas, numpy, openpyxl, reportlab, openai
# Install: pip install streamlit pandas numpy openpyxl reportlab openai

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os
import datetime
from pathlib import Path
import re
import json

# Optional OpenAI usage
try:
    import openai
except Exception:
    openai = None

st.set_page_config(layout="wide", page_title="Fastener Costing App (GPT-Fallback)")

# ---------- Persistence ----------
DATA_DIR = Path("fastener_data")
DATA_DIR.mkdir(exist_ok=True)
MATERIALS_CSV = DATA_DIR / "materials.csv"
VENDOR_CSV = DATA_DIR / "vendor_db.csv"
HISTORY_CSV = DATA_DIR / "cost_history.csv"
DIN_DB_CSV = DATA_DIR / "din_dimensions.csv"  # local DIN/ISO dimension DB

# ---------- Utilities ----------
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
        if not inner_dim:
            return area_round(dim) * total_length
        return (area_round(dim) - area_round(inner_dim)) * total_length
    if stock_type == "Sheet/Cold Formed":
        if thickness is None or width is None:
            return dim * total_length
        return thickness * width * total_length
    return area_round(dim) * total_length

def bytes_to_excel_bytes(df_dict):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return out.getvalue()

def pdf_quote_bytes(costing_summary: dict):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    margin = 40
    y = h - margin
    def write(txt, shift=14, bold=False):
        nonlocal y
        if y < margin+60:
            c.showPage()
            y = h - margin
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        c.drawString(margin, y, txt)
        y -= shift
    write("Quotation / Costing Summary", 18, bold=True)
    write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write("")
    for k,v in costing_summary.items():
        write(f"{k}: {v}")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# ---------- Load / init persistent tables ----------
def load_or_init_csv(path, columns, default_df=None):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    if default_df is not None:
        default_df.to_csv(path, index=False)
        return default_df.copy()
    return pd.DataFrame(columns=columns)

DEFAULT_MATERIALS = pd.DataFrame([
    ["A2 Stainless (304)", 8000, 220],
    ["A4 Stainless (316)", 8020, 300],
    ["Steel (C45 / EN8)", 7850, 80],
    ["Brass (CW614N)", 8530, 450],
    ["Aluminium 6061", 2700, 300],
    ["Copper", 8960, 700],
], columns=["Material", "Density (kg/m3)", "Default Price (₹/kg)"])

MATERIALS = load_or_init_csv(MATERIALS_CSV, ["Material","Density (kg/m3)","Default Price (₹/kg)"], default_df=DEFAULT_MATERIALS)
VENDOR_DB = load_or_init_csv(VENDOR_CSV, ["Vendor","Item/Spec","Unit Price (₹)","Lead Time (days)","Notes"])
HISTORY = load_or_init_csv(HISTORY_CSV, ["Timestamp","PartDesc","Qty","Material","Diameter(mm)","Length(mm)","Parting(mm)","UnitPrice_INR","Total_INR","TargetPrice","Notes"])

# DIN DB: columns: Standard, Size (e.g. M8), head_type, d (nominal), dk (head dia), k (head height), s (across flats), pitch, notes
DIN_DEFAULTS = pd.DataFrame([
    # small example rows — you can add more
    ["DIN 933","M8","hex_bolt",8,13,5.3,13,1.25,"Hex bolt full thread - example dims"],
    ["DIN 933","M10","hex_bolt",10,16,6.4,17,1.5,"Hex bolt example"],
    ["DIN 935","M30","castle_nut",30,48,12,46,2.5,"Castle nut example"],
], columns=["Standard","Size","HeadType","d","dk","k","s","pitch","Notes"])
DIN_DB = load_or_init_csv(DIN_DB_CSV, ["Standard","Size","HeadType","d","dk","k","s","pitch","Notes"], default_df=DIN_DEFAULTS)

# ---------- GPT fallback helper ----------
def parse_item_text(item_text):
    """
    Try to parse strings like 'DIN 933 M8x20', 'DIN933 M10 x 30', 'DIN 934 M12' etc.
    Return (standard, size, length_mm or None)
    """
    s = item_text.strip()
    # standard: look for 'DIN' or 'ISO' followed by digits
    m = re.search(r'((DIN|ISO)\s*\d{2,5})', s, re.IGNORECASE)
    standard = m.group(1).upper().replace(" ","") if m else None
    # size: M8 or M8x20 patterns
    m2 = re.search(r'(M\d{1,3})(?:x(\d{1,4}))?', s, re.IGNORECASE)
    size = None; length = None
    if m2:
        size = m2.group(1).upper()
        if m2.group(2):
            length = float(m2.group(2))
    else:
        # maybe given as 8x20 without M
        m3 = re.search(r'(\d{1,3})x(\d{1,4})', s)
        if m3:
            size = f"M{m3.group(1)}"
            length = float(m3.group(2))
    return standard, size, length

def build_gpt_prompt(standard, size):
    """
    Create a prompt instructing GPT to return a JSON with the fields:
    standard, size, head_type, d, dk, k, s, pitch, notes
    All numeric fields should be numbers (no units) or null if not applicable.
    Respond only with JSON.
    """
    prompt = f"""
You are a helpful engineering assistant. Return precise mechanical dimensional data for the requested DIN/ISO fastener.
Respond ONLY with a single JSON object (no additional explanation).

Input:
  standard: "{standard}"
  size: "{size}"

Return JSON with the following keys:
  standard: string (the standard e.g. "DIN 933")
  size: string (e.g. "M8")
  head_type: string (e.g. "hex_bolt", "hex_nut", "socket_cap", "csk", "no_head" etc.)
  d: number (nominal thread diameter in mm) or null
  dk: number (head diameter in mm) or null
  k: number (head height in mm) or null
  s: number (across flats mm) or null
  pitch: number (thread pitch in mm) or null
  notes: string with brief reference or caution

If exact standard data is not available, return best approximate common values and set notes describing source/assumption.

Example response:
{{"standard":"DIN 933","size":"M8","head_type":"hex_bolt","d":8,"dk":13,"k":5.3,"s":13,"pitch":1.25,"notes":"common hex bolt dimensions"}}
"""
    return prompt

def fetch_dimensions_via_gpt(api_key, standard, size, model="gpt-4o-mini" ):
    """
    Use OpenAI (chat completions) to request dimension JSON.
    Returns dict or None.
    """
    if openai is None:
        raise RuntimeError("openai library not installed.")
    openai.api_key = api_key
    prompt = build_gpt_prompt(standard, size)
    try:
        # system + user style via ChatCompletion
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise dimensional assistant. Reply ONLY with JSON."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        text = resp.choices[0].message['content'].strip()
        # Attempt to extract a JSON object from the response
        json_text = text
        # If there is surrounding markdown or backticks, strip them
        json_text = re.sub(r"^```json\\n?|\\n?```$", "", json_text, flags=re.IGNORECASE).strip()
        # sometimes GPT returns text before/after JSON, try to find first {...} block
        match = re.search(r'(\{.*\})', json_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        data = json.loads(json_text)
        # Normalize keys and numeric conversions
        for k in ["d","dk","k","s","pitch"]:
            if k in data and (data[k] is None or data[k]=="" ):
                data[k] = None
            elif k in data:
                try:
                    data[k] = float(data[k])
                except:
                    data[k] = None
        return data
    except Exception as e:
        st.error(f"GPT fetch failed: {e}")
        return None

# ---------- Sidebar settings ----------
st.sidebar.header("Settings & OpenAI")
st.sidebar.markdown("If you want the app to fetch missing DIN dims automatically, paste your OpenAI API key here.")
api_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")
use_gpt_fallback = st.sidebar.checkbox("Enable GPT fallback for missing DINs", value=True if api_key else False)
if api_key and openai is None:
    st.sidebar.warning("openai package not installed; GPT fallback will not work until you install openai.")

# Machine rates and defaults (editable)
st.sidebar.subheader("Machine Rates (₹/hr)")
if "rates" not in st.session_state:
    st.session_state.rates = {
        "Traub":700.0,"CNC Turning":650.0,"VMC Milling":600.0,"Drilling":500.0,"Threading":450.0,"Punching":400.0
    }
for rname in list(st.session_state.rates.keys()):
    st.session_state.rates[rname] = st.sidebar.number_input(rname+" (₹/hr)", value=float(st.session_state.rates[rname]), format="%.2f")
st.sidebar.subheader("Exchange Rates (INR per 1 unit)")
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0)
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0)
st.sidebar.subheader("Other defaults")
default_scrap = st.sidebar.number_input("Default scrap (%)", value=2.0)
default_overhead = st.sidebar.number_input("Default overhead (%)", value=10.0)
default_profit = st.sidebar.number_input("Default profit (%)", value=8.0)

# ---------- App tabs ----------
tabs = st.tabs(["Bulk Upload / Search DIN", "Calculator", "DIN DB", "Materials", "Vendor DB", "History", "Export"])

# ---------- DIN DB tab: view / edit local DB ----------
with tabs[2]:
    st.header("Local DIN / ISO Dimensions DB")
    st.markdown("This is the local DB used for fast lookup. You can edit entries and Save to persist.")
    din_df = DIN_DB.copy()
    edited = st.experimental_data_editor(din_df, num_rows="dynamic")
    st.session_state.din_db = edited
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Save DIN DB"):
            st.session_state.din_db.to_csv(DIN_DB_CSV, index=False)
            st.success(f"Saved DIN DB to {DIN_DB_CSV}")
    with c2:
        if st.button("Reload DIN DB"):
            st.session_state.din_db = load_or_init_csv(DIN_DB_CSV, DIN_DB.columns.tolist(), default_df=DIN_DEFAULTS)
            st.success("Reloaded DIN DB")

# ---------- Materials tab ----------
with tabs[3]:
    st.header("Materials (editable)")
    mat_df = MATERIALS.copy()
    edited_mat = st.experimental_data_editor(mat_df, num_rows="dynamic")
    st.session_state.materials_df = edited_mat
    m1, m2 = st.columns(2)
    with m1:
        if st.button("Save Materials"):
            st.session_state.materials_df.to_csv(MATERIALS_CSV, index=False)
            st.success("Materials saved")
    with m2:
        if st.button("Reload Materials"):
            st.session_state.materials_df = load_or_init_csv(MATERIALS_CSV, MATERIALS.columns.tolist(), default_df=DEFAULT_MATERIALS)
            st.success("Materials reloaded")

# ---------- Vendor tab ----------
with tabs[4]:
    st.header("Vendor DB (editable)")
    vendor_df = VENDOR_DB.copy()
    edited_v = st.experimental_data_editor(vendor_df, num_rows="dynamic")
    st.session_state.vendor_db = edited_v
    v1,v2 = st.columns(2)
    with v1:
        if st.button("Save Vendor DB"):
            st.session_state.vendor_db.to_csv(VENDOR_CSV, index=False)
            st.success("Vendor saved")
    with v2:
        if st.button("Reload Vendor DB"):
            st.session_state.vendor_db = load_or_init_vendor()
            st.success("Vendor DB reloaded")

# ---------- History tab ----------
with tabs[5]:
    st.header("Saved Costing History")
    hist_df = HISTORY.copy()
    edited_h = st.experimental_data_editor(hist_df, num_rows="dynamic")
    st.session_state.cost_history = edited_h
    h1,h2 = st.columns(2)
    with h1:
        if st.button("Save History CSV"):
            st.session_state.cost_history.to_csv(HISTORY_CSV, index=False)
            st.success("History saved")
    with h2:
        if st.button("Reload History CSV"):
            st.session_state.cost_history = load_or_init_history()
            st.success("History reloaded")

# ---------- Bulk Upload / Search DIN tab ----------
with tabs[0]:
    st.header("Bulk Upload / DIN Search")
    st.markdown("Upload a CSV with an item column (e.g. 'Item') containing texts like 'DIN 933 M8x20' or 'DIN 933 M8 x20'. The app will parse and fill dimensions where possible.")

    uploaded = st.file_uploader("Upload CSV (columns allowed: Item, Qty, Material, OverrideDiameter, OverrideLength)", type=["csv","xlsx"])
    sample_btn = st.button("Download sample CSV")
    if sample_btn:
        sample = pd.DataFrame({
            "Item":["DIN 933 M8x20","DIN 934 M10","DIN 935 M30"],
            "Qty":[100,200,50],
            "Material":["A2 Stainless (304)","Steel (C45 / EN8)","A2 Stainless (304)"]
        })
        st.download_button("Download sample CSV", bytes_to_excel_bytes({"sample":sample}), file_name="sample_items.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_items = pd.read_csv(uploaded)
            else:
                df_items = pd.read_excel(uploaded)
        except Exception as e:
            st.error("Failed to read uploaded file: "+str(e))
            df_items = None

        if df_items is not None:
            st.write("Preview uploaded items (you can edit the table):")
            # ensure 'Item' column exists
            if "Item" not in df_items.columns:
                st.error("Uploaded file must contain an 'Item' column with item text like 'DIN 933 M8x20'.")
            else:
                edited_items = st.experimental_data_editor(df_items, num_rows="dynamic")
                # parse each item
                results = []
                missing = []
                for idx,row in edited_items.iterrows():
                    item_text = str(row.get("Item","")).strip()
                    qty = int(row.get("Qty",1)) if not pd.isna(row.get("Qty",1)) else 1
                    mat = row.get("Material", None)
                    override_d = row.get("OverrideDiameter", None) if "OverrideDiameter" in edited_items.columns else None
                    override_l = row.get("OverrideLength", None) if "OverrideLength" in edited_items.columns else None

                    standard, size, length = parse_item_text(item_text)
                    if override_d and not pd.isna(override_d): d_val = float(override_d)
                    else: d_val = None
                    if override_l and not pd.isna(override_l): length = float(override_l)

                    # lookup in DIN DB
                    found = None
                    if standard and size:
                        # search DIN DB
                        din_db = st.session_state.get("din_db", DIN_DB)
                        match = din_db[(din_db["Standard"].str.upper()==standard.upper()) & (din_db["Size"].str.upper()==size.upper())]
                        if not match.empty:
                            found = match.iloc[0].to_dict()

                    if found:
                        # use found dims but allow override by user columns
                        d_final = d_val if d_val else float(found.get("d") or 0)
                        dk = float(found.get("dk") or 0)
                        k = float(found.get("k") or 0)
                        s = float(found.get("s") or 0)
                        pitch = float(found.get("pitch") or 0)
                        head_type = found.get("HeadType")
                        notes = found.get("Notes","")
                    else:
                        # missing from local DB
                        if use_gpt_fallback and api_key:
                            st.write(f"Fetching dims for {standard} {size} via GPT...")
                            data = fetch_dimensions_via_gpt(api_key, standard, size)
                            if data:
                                st.write("GPT returned:", data)
                                # show for confirm and allow edit before adding to DB
                                st.write("Please review and edit/confirm before adding to local DB (you can cancel).")
                                # present editable fields
                                with st.form(key=f"confirm_form_{idx}"):
                                    std_in = st.text_input("Standard", value=data.get("standard",standard or ""))
                                    size_in = st.text_input("Size", value=data.get("size",size or ""))
                                    head_in = st.text_input("Head Type", value=data.get("head_type",""))
                                    d_in = st.number_input("d (mm)", value=float(data.get("d") or 0.0))
                                    dk_in = st.number_input("dk (head dia mm)", value=float(data.get("dk") or 0.0))
                                    k_in = st.number_input("k (head height mm)", value=float(data.get("k") or 0.0))
                                    s_in = st.number_input("s (af mm)", value=float(data.get("s") or 0.0))
                                    pitch_in = st.number_input("pitch (mm)", value=float(data.get("pitch") or 0.0))
                                    notes_in = st.text_area("notes", value=data.get("notes",""))
                                    col1,col2 = st.columns(2)
                                    with col1:
                                        add_db = st.form_submit_button("Add to local DB")
                                    with col2:
                                        skip_db = st.form_submit_button("Skip (do not add)")
                                if add_db:
                                    # append to local DB
                                    new_row = {"Standard":std_in,"Size":size_in,"HeadType":head_in,"d":d_in,"dk":dk_in,"k":k_in,"s":s_in,"pitch":pitch_in,"Notes":notes_in}
                                    st.session_state.din_db = st.session_state.get("din_db", DIN_DB).append(new_row, ignore_index=True)
                                    st.session_state.din_db.to_csv(DIN_DB_CSV, index=False)
                                    st.success(f"Added {std_in} {size_in} to local DIN DB.")
                                    d_final = d_in; dk = dk_in; k=k_in; s=s_in; pitch=pitch_in; head_type=head_in; notes=notes_in
                                else:
                                    st.info("Skipped adding to DB. You can add manually in DIN DB tab.")
                                    missing.append({"Item":item_text,"Reason":"Missing in local DB; GPT fetch skipped or failed"})
                                    continue
                            else:
                                st.error(f"No data from GPT for {standard} {size}")
                                missing.append({"Item":item_text,"Reason":"GPT fetch failed"})
                                continue
                        else:
                            missing.append({"Item":item_text,"Reason":"Missing in local DB"})
                            continue

                    # if we reach here we have dims and can compute costing via a brief pipeline
                    total_length = (length if length else 0) + (float(row.get("Parting", compute_auto_parting(length)) or compute_auto_parting(length)))
                    # minimal costing: compute mass and material cost (user may want to process later in Calculator)
                    vol = volume_by_stock("Round Bar", float(d_final), total_length) if d_final else None
                    mass = vol/1e9*float(row.get("Density", MATERIALS.loc[0,"Density (kg/m3)"])) if vol else None
                    results.append({
                        "Item":item_text,"Standard":standard,"Size":size,"Qty":qty,
                        "d":d_final,"dk":dk,"k":k,"s":s,"pitch":pitch,
                        "Length":length,"Parting":parting,"TotalLength":total_length,
                        "MassKgPerPc": mass
                    })

                # show results and allow export
                if results:
                    st.subheader("Matched / Processed Items")
                    st.dataframe(pd.DataFrame(results))
                    if st.button("Export matched items to Excel"):
                        st.download_button("Download Matched Items", bytes_to_excel_bytes({"Matched":pd.DataFrame(results)}), file_name="matched_items.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                if missing:
                    st.subheader("Missing Items (manual action required)")
                    st.dataframe(pd.DataFrame(missing))
                    st.warning("Missing items are those not found in the local DB. Use GPT fallback (sidebar) or add them manually in DIN DB tab.")

# ---------- Calculator Tab ----------
with tabs[1]:
    st.header("Single Item Calculator (full costing)")
    st.markdown("Enter or select a DIN from local DB, or freely type item and parse. All fields are editable.")

    # quick select from DIN DB
    din_options = st.session_state.get("din_db", DIN_DB)
    std_list = sorted(din_options["Standard"].unique().tolist())
    selected_std = st.selectbox("Select Standard (or choose 'Custom')", ["Custom"]+std_list)
    if selected_std != "Custom":
        sizes_for_std = sorted(din_options[din_options["Standard"]==selected_std]["Size"].tolist())
        selected_size = st.selectbox("Select Size", sizes_for_std)
        row = din_options[(din_options["Standard"]==selected_std)&(din_options["Size"]==selected_size)].iloc[0]
        prefill = {
            "stock_type":"Round Bar",
            "diameter": float(row["d"]) if pd.notna(row["d"]) else float(row["Size"].replace("M","")),
            "length": 50.0,
            "parting": compute_auto_parting if False else 3.0, # will override below
            "material": MATERIALS["Material"].iloc[0],
            "density": float(MATERIALS["Density (kg/m3)"].iloc[0]),
            "mat_price": float(MATERIALS["Default Price (₹/kg)"].iloc[0]),
            "head_type": row.get("HeadType",""),
            "dk": row.get("dk",None),
            "k": row.get("k",None),
            "s": row.get("s",None),
            "pitch": row.get("pitch",None)
        }
    else:
        prefill = {"stock_type":"Round Bar","diameter":30.0,"length":50.0,"parting":3.0,"material":MATERIALS["Material"].iloc[0],"density":MATERIALS["Density (kg/m3)"].iloc[0],"mat_price":MATERIALS["Default Price (₹/kg)"].iloc[0]}

    colA,colB = st.columns([2,1])
    with colA:
        stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"], index=0)
        diameter = st.number_input("Diameter / Across Flats / Side / OD (mm)", value=prefill["diameter"])
        length = st.number_input("Length (mm)", value=prefill["length"])
        # auto parting logic
        def compute_auto_parting(L):
            if L <= 25: return 3.0
            if L <= 35: return 4.0
            if L <= 50: return 5.0
            if L <= 65: return 6.0
            return 7.0
        auto_parting_flag = st.checkbox("Use automatic parting rule", value=True)
        if auto_parting_flag:
            parting = compute_auto_parting(length)
            st.write(f"Auto parting set to {parting} mm")
        else:
            parting = st.number_input("Parting (mm) override", value=prefill.get("parting",3.0))
        thread = st.text_input("Thread (e.g. M10)", value=selected_size if selected_std!="Custom" else "M30")
        qty = st.number_input("Quantity", value=100, min_value=1)
    with colB:
        material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
        mat_row = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
        density = st.number_input("Density (kg/m3)", value=float(mat_row["Density (kg/m3)"]))
        mat_price = st.number_input("Material Price (₹/kg)", value=float(mat_row["Default Price (₹/kg)"]))
        traub_cycle = st.number_input("Traub / Turning cycle (sec)", value=max(12, (length+parting)*0.4 + 12))
        milling_cycle = st.number_input("Milling cycle (sec)", value=max(8, (length+parting)*0.25 + 8))
        threading_cycle = st.number_input("Threading cycle (sec)", value=20.0)
        punching_cycle = st.number_input("Punching (sec)", value=max(6, 6 + (length+parting)*0.03))
        tooling_cost = st.number_input("Tooling cost (₹)", value=2000.0)
        run_qty = st.number_input("Tooling run qty", value=5000, min_value=1)
        heat_treat = st.number_input("Heat treat/passivation (₹/pc)", value=0.3)
        plating = st.number_input("Plating (₹/pc)", value=0.0)
        inspection = st.number_input("Inspection / Deburr (₹/pc)", value=0.1)
        packaging = st.number_input("Packaging (₹/pc)", value=0.05)
        labour_add = st.number_input("Labour add-on (₹/pc)", value=0.1)
        scrap_pct = st.number_input("Scrap (%)", value=default_scrap)

    total_length = length + parting
    vol_mm3 = volume_by_stock(stock_type, diameter, total_length)
    mass_kg = vol_mm3 / 1e9 * density
    material_cost = mass_kg * mat_price

    rates = st.session_state.rates
    traub_cost = (rates["Traub"] * traub_cycle)/3600.0
    milling_cost = (rates["VMC Milling"] * milling_cycle)/3600.0
    threading_cost = (rates["Threading (machine) (tapping/rolling)"] * threading_cycle)/3600.0
    punching_cost = (rates["Punching / Press"] * punching_cycle)/3600.0
    tooling_per_piece = tooling_cost/run_qty
    other_direct = heat_treat + plating + inspection + packaging + labour_add

    direct_subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_per_piece + other_direct
    direct_with_scrap = direct_subtotal / (1 - scrap_pct/100.0) if (1 - scrap_pct/100.0)!=0 else direct_subtotal
    overhead = direct_with_scrap * (default_overhead/100.0)
    profit_amount = (direct_with_scrap + overhead) * (default_profit/100.0)
    final_price_inr = direct_with_scrap + overhead + profit_amount

    st.subheader("Result")
    st.metric("Material kg/pc", f"{mass_kg:.6f}")
    st.metric("Final price / pc (INR)", f"₹ {final_price_inr:.4f}")
    if enable_multi_currency := True:
        st.metric("Final price / pc (USD)", f"$ {final_price_inr/usd_rate:.4f}")
        st.metric("Final price / pc (EUR)", f"€ {final_price_inr/eur_rate:.4f}")
    st.write(f"Total for {qty} pcs: ₹ {final_price_inr*qty:.2f}")
    st.write(f"Total for 100 pcs: ₹ {final_price_inr*100:.2f}")

    # Save to history
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

# ---------- Export tab ----------
with tabs[6]:
    st.header("Export / Bulk Results")
    if st.button("Export Materials, DIN DB, Vendor DB, History as Excel"):
        out = bytes_to_excel_bytes({
            "Materials": st.session_state.materials_df,
            "DIN_DB": st.session_state.din_db if "din_db" in st.session_state else DIN_DB,
            "VendorDB": st.session_state.vendor_db,
            "History": st.session_state.cost_history
        })
        st.download_button("Download All Data", out, file_name="fastener_data_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.write("App ready. GPT fallback option will fetch and propose dimensions (JSON) for missing DINs — you must confirm before adding to local DB.")
