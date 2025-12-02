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
usd_rate = st.sidebar.number_input("1 USD = ? INR", value=83.0, min_value=0.01, step=0.01, format="%.2f")
eur_rate = st.sidebar.number_input("1 EUR = ? INR", value=90.0, min_value=0.01, step=0.01, format="%.2f")
default_scrap = st.sidebar.number_input("Scrap (%)", value=2.0, min_value=0.0, step=0.1, format="%.2f")
default_overhead = st.sidebar.number_input("Overhead (%)", value=10.0, min_value=0.0, step=0.1, format="%.2f")
default_profit = st.sidebar.number_input("Profit (%)", value=8.0, min_value=0.0, step=0.1, format="%.2f")

# -------------------- Tabs --------------------
tabs = st.tabs(["Bulk / DIN","Single Calculator","DIN DB","Materials","History"])

# -------------------- Single Calculator --------------------
with tabs[1]:
    st.header("Single Item Calculator")
    stock_type = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    diameter = st.number_input("Diameter / AF / Side / OD (mm)", value=30.0, min_value=0.1, step=0.1, format="%.2f")
    length = st.number_input("Length (mm)", value=50.0, min_value=0.1, step=0.1, format="%.2f")
    auto_parting = st.checkbox("Auto parting", True)
    parting = compute_auto_parting(length) if auto_parting else st.number_input("Parting (mm)", value=5.0, min_value=0.1, step=0.1, format="%.2f")
    qty = st.number_input("Quantity", value=100, min_value=1, step=1, format="%d")
    material = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow = st.session_state.materials_df[st.session_state.materials_df["Material"]==material].iloc[0]
    density = st.number_input("Density (kg/m3)", value=float(mrow["Density (kg/m3)"]), min_value=0.1, step=0.1, format="%.2f")
    mat_price = st.number_input("Material price (₹/kg)", value=float(mrow["Default Price (₹/kg)"]), min_value=0.01, step=0.01, format="%.2f")

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
                bulk_df['Qty'] = 100
            st.data_editor(bulk_df, num_rows="dynamic")

    openai_key = st.text_input("OpenAI API Key (for GPT lookup)", type="password")
    stock_type_bulk = st.selectbox("Stock Type", ["Round Bar","Hex Bar","Square Bar","Tube","Sheet/Cold Formed"])
    material_bulk = st.selectbox("Material", st.session_state.materials_df["Material"].tolist())
    mrow_bulk = st.session_state.materials_df[st.session_state.materials_df["Material"]==material_bulk].iloc[0]
    density_bulk = st.number_input("Density (kg/m3)", value=float(mrow_bulk["Density (kg/m3)"]), min_value=0.1, step=0.1, format="%.2f")
    mat_price_bulk = st.number_input("Material price (₹/kg)", value=float(mrow_bulk["Default Price (₹/kg)"]), min_value=0.01, step=0.01, format="%.2f")
    auto_parting_bulk = st.checkbox("Auto parting for bulk", True)

    if st.button("Process Bulk DIN List"):
        if 'bulk_df' not in locals():
            st.warning("Upload bulk file first!")
        else:
            processed_results = []
            updated_rows = 0
            for idx, r in bulk_df.iterrows():
                din_number = str(r['DIN']).strip()
                size = str(r.get('Size','')).strip()
                qty_item = int(r.get('Qty',100))

                matched = st.session_state.din_db[
                    (st.session_state.din_db['Standard'].str.upper() == din_number.upper()) &
                    ((st.session_state.din_db['Size'].str.upper() == size.upper()) if size else True)
                ]

                if not matched.empty:
                    row = matched.iloc[0]
                    diameter = float(row.get('d',10.0))
                    length = float(row.get('Length',50.0))
                    parting = compute_auto_parting(length) if auto_parting_bulk else float(row.get('Parting',5.0))
                else:
                    if openai_key.strip() != "":
                        dims = query_gpt_for_din(din_number, size, "unknown", openai_key)
                        if dims:
                            diameter = float(dims[0])
                            length = 50.0
                            parting = compute_auto_parting(length)
                            updated_rows +=1
                        else:
                            st.warning(f"Dimensions not found for {din_number} {size}")
                            continue
                    else:
                        st.warning(f"Dimensions not found for {din_number} {size}")
                        continue

                mass_kg = volume_by_stock(stock_type_bulk, diameter, length+parting)*density_bulk/1e9
                material_cost = mass_kg*mat_price_bulk
                traub_cost = material_cost*0.1
                milling_cost = material_cost*0.05
                threading_cost = material_cost*0.05
                punching_cost = material_cost*0.02
                tooling_cost = material_cost*0.01

                subtotal = material_cost + traub_cost + milling_cost + threading_cost + punching_cost + tooling_cost
                subtotal *= (1 + default_scrap/100)*(1 + default_overhead/100)*(1 + default_profit/100)

                processed_results.append({
                    "PartDesc": f"{din_number} {size}",
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
                    "UnitPrice_USD": round(subtotal/usd_rate,2),
                    "UnitPrice_EUR": round(subtotal/eur_rate,2),
                    "Qty": qty_item,
                    "Total_INR": round(subtotal*qty_item,2),
                    "Total_USD": round(subtotal*qty_item/usd_rate,2),
                    "Total_EUR": round(subtotal*qty_item/eur_rate,2),
                    "TargetPrice": "",
                    "Notes": ""
                })

            st.success(f"Processed {len(processed_results)} items. Updated {updated_rows} via GPT.")
            processed_df = pd.DataFrame(processed_results)
            st.data_editor(processed_df, num_rows="dynamic")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hist_df = st.session_state.cost_history.copy()
            for _, r in processed_df.iterrows():
                r_dict = r.to_dict()
                r_dict["Timestamp"] = timestamp
                hist_df = hist_df.append(r_dict, ignore_index=True)
            st.session_state.cost_history = hist_df
            st.session_state.cost_history.to_csv(HISTORY_CSV,index=False)
            st.success("Saved bulk DIN costing to History tab.")

            if st.button("Generate Quotation PDF for Bulk DIN"):
                total_batch = processed_df["Total_INR"].sum()
                summary = {"Total batch cost (₹)": total_batch, "USD Rate": usd_rate, "EUR Rate": eur_rate}
                pdf_bytes = pdf_quote_bytes(summary)
                st.download_button("Download Bulk DIN Quotation PDF", data=pdf_bytes, file_name="bulk_din_quotation.pdf", mime="application/pdf")

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
