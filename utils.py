# utils.py
import numpy as np
import pandas as pd
import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

try:
    import openai
except:
    openai = None

# -------------------- Volume Calculations --------------------
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

# -------------------- Auto Parting --------------------
def compute_auto_parting(L):
    if L<=25: return 3.0
    if L<=35: return 4.0
    if L<=50: return 5.0
    if L<=65: return 6.0
    return 7.0

# -------------------- CSV Loader --------------------
def load_or_init_csv(path, cols, default_df=None):
    import os
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except: pass
    if default_df is not None:
        default_df.to_csv(path,index=False)
        return default_df.copy()
    return pd.DataFrame(columns=cols)

# -------------------- GPT DIN Lookup --------------------
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
        print("GPT lookup failed:", e)
    return None

# -------------------- PDF Generation --------------------
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
