import pandas as pd
import io

# Load CSV or initialize if missing
def load_or_init_csv(path, columns, default_df):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = default_df.copy()
        df.to_csv(path, index=False)
    return df

# Volume calculation for different stock types
def volume_by_stock(stock_type, diameter, length):
    if stock_type.lower() == "round bar":
        r = diameter/2
        return 3.14159 * r**2 * length
    elif stock_type.lower() == "hex bar":
        side = diameter/2
        return 2.598 * side**2 * length
    elif stock_type.lower() == "square bar":
        return diameter**2 * length
    elif stock_type.lower() == "tube":
        wall = diameter*0.1
        return 3.14159*((diameter/2)**2 - ((diameter/2-wall)**2))*length
    elif stock_type.lower() == "sheet/cold formed":
        thickness = 5
        return diameter * length * thickness
    else:
        return diameter * diameter * length

# Auto parting
def compute_auto_parting(length):
    if length < 25: return 3.0
    elif length < 35: return 4.0
    elif length < 50: return 5.0
    elif length < 65: return 6.0
    else: return 8.0

# GPT DIN lookup placeholder
def query_gpt_for_din(din_number, size, stock_type, openai_key):
    return [10.0, 5.0]

# PDF placeholder
def pdf_quote_bytes(summary_dict):
    pdf_bytes = io.BytesIO()
    pdf_bytes.write(b"%PDF-1.4\n%Dummy PDF content for quotation")
    pdf_bytes.seek(0)
    return pdf_bytes
