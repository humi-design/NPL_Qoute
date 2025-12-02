import pandas as pd

# Default Materials
DEFAULT_MATERIALS = pd.DataFrame([
    {"Material":"A2 Stainless Steel", "Density (kg/m3)":8000, "Default Price (₹/kg)":200},
    {"Material":"Brass", "Density (kg/m3)":8500, "Default Price (₹/kg)":600},
    {"Material":"Carbon Steel", "Density (kg/m3)":7850, "Default Price (₹/kg)":150}
])

# Default DIN/ISO DB
DIN_DEFAULTS = pd.DataFrame([
    {"DIN":"DIN 933", "Size":"M8x20", "Diameter(mm)":8, "Length(mm)":20, "HeadDia(mm)":13},
    {"DIN":"DIN 934", "Size":"M8", "Diameter(mm)":8, "Length(mm)":None, "HeadDia(mm)":13},
    {"DIN":"DIN 912", "Size":"M10x30", "Diameter(mm)":10, "Length(mm)":30, "HeadDia(mm)":16},
    {"DIN":"DIN 6912", "Size":"M12x40", "Diameter(mm)":12, "Length(mm)":40, "HeadDia(mm)":18},
])
