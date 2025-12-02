# data.py
import pandas as pd

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
