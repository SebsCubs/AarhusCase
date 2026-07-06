"""Check (1) the shunt's max achievable supply vs the curve command, and
(2) fit the outdoor-reset curve to the MEASURED supply-vs-outdoor data, to see
how far the default s=1.5/b=35 curve is from the real building."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from aarhus_model.skoven_model import DATA_DIR, SHUNT_DEFAULTS


def load(name):
    s = pd.read_csv(os.path.join(DATA_DIR, name), index_col=0)["value"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


T_oa = load("outdoor_temperature.csv")
T_sup = load("ecl310_TSupHea_y_processed.csv")
T_ret = load("ecl310_TRetHea_y_processed.csv")

# align on common timestamps, winter heating season only (T_sup > 30 => heating on)
df = pd.concat({"T_oa": T_oa, "T_sup": T_sup, "T_ret": T_ret}, axis=1).dropna()
heat = df[df["T_sup"] > 30.0]
print(f"aligned samples: {len(df)}, heating-on (T_sup>30): {len(heat)}")
print(f"measured supply : mean {heat.T_sup.mean():.1f}  min {heat.T_sup.min():.1f}  "
      f"max {heat.T_sup.max():.1f}")
print(f"measured return : mean {heat.T_ret.mean():.1f}")
print(f"outdoor (heating): mean {heat.T_oa.mean():.1f}  min {heat.T_oa.min():.1f}  "
      f"max {heat.T_oa.max():.1f}")

# Fit T_sup = b + s*(21 - T_oa)  ->  linear in x=(21 - T_oa)
x = 21.0 - heat["T_oa"].values
y = heat["T_sup"].values
s, b = np.polyfit(x, y, 1)
print(f"\nFITTED curve to measured data:  s = {s:.3f} ,  b = {b:.2f}")
print(f"DEFAULT curve in skoven.yaml:   s = 1.500 ,  b = 35.00")
for toa in (-5, 0, 5, 10):
    d_def = 35 + 1.5 * (21 - toa)
    d_fit = b + s * (21 - toa)
    print(f"  T_oa={toa:+3d}C:  default curve -> {d_def:5.1f}C ,  fitted -> {d_fit:5.1f}C")

# Shunt max achievable supply = full-primary blend of T_primary with return
Tp = SHUNT_DEFAULTS["T_primary_C"]
pmax = SHUNT_DEFAULTS["primary_max_kgs"]
recirc = SHUNT_DEFAULTS["recirc_kgs"]
print(f"\nShunt: T_primary={Tp}C, primary_max={pmax} kg/s, recirc={recirc} kg/s")
for Tr in (22, 25, 37):
    max_sup = (pmax * Tp + recirc * Tr) / (pmax + recirc)
    print(f"  return={Tr}C -> MAX achievable supply (valve wide open) = {max_sup:.1f}C")
print("=> if the curve commands above this, the ECL310 valve saturates and the "
      "supply (hence radiator power) is stuck ~constant, defeating the reset.")
