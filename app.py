# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import haversine_distances

st.set_page_config(page_title="Store Distance Matrix", layout="wide")
st.title("📍 Store Distance Matrix (Haversine, km)")

st.markdown(
    """
อัปโหลดไฟล์ **CSV** ที่มีคอลัมน์:
- **STORE CODE**
- **STORE NAME_ENGLISH**
- **LATITUDE**
- **LONGITUDE**

ระบบจะคำนวณ **Distance Matrix (km)** และให้ดาวน์โหลดเป็น CSV/Excel ได้
"""
)

# -----------------------------
# Helpers
# -----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names (remove \n, extra spaces, make upper, replace spaces with underscore)."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)  # removes \n, \t, multiple spaces
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    return out


def coerce_latlon(df: pd.DataFrame, lat="LATITUDE", lon="LONGITUDE") -> pd.DataFrame:
    """Clean and coerce LAT/LON to numeric (robust against commas, whitespace, stray chars)."""
    out = df.copy()
    for c in [lat, lon]:
        out[c] = (
            out[c]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)          # remove commas
            .str.replace(r"[^\d\.\-]+", "", regex=True) # keep digits . -
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compute_matrix_km(df: pd.DataFrame) -> pd.DataFrame:
    coords = np.radians(df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
    dist_km = haversine_distances(coords, coords) * 6371.0
    return pd.DataFrame(
        dist_km,
        index=df["STORE_CODE"].astype(str),
        columns=df["STORE_CODE"].astype(str),
    )


def to_long(distance_df: pd.DataFrame) -> pd.DataFrame:
    # ทำให้ชื่อ index/columns ไม่ชนกับคอลัมน์ที่จะสร้าง
    s = distance_df.copy()
    s.index.name = "FROM_STORE_CODE"
    s.columns.name = "TO_STORE_CODE"

    long_df = (
        s.stack(dropna=False)
         .rename("DIST_KM")
         .reset_index()
    )
    return long_df


# -----------------------------
# UI Controls
# -----------------------------
uploaded = st.file_uploader("📤 Upload stores CSV", type=["csv"])

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
with c1:
    round_decimals = st.number_input("Round decimals", min_value=0, max_value=6, value=2, step=1)
with c2:
    remove_diagonal = st.checkbox("Set diagonal to 0", value=True)
with c3:
    show_long = st.checkbox("Show long-format table", value=True)
with c4:
    show_raw = st.checkbox("Show raw preview", value=True)

if not uploaded:
    st.info("อัปโหลดไฟล์ CSV เพื่อเริ่มคำนวณ ✨")
    st.stop()

# -----------------------------
# Load CSV (with utf-8-sig fallback)
# -----------------------------
try:
    df_raw = pd.read_csv(uploaded)
except UnicodeDecodeError:
    uploaded.seek(0)
    df_raw = pd.read_csv(uploaded, encoding="utf-8-sig")

if show_raw:
    st.subheader("1) Raw data preview")
    st.dataframe(df_raw.head(100), use_container_width=True)

# -----------------------------
# Clean + Validate
# -----------------------------
df = clean_columns(df_raw)

required = ["STORE_CODE", "STORE_NAME_ENGLISH", "LATITUDE", "LONGITUDE"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns after cleaning: {missing}")
    st.info(f"Detected columns: {df.columns.tolist()}")
    st.stop()

df = coerce_latlon(df)

# show invalid rows
bad = df[df[["LATITUDE", "LONGITUDE"]].isna().any(axis=1)].copy()
if len(bad) > 0:
    st.warning(f"Found {len(bad)} rows with invalid LAT/LON (will be dropped). Showing first 20:")
    st.dataframe(bad.head(20), use_container_width=True)

# drop invalid + range sanity
df = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
df = df[df["LATITUDE"].between(-90, 90) & df["LONGITUDE"].between(-180, 180)].copy()

if len(df) < 2:
    st.error("Need at least 2 valid stores to compute distance matrix.")
    st.stop()

st.subheader("2) Cleaned data (used for calculation)")
st.dataframe(df[required].head(200), use_container_width=True)

# -----------------------------
# Compute
# -----------------------------
distance_df = compute_matrix_km(df)

if remove_diagonal:
    np.fill_diagonal(distance_df.values, 0.0)

distance_df = distance_df.round(int(round_decimals))

st.subheader("3) Distance Matrix (km)")
st.dataframe(distance_df, use_container_width=True, height=520)

long_df = None
if show_long:
    st.subheader("4) Long format (FROM → TO)")
    long_df = to_long(distance_df)
    st.dataframe(long_df.head(500), use_container_width=True, height=420)

# -----------------------------
# Downloads
# -----------------------------
st.subheader("⬇️ Download")

st.download_button(
    "Download Matrix CSV",
    data=distance_df.to_csv().encode("utf-8"),
    file_name="distance_matrix_km.csv",
    mime="text/csv",
)

output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    distance_df.to_excel(writer, sheet_name="matrix_km")
    if long_df is not None:
        long_df.to_excel(writer, sheet_name="long_format", index=False)

st.download_button(
    "Download Excel (matrix + long)",
    data=output.getvalue(),
    file_name="distance_matrix_km.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
