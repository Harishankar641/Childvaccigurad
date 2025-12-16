# utils.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_dataset(path):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    required_cols = [
        "child_id","gender","age","state","district","village",
        "phc_code","dose_number","vaccine","status",
        "last_due_date","provider_type"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["last_due_date"] = pd.to_datetime(df["last_due_date"], errors="coerce")
    df["status"] = df["status"].str.lower().str.strip()
    df["state"] = df["state"].str.strip()
    df["district"] = df["district"].str.strip()
    df["vaccine"] = df["vaccine"].str.strip()

    return df
