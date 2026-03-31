import base64
import os
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.analyzer import analyze_dataset


load_dotenv()

st.set_page_config(page_title="Data Insight Studio", page_icon="📊", layout="wide")
st.title("Data Insight Studio")
st.write("Upload a CSV or Excel file to generate quick stats, charts, and an optional Groq-powered narrative.")


def _decode_plot(image_b64: Optional[str]):
    if not image_b64:
        return None
    try:
        return base64.b64decode(image_b64)
    except Exception:
        return None


uploaded = st.file_uploader("Upload dataset", type=["csv", "xls", "xlsx"])

if uploaded:
    with st.spinner("Analyzing dataset..."):
        content = uploaded.read()
        if not content:
            st.error("The uploaded file is empty.")
        else:
            api_key: Optional[str] = os.getenv("GROQ_API_KEY")
            try:
                result = analyze_dataset(content, uploaded.name, api_key)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    summary = result["summary"]
    plots = result["plots"]

    st.subheader("Dataset overview")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Rows", summary["meta"].get("rows", 0))
    meta_cols[1].metric("Columns", summary["meta"].get("columns", 0))
    meta_cols[2].write(
        f"Columns: {', '.join(summary['meta'].get('column_names', []))}" if summary["meta"].get("column_names") else ""
    )

    dtypes_df = pd.DataFrame(summary.get("dtypes", []))
    if not dtypes_df.empty:
        st.write("Detected types")
        st.dataframe(dtypes_df, use_container_width=True)

    if not summary.get("numeric_summary", pd.DataFrame()).empty:
        st.subheader("Numeric summary")
        st.dataframe(summary["numeric_summary"], use_container_width=True)

    missing_df = summary.get("missing", pd.DataFrame())
    if not missing_df.empty:
        st.subheader("Missingness")
        st.dataframe(missing_df, use_container_width=True)
        missing_plot = _decode_plot(plots.get("missing"))
        if missing_plot:
            st.image(missing_plot, caption="Missing data per column", use_column_width=True)

    corr_plot = _decode_plot(plots.get("correlation"))
    if corr_plot:
        st.subheader("Correlations")
        st.image(corr_plot, caption="Correlation heatmap", use_column_width=True)

    dist_plot = _decode_plot(plots.get("distribution"))
    if dist_plot:
        st.subheader("Distribution sample")
        st.image(dist_plot, caption="Distribution", use_column_width=True)

    if result.get("forecast"):
        forecast = result["forecast"]
        st.subheader(f"7-day forecast ({forecast.get('target')})")
        forecast_plot = _decode_plot(forecast.get("chart"))
        if forecast_plot:
            st.image(forecast_plot, caption="Forecast", use_column_width=True)
        forecast_df = pd.DataFrame(forecast.get("forecast_head", []))
        if not forecast_df.empty:
            st.dataframe(forecast_df, use_container_width=True)

    st.subheader("AI narrative")
    if result.get("insights"):
        st.success(result["insights"])
    else:
        st.info("Set GROQ_API_KEY in the environment to enable the LLM-generated summary.")
else:
    st.info("Waiting for a file upload. Supported: CSV, XLS, XLSX.")
