import base64
import os
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.analyzer import analyze_dataset

load_dotenv()

st.set_page_config(
    page_title="DataBrief AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"], .stApp {
    background-color: #08080f !important;
    color: #e0ddf5;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1400px; }

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2.2rem 0;
    margin-bottom: 2.2rem;
    border-bottom: 1px solid #1c1c2e;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(120deg, #ffffff 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #52506a;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Upload zone — force dark ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: #0f0f1a !important;
    border: 1.5px dashed #2a2840 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #7c6fff !important;
}
[data-testid="stFileUploader"] label { color: #52506a !important; }
[data-testid="stFileUploader"] section {
    background: #0f0f1a !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] button {
    background: #1c1c2e !important;
    color: #a78bfa !important;
    border: 1px solid #2a2840 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] svg { color: #52506a !important; }
[data-testid="stFileUploader"] p { color: #52506a !important; }

/* ── Metric cards ── */
.cards-row {
    display: flex;
    gap: 1rem;
    margin: 1.8rem 0;
    flex-wrap: wrap;
}
.card {
    flex: 1;
    min-width: 150px;
    background: #0f0f1a;
    border: 1px solid #1c1c2e;
    border-radius: 14px;
    padding: 1.2rem 1.6rem 1rem;
    transition: border-color 0.25s, transform 0.15s;
}
.card:hover {
    border-color: #7c6fff44;
    transform: translateY(-1px);
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #52506a;
    margin-bottom: 0.4rem;
}
.card-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e0ddf5;
    line-height: 1.1;
}
.card-value.small { font-size: 1.05rem; padding-top: 0.45rem; }
.card-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #7c6fff;
    margin-top: 0.25rem;
}

/* ── Section title ── */
.sec {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7c6fff;
    margin: 2rem 0 0.9rem;
}

/* ── Column tags ── */
.tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 1.8rem; }
.tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    padding: 0.2rem 0.65rem;
    border-radius: 6px;
    border: 1px solid #2a2840;
    background: #0f0f1a;
    color: #b0adc8;
}
.tag.num  { color: #a78bfa; border-color: #7c6fff44; }
.tag.cat  { color: #fbbf24; border-color: #f59e0b44; }
.tag.disc { color: #34d399; border-color: #34d39944; }
.tag.id   { color: #52506a; border-color: #ffffff10; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1c1c2e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #52506a !important;
    padding: 0.8rem 1.4rem;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c6fff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── AI Insight box ── */
.insight {
    background: #0f0f1a;
    border: 1px solid #2a1f4a;
    border-left: 3px solid #7c6fff;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    font-size: 0.88rem;
    line-height: 1.9;
    color: #ccc9e8;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ── Dataframe ── */
.stDataFrame iframe { border-radius: 10px !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c6fff !important; }

/* ── Info / success ── */
.stAlert { border-radius: 10px !important; background: #0f0f1a !important; }
</style>
""", unsafe_allow_html=True)


def _decode(b64: Optional[str]):
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">DataBrief AI</div>
    <div class="hero-sub">Drop a dataset — get instant intelligence.</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "upload", type=["csv", "xls", "xlsx"],
    label_visibility="collapsed",
)
if not uploaded:
    st.markdown('<p style="color:#52506a;font-size:0.8rem;margin-top:0.5rem;">Supported: CSV · XLS · XLSX</p>',
                unsafe_allow_html=True)
    st.stop()

# ── Analyze ────────────────────────────────────────────────────────────────────
with st.spinner("Analysing…"):
    content = uploaded.read()
    if not content:
        st.error("The uploaded file is empty.")
        st.stop()
    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    try:
        result = analyze_dataset(content, uploaded.name, api_key)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

summary    = result["summary"]
meta       = summary["meta"]
continuous = summary["continuous_columns"]
discrete   = summary["discrete_columns"]
cat_cols   = summary["categorical_columns"]
id_cols    = summary.get("id_columns", [])
missing_df = summary.get("missing", pd.DataFrame())
total_miss = int(missing_df["missing"].sum()) if not missing_df.empty else 0

# ── Metric cards ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cards-row">
  <div class="card">
    <div class="card-label">Rows</div>
    <div class="card-value">{meta['rows']:,}</div>
    <div class="card-sub">records</div>
  </div>
  <div class="card">
    <div class="card-label">Columns</div>
    <div class="card-value">{meta['columns']}</div>
    <div class="card-sub">{len(continuous)} continuous · {len(discrete)} discrete · {len(cat_cols)} categorical</div>
  </div>
  <div class="card">
    <div class="card-label">Completeness</div>
    <div class="card-value">{meta['completeness']}%</div>
    <div class="card-sub">{total_miss:,} missing values</div>
  </div>
  <div class="card">
    <div class="card-label">File</div>
    <div class="card-value small">{uploaded.name}</div>
    <div class="card-sub">{round(len(content)/1024, 1)} KB</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Column tags ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Columns</div>', unsafe_allow_html=True)
html = '<div class="tags">'
for col in meta["column_names"]:
    if col in id_cols:
        cls, tip = "id", "ID column — excluded"
    elif col in continuous:
        cls, tip = "num", "continuous numeric"
    elif col in discrete:
        cls, tip = "disc", "discrete numeric"
    elif col in cat_cols:
        cls, tip = "cat", "categorical"
    else:
        cls, tip = "", ""
    html += f'<span class="tag {cls}" title="{tip}">{col}</span>'
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_dash, tab_stats, tab_forecast, tab_ai = st.tabs(
    ["  📊  Dashboard  ", "  📋  Statistics  ", "  📡  Forecast  ", "  🤖  AI Narrative  "]
)

# ── Dashboard ──────────────────────────────────────────────────────────────────
with tab_dash:
    img = _decode(result.get("dashboard"))
    if img:
        st.image(img, use_column_width=True)
    else:
        st.info("Dashboard could not be generated.")

# ── Statistics ─────────────────────────────────────────────────────────────────
with tab_stats:
    num_summary = summary.get("numeric_summary", pd.DataFrame())
    if not num_summary.empty:
        st.markdown('<div class="sec">Numeric summary</div>', unsafe_allow_html=True)
        st.dataframe(num_summary.set_index("column"), use_container_width=True)

    if not missing_df.empty:
        st.markdown('<div class="sec">Missing values</div>', unsafe_allow_html=True)
        st.dataframe(
            missing_df.set_index("column").sort_values("percent", ascending=False),
            use_container_width=True,
        )

    top_cats = summary.get("top_categories", [])
    if top_cats:
        st.markdown('<div class="sec">Top category values</div>', unsafe_allow_html=True)
        for i in range(0, len(top_cats), 3):
            chunk = top_cats[i:i+3]
            cols = st.columns(len(chunk))
            for j, cat in enumerate(chunk):
                with cols[j]:
                    st.markdown(
                        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                        f'color:#a78bfa;margin-bottom:0.3rem;">{cat["column"]}</p>',
                        unsafe_allow_html=True,
                    )
                    df_c = pd.DataFrame({
                        "Value": list(cat["values"].keys()),
                        "Count": list(cat["values"].values()),
                    })
                    st.dataframe(df_c, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Data types</div>', unsafe_allow_html=True)
    dtypes_df = pd.DataFrame(summary.get("dtypes", []))
    if not dtypes_df.empty:
        st.dataframe(dtypes_df.set_index("name"), use_container_width=True)

# ── Forecast ───────────────────────────────────────────────────────────────────
with tab_forecast:
    if result.get("forecast"):
        fc = result["forecast"]
        target = fc.get("target", "")
        st.markdown(f'<div class="sec">7-day forecast · {target}</div>', unsafe_allow_html=True)
        fc_img = _decode(fc.get("chart"))
        if fc_img:
            st.image(fc_img, use_column_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#52506a;margin-bottom:0.4rem;">RECENT HISTORY</p>', unsafe_allow_html=True)
            hist = pd.DataFrame(fc.get("history_tail", []))
            if not hist.empty:
                st.dataframe(hist, use_container_width=True, hide_index=True)
        with c2:
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#52506a;margin-bottom:0.4rem;">FORECAST</p>', unsafe_allow_html=True)
            fc_df = pd.DataFrame(fc.get("forecast_head", []))
            if not fc_df.empty:
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Forecast unavailable — needs a datetime column with ≥ 15 daily data points.")

# ── AI Narrative ───────────────────────────────────────────────────────────────
with tab_ai:
    if result.get("insights"):
        st.markdown(f'<div class="insight">{result["insights"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Set `GROQ_API_KEY` in your `.env` to enable AI-generated insights.")