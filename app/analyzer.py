import base64
import io
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from groq import Groq
from statsmodels.tsa.holtwinters import ExponentialSmoothing

matplotlib.use("agg")


def _fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()
    stream = io.BytesIO(file_bytes)
    if extension in {".csv", ".txt"}:
        return pd.read_csv(stream)
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(stream)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def summarize_dataframe(df: pd.DataFrame) -> Dict:
    meta = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
    }

    dtypes = []
    for name, dtype in df.dtypes.items():
        dtypes.append({"name": name, "dtype": str(dtype)})
    meta["dtypes"] = dtypes

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_table = (
        df[numeric_cols].describe(include="all").T.reset_index().rename(columns={"index": "column"})
        if numeric_cols
        else pd.DataFrame()
    )

    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
        .assign(percent=lambda x: (x["missing"] / len(df) * 100).round(2))
    )

    top_categories: List[Dict] = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        value_counts = df[col].value_counts(dropna=True).head(5)
        top_categories.append({
            "column": col,
            "values": value_counts.to_dict(),
        })

    return {
        "meta": meta,
        "numeric_summary": summary_table,
        "missing": missing,
        "top_categories": top_categories,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
    }


def correlation_plot(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
    ax.set_title("Correlation heatmap")
    return _fig_to_base64(fig)


def distribution_plot(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax, color="#2563eb")
    ax.set_title(f"Distribution of {col}")
    return _fig_to_base64(fig)


def missingness_plot(missing: pd.DataFrame) -> Optional[str]:
    if missing.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=missing, x="column", y="percent", color="#10b981", ax=ax)
    ax.set_ylabel("Missing %")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Missing data per column")
    return _fig_to_base64(fig)


def detect_time_series(df: pd.DataFrame) -> Optional[Tuple[str, pd.DataFrame]]:
    if df.empty:
        return None

    datetime_col = None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            datetime_col = col
            break
        try:
            parsed = pd.to_datetime(df[col])
            # keep parseable columns with enough unique timestamps
            if parsed.notna().sum() >= 10:
                df[col] = parsed
                datetime_col = col
                break
        except Exception:
            continue

    if not datetime_col:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in numeric_cols if c != datetime_col]
    if not target_cols:
        return None

    target = target_cols[0]
    ts = df[[datetime_col, target]].dropna()
    if ts.empty:
        return None

    ts = ts.sort_values(datetime_col)
    ts = ts.set_index(datetime_col)
    ts = ts[target].resample("D").mean().ffill()
    if len(ts) < 15:
        return None

    train = ts.iloc[:-7] if len(ts) > 21 else ts
    try:
        model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
        forecast = model.forecast(7)
    except Exception:
        return None

    forecast_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
    forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.date

    fig, ax = plt.subplots(figsize=(7, 4))
    ts.tail(90).plot(ax=ax, label="History")
    forecast.plot(ax=ax, label="Forecast", color="#ef4444")
    ax.legend()
    ax.set_title(f"7-day forecast for {target}")
    chart = _fig_to_base64(fig)

    return target, pd.DataFrame({"history": ts.tail(10)}), chart, forecast_df


def generate_llm_insights(
    summary: Dict,
    sample_rows: List[Dict],
    forecast: Optional[Dict],
    api_key: Optional[str],
) -> Optional[str]:
    if not api_key:
        return None

    client = Groq(api_key=api_key)

    prompt = {
        "role": "user",
        "content": (
            "You are a senior enterprise data analyst.\n"
            "Analyze the dataset and generate structured insights.\n\n"
            "Give output in this format:\n"
            "1. Key Insights (with explanation)\n"
            "2. Business Implications\n"
            "3. Risks / Issues\n"
            "4. Recommendations\n\n"
            "Make insights specific, non-obvious, and useful for decision-making.\n"
            "Avoid generic statements."
        ),
    }

    payload = {
        "meta": summary.get("meta"),
        "numeric_columns": summary.get("numeric_columns"),
        "categorical_columns": summary.get("categorical_columns"),
        "missing": summary.get("missing").to_dict(orient="records") if not summary.get("missing").empty else [],
        "sample_rows": sample_rows,
        "forecast_head": forecast.get("forecast_head") if forecast else None,
    }

    message = json.dumps(payload, default=str)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[prompt, {"role": "user", "content": message}],
        temperature=0.4,
        max_tokens=256,
    )

    return completion.choices[0].message.content.strip()


def analyze_dataset(file_bytes: bytes, filename: str, api_key: Optional[str]) -> Dict:
    df = load_dataframe(file_bytes, filename)
    summary = summarize_dataframe(df)

    plots: Dict[str, Optional[str]] = {
        "correlation": correlation_plot(df, summary["numeric_columns"]),
        "distribution": distribution_plot(df, summary["numeric_columns"]),
        "missing": missingness_plot(summary["missing"]),
    }

    forecast_payload = None
    ts_result = detect_time_series(df.copy())
    if ts_result:
        target, history_tail, chart, forecast_df = ts_result
        forecast_payload = {
            "target": target,
            "history_tail": history_tail.reset_index().to_dict(orient="records"),
            "forecast_head": forecast_df.to_dict(orient="records"),
            "chart": chart,
        }

    sample_rows = df.head(5).to_dict(orient="records")
    insights = generate_llm_insights(summary, sample_rows, forecast_payload, api_key)

    return {
        "summary": summary,
        "plots": plots,
        "forecast": forecast_payload,
        "insights": insights,
    }
