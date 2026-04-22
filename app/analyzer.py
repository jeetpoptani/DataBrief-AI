import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from groq import Groq
from statsmodels.tsa.holtwinters import ExponentialSmoothing

matplotlib.use("agg")

# ── Palette ────────────────────────────────────────────────────────────────────
DARK_BG = "#08080f"
PANEL   = "#0f0f1a"
CARD    = "#13131f"
BORDER  = "#1c1c2e"
ACCENT  = "#7c6fff"
ACCENT2 = "#c084fc"
GREEN   = "#34d399"
ORANGE  = "#fb923c"
RED     = "#f87171"
YELLOW  = "#fbbf24"
BLUE    = "#60a5fa"
PINK    = "#f472b6"
TEXT    = "#e8e6f0"
MUTED   = "#52506a"
PALETTE = [ACCENT, GREEN, ORANGE, ACCENT2, RED, YELLOW, BLUE, PINK, "#a3e635", "#22d3ee"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _style(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)


def _fig_to_b64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Smart column classification ────────────────────────────────────────────────

def _classify_columns(df: pd.DataFrame):
    """
    Splits columns into:
    - continuous_numeric: float/int with >10 unique values (good for histograms/scatter)
    - discrete_numeric: int with <=10 unique values (treat like categories for bar charts)
    - categorical: object/category with low cardinality
    - id_cols: high-cardinality monotonic int (row index)
    - high_card_cols: high-cardinality text (names, tickets)
    """
    continuous, discrete, categorical, id_cols, high_card = [], [], [], [], []

    for col in df.columns:
        series = df[col].dropna()
        n_unique = series.nunique()
        n_total = max(len(df), 1)

        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = n_unique / n_total
            # ID detection: near-unique, monotonic, non-negative
            if unique_ratio > 0.95 and series.min() >= 0 and df[col].is_monotonic_increasing:
                id_cols.append(col)
            elif n_unique > 10:
                continuous.append(col)
            else:
                discrete.append(col)  # binary or small integer — bar chart
        else:
            unique_ratio = n_unique / n_total
            if unique_ratio > 0.5:
                high_card.append(col)
            else:
                categorical.append(col)

    return continuous, discrete, categorical, id_cols, high_card


# ── File loading ───────────────────────────────────────────────────────────────

def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    stream = io.BytesIO(file_bytes)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(stream)
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(stream)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


# ── Summary ────────────────────────────────────────────────────────────────────

def summarize_dataframe(df: pd.DataFrame) -> Dict:
    continuous, discrete, categorical, id_cols, high_card = _classify_columns(df)
    all_analysis_numeric = continuous + discrete

    missing = (
        df.isna().sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
        .assign(percent=lambda x: (x["missing"] / len(df) * 100).round(2))
    )
    total_missing = int(missing["missing"].sum())
    completeness = round((1 - total_missing / max(len(df) * len(df.columns), 1)) * 100, 1)

    numeric_summary = (
        df[all_analysis_numeric].describe().T.reset_index().rename(columns={"index": "column"})
        if all_analysis_numeric else pd.DataFrame()
    )

    top_categories = []
    for col in (categorical + discrete):
        vc = df[col].value_counts(dropna=True).head(8)
        top_categories.append({"column": col, "values": vc.to_dict()})

    dtypes = [{"name": n, "dtype": str(t)} for n, t in df.dtypes.items()]

    return {
        "meta": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "completeness": completeness,
        },
        "numeric_summary": numeric_summary,
        "missing": missing,
        "top_categories": top_categories,
        "continuous_columns": continuous,
        "discrete_columns": discrete,
        "numeric_columns": all_analysis_numeric,
        "categorical_columns": categorical,
        "id_columns": id_cols,
        "high_card_columns": high_card,
        "dtypes": dtypes,
    }


# ── Individual chart helpers ───────────────────────────────────────────────────

def _draw_histogram(ax, df, col, color):
    data = df[col].dropna()
    if data.empty:
        return
    ax.hist(data, bins=30, color=color, alpha=0.65, edgecolor=DARK_BG, linewidth=0.4)
    # KDE twin
    if len(data) > 5:
        try:
            kde = gaussian_kde(data, bw_method=0.3)
            xr = np.linspace(data.min(), data.max(), 300)
            ax2 = ax.twinx()
            ax2.plot(xr, kde(xr), color=TEXT, linewidth=1.8, alpha=0.6)
            ax2.set_yticks([])
            ax2.set_facecolor(PANEL)
            for sp in ax2.spines.values():
                sp.set_edgecolor(BORDER)
        except Exception:
            pass
    ax.axvline(data.mean(),   color=YELLOW, lw=1.2, ls="--", alpha=0.9)
    ax.axvline(data.median(), color=GREEN,  lw=1.2, ls="--", alpha=0.9)
    ax.text(0.97, 0.97,
            f"μ = {data.mean():.2f}\nmed = {data.median():.2f}\nσ = {data.std():.2f}\nskew = {data.skew():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=MUTED, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=DARK_BG, edgecolor=BORDER, alpha=0.9))
    ax.set_title(col, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel("")


def _draw_bar(ax, df, col, color, horizontal=True):
    vc = df[col].value_counts(dropna=True).head(10)
    if vc.empty:
        return
    vals = vc.values
    labels = [str(x) for x in vc.index]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(vc))]
    if horizontal:
        bars = ax.barh(labels[::-1], vals[::-1], color=bar_colors[::-1],
                       edgecolor=DARK_BG, linewidth=0.5, height=0.62)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + vc.max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=8, color=MUTED, fontfamily="monospace")
        ax.set_xlim(0, vc.max() * 1.28)
        ax.set_xlabel("count", fontsize=8)
        ax.grid(axis="x", color=BORDER, linewidth=0.4)
        ax.grid(axis="y", visible=False)
    else:
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor=DARK_BG, linewidth=0.5, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vc.max() * 0.02,
                    f"{val:,}", ha="center", fontsize=8, color=TEXT, fontfamily="monospace")
        ax.set_ylim(0, vc.max() * 1.2)
        ax.set_ylabel("count", fontsize=8)
        ax.grid(axis="y", color=BORDER, linewidth=0.4)
        ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_title(col, fontsize=10, fontweight="bold", pad=7)


# ── Dashboard builder ──────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, summary: Dict) -> str:
    continuous = summary["continuous_columns"]
    discrete   = summary["discrete_columns"]
    cat_cols   = summary["categorical_columns"]
    missing_df = summary["missing"]
    meta       = summary["meta"]

    bar_candidates = discrete + cat_cols
    has_continuous = len(continuous) >= 1
    has_corr       = len(continuous) >= 2
    has_bars       = len(bar_candidates) >= 1
    has_missing    = not missing_df[missing_df["missing"] > 0].empty
    has_scatter    = len(continuous) >= 2

    # ── Layout plan ────────────────────────────────────────────────────────────
    # Fixed 2-column grid (left | right) per row so panels are always equal size.
    # Row 0  : histograms — up to 2 shown side by side (col 0 | col 1)
    # Row 1  : corr heatmap (left) | first bar chart (right)
    # Row 2  : missing data (left) | scatter (right)
    # Every panel gets exactly one grid cell — no spanning.

    rows, heights = [], []

    if has_continuous:
        rows.append("dist");       heights.append(4.2)
    if has_corr or has_bars:
        rows.append("corr_bar");   heights.append(4.4)
    if has_missing or has_scatter:
        rows.append("miss_scatter"); heights.append(4.0)

    if not rows:
        rows.append("empty"); heights.append(2)

    n_rows = len(rows)
    fig_h  = sum(heights) + 2.4
    fig    = plt.figure(figsize=(18, fig_h), facecolor=DARK_BG)

    # 2-column grid — each row has exactly 2 equal cells
    gs = gridspec.GridSpec(
        n_rows, 2, figure=fig,
        height_ratios=heights,
        hspace=0.52, wspace=0.28,
        left=0.06, right=0.97,
        top=0.94, bottom=0.04,
    )

    # ── Header ─────────────────────────────────────────────────────────────────
    fig.text(0.06, 0.978, "DataBrief AI  ·  Dataset Analysis",
             color=TEXT, fontsize=17, fontweight="bold", fontfamily="monospace")
    fig.text(0.06, 0.966,
             f"{meta['rows']:,} rows  ·  {meta['columns']} cols  ·  "
             f"{meta['completeness']}% complete  ·  "
             f"{len(continuous)} continuous  ·  {len(discrete)} discrete  ·  {len(cat_cols)} categorical",
             color=MUTED, fontsize=8.5, fontfamily="monospace")
    fig.add_artist(plt.Line2D(
        [0.06, 0.97], [0.962, 0.962],
        transform=fig.transFigure, color=ACCENT, linewidth=0.8, alpha=0.4))

    # ── Row 0: Distributions — first two continuous columns ────────────────────
    if "dist" in rows:
        r = rows.index("dist")
        cols_to_show = continuous[:2]          # exactly 2 panels → left | right
        for i, col in enumerate(cols_to_show):
            ax = fig.add_subplot(gs[r, i])
            _style(ax)
            _draw_histogram(ax, df, col, PALETTE[i % len(PALETTE)])

        # If only 1 continuous col, fill right with a box-plot of the same column
        if len(cols_to_show) == 1:
            ax_box = fig.add_subplot(gs[r, 1])
            _style(ax_box)
            col = continuous[0]
            data = df[col].dropna()
            ax_box.boxplot(data, vert=True, patch_artist=True, widths=0.45,
                           medianprops=dict(color=ACCENT2, linewidth=2.5),
                           boxprops=dict(facecolor=ACCENT + "28", edgecolor=ACCENT),
                           whiskerprops=dict(color=MUTED, lw=1.2),
                           capprops=dict(color=MUTED),
                           flierprops=dict(marker="o", color=ACCENT, markersize=3, alpha=0.4))
            ax_box.set_title(f"{col}  — box plot", fontsize=10, fontweight="bold", pad=7)
            ax_box.set_xticks([])

    # ── Row 1: Correlation heatmap (left) | Bar chart (right) ──────────────────
    if "corr_bar" in rows:
        r = rows.index("corr_bar")

        # Left — correlation heatmap (always takes left cell)
        if has_corr:
            ax_corr = fig.add_subplot(gs[r, 0])
            _style(ax_corr)
            corr = df[continuous].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            cmap = sns.diverging_palette(260, 10, s=80, l=45, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                        linewidths=1.0, linecolor=DARK_BG, square=True, ax=ax_corr,
                        annot_kws={"size": 8.5, "color": TEXT},
                        cbar_kws={"shrink": 0.72}, vmin=-1, vmax=1)
            ax_corr.set_title("Correlation matrix  (continuous columns)", fontsize=10, fontweight="bold")
            ax_corr.tick_params(axis="x", rotation=35, labelsize=8.5)
            ax_corr.tick_params(axis="y", rotation=0, labelsize=8.5)
            cbar = ax_corr.collections[0].colorbar
            cbar.ax.tick_params(labelsize=7, colors=MUTED)
            cbar.outline.set_edgecolor(BORDER)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
        elif has_bars:
            # No corr but has bars — put first bar in left cell
            col = bar_candidates[0]
            ax_bl = fig.add_subplot(gs[r, 0])
            _style(ax_bl)
            _draw_bar(ax_bl, df, col, ACCENT,
                      horizontal=(df[col].nunique() > 4 or df[col].dtype == object))

        # Right — first bar chart
        if has_bars:
            col = bar_candidates[0]
            ax_bar = fig.add_subplot(gs[r, 1])
            _style(ax_bar)
            _draw_bar(ax_bar, df, col, ACCENT,
                      horizontal=(df[col].nunique() > 4 or df[col].dtype == object))

    # ── Row 2: Missing data (left) | Scatter plot (right) ──────────────────────
    if "miss_scatter" in rows:
        r = rows.index("miss_scatter")

        # Left — missing data chart
        if has_missing:
            mv = missing_df[missing_df["missing"] > 0].sort_values("percent", ascending=True)
            ax_miss = fig.add_subplot(gs[r, 0])
            _style(ax_miss)
            bar_colors_m = [RED if p > 50 else ORANGE if p > 20 else YELLOW for p in mv["percent"]]
            bars_m = ax_miss.barh(mv["column"], mv["percent"],
                                  color=bar_colors_m, edgecolor=DARK_BG,
                                  linewidth=0.5, height=0.55)
            for bar, pct, cnt in zip(bars_m, mv["percent"], mv["missing"]):
                ax_miss.text(bar.get_width() + 0.8,
                             bar.get_y() + bar.get_height() / 2,
                             f"{pct:.1f}%  ·  {cnt:,} rows",
                             va="center", fontsize=8, color=MUTED, fontfamily="monospace")
            ax_miss.set_xlim(0, min(mv["percent"].max() * 1.45, 108))
            ax_miss.set_xlabel("% missing", fontsize=8)
            ax_miss.set_title("Missing data", fontsize=10, fontweight="bold")
            ax_miss.grid(axis="x", color=BORDER, linewidth=0.4)
            ax_miss.grid(axis="y", visible=False)
            legend_patches = [
                mpatches.Patch(color=YELLOW, label="< 20%"),
                mpatches.Patch(color=ORANGE, label="20–50%"),
                mpatches.Patch(color=RED,    label="≥ 50%"),
            ]
            ax_miss.legend(handles=legend_patches, fontsize=7.5, facecolor=CARD,
                           edgecolor=BORDER, labelcolor=TEXT, loc="lower right")
        elif has_bars and len(bar_candidates) >= 2:
            # No missing data — show second bar chart in left cell instead
            col2 = bar_candidates[1]
            ax_bl2 = fig.add_subplot(gs[r, 0])
            _style(ax_bl2)
            _draw_bar(ax_bl2, df, col2, ACCENT2,
                      horizontal=(df[col2].nunique() > 4 or df[col2].dtype == object))

        # Right — scatter plot
        if has_scatter:
            ax_sc = fig.add_subplot(gs[r, 1])
            _style(ax_sc)
            x_col, y_col = continuous[0], continuous[1]

            color_col = None
            if cat_cols:
                color_col = cat_cols[0]
            elif discrete:
                color_col = discrete[0]

            if color_col:
                unique_vals = df[color_col].dropna().unique()
                for j, val in enumerate(unique_vals[:8]):
                    subset = df[df[color_col] == val][[x_col, y_col]].dropna()
                    ax_sc.scatter(subset[x_col], subset[y_col],
                                  color=PALETTE[j % len(PALETTE)],
                                  s=22, alpha=0.6, edgecolors="none",
                                  label=str(val))
                ax_sc.legend(fontsize=7.5, facecolor=CARD, edgecolor=BORDER,
                             labelcolor=TEXT, loc="best",
                             title=color_col, title_fontsize=7.5,
                             markerscale=1.3)
                ax_sc.set_title(f"{x_col} vs {y_col}  (colour = {color_col})",
                                fontsize=10, fontweight="bold")
            else:
                sc_df = df[[x_col, y_col]].dropna()
                ax_sc.scatter(sc_df[x_col], sc_df[y_col],
                              c=ACCENT, s=22, alpha=0.5, edgecolors="none")
                ax_sc.set_title(f"{x_col} vs {y_col}", fontsize=10, fontweight="bold")
            ax_sc.set_xlabel(x_col, fontsize=9)
            ax_sc.set_ylabel(y_col, fontsize=9)
        elif has_bars and len(bar_candidates) >= 2 and not has_scatter:
            # No scatter — second bar chart on right
            col2 = bar_candidates[1 if has_missing else (2 if len(bar_candidates) > 2 else 1)]
            ax_br = fig.add_subplot(gs[r, 1])
            _style(ax_br)
            _draw_bar(ax_br, df, col2, ACCENT2,
                      horizontal=(df[col2].nunique() > 4 or df[col2].dtype == object))

    return _fig_to_b64(fig)


# ── Time series forecast ───────────────────────────────────────────────────────

def detect_time_series(df: pd.DataFrame):
    if df.empty:
        return None
    datetime_col = None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            datetime_col = col
            break
        try:
            parsed = pd.to_datetime(df[col])
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

    ts = ts.sort_values(datetime_col).set_index(datetime_col)
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

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    history_tail = ts.tail(90)
    ax.fill_between(history_tail.index, history_tail.values, alpha=0.12, color=ACCENT)
    ax.plot(history_tail.index, history_tail.values, color=ACCENT, linewidth=1.8, label="History")
    ax.plot(forecast.index, forecast.values, color=RED, linewidth=2,
            linestyle="--", label="7-day forecast", marker="o", markersize=4)
    ax.axvspan(forecast.index[0], forecast.index[-1], alpha=0.06, color=RED)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_title(f"7-day forecast  ·  {target}", fontsize=10, color=TEXT, fontweight="bold")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    fig.tight_layout()
    chart = _fig_to_b64(fig)

    return target, pd.DataFrame({"history": ts.tail(10)}), chart, forecast_df


# ── LLM narrative ─────────────────────────────────────────────────────────────

def generate_llm_insights(
    summary: Dict,
    sample_rows: List[Dict],
    forecast: Optional[Dict],
    api_key: Optional[str],
) -> Optional[str]:
    if not api_key:
        return None
    client = Groq(api_key=api_key)
    system_prompt = {
        "role": "system",
        "content": (
            "You are a senior data analyst. Produce a structured, quantitative report.\n"
            "Use exactly these sections:\n"
            "## Key Insights\n## Business Implications\n## Data Quality Issues\n## Recommendations\n\n"
            "Be specific — reference actual column names, values, and numbers. No filler."
        ),
    }
    missing_df  = summary.get("missing", pd.DataFrame())
    num_summary = summary.get("numeric_summary", pd.DataFrame())
    payload = {
        "meta": summary.get("meta"),
        "continuous_columns": summary.get("continuous_columns"),
        "discrete_columns": summary.get("discrete_columns"),
        "categorical_columns": summary.get("categorical_columns"),
        "missing": missing_df.to_dict(orient="records") if not missing_df.empty else [],
        "numeric_summary": num_summary.to_dict(orient="records") if not num_summary.empty else [],
        "top_categories": summary.get("top_categories", []),
        "sample_rows": sample_rows,
        "forecast_head": forecast.get("forecast_head") if forecast else None,
    }
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_prompt, {"role": "user", "content": json.dumps(payload, default=str)}],
        temperature=0.3,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


# ── Entry point ────────────────────────────────────────────────────────────────

def analyze_dataset(file_bytes: bytes, filename: str, api_key: Optional[str]) -> Dict:
    df = load_dataframe(file_bytes, filename)
    summary = summarize_dataframe(df)
    dashboard_b64 = build_dashboard(df, summary)

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
        "dashboard": dashboard_b64,
        "forecast": forecast_payload,
        "insights": insights,
    }