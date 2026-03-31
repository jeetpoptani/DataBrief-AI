# DataBrief AI

DataBrief AI is a Streamlit-based analytics app that turns raw CSV or Excel files into:
- quick statistical insights,
- visual summaries,
- an optional short-term forecast,
- and an AI-generated business narrative using Groq.

The app is designed for fast exploratory analysis with minimal setup.

## What The Project Does

1. Upload a dataset (`.csv`, `.xls`, `.xlsx`).
2. Automatically detect data types and generate profiling outputs.
3. Show core analysis views:
   - dataset metadata,
   - numeric summary,
   - missing-value report,
   - correlation heatmap,
   - distribution chart.
4. Attempt a 7-day forecast if a valid datetime column and numeric target are available.
5. Generate a structured AI narrative (insights, implications, risks, recommendations) if `GROQ_API_KEY` is set.

## How It Works

### 1. File Ingestion
- The Streamlit UI accepts file upload.
- The analyzer reads file content in memory:
  - CSV via `pandas.read_csv`
  - Excel via `pandas.read_excel`

### 2. Data Profiling
- Computes row/column counts and data types.
- Builds numeric descriptive statistics.
- Computes per-column missing counts and percentages.
- Detects top values for categorical columns.

### 3. Visualization
- Correlation heatmap for numeric columns.
- Histogram + KDE distribution for one numeric column.
- Missingness bar chart.
- Charts are generated with Matplotlib/Seaborn and returned as base64 images for Streamlit rendering.

### 4. Forecasting (Conditional)
- Tries to detect a datetime column.
- Picks a numeric target column.
- Resamples to daily series and fills missing values.
- Trains an Exponential Smoothing model (`statsmodels`).
- Produces a 7-day forecast and chart.
- If requirements are not met, forecasting is skipped gracefully.

### 5. AI Narrative (Optional)
- If `GROQ_API_KEY` is available, the app sends compact analysis payloads to Groq.
- Output is a concise structured summary for decision-making.
- If no API key is set, the app still works with full non-LLM analytics.

## Tech Stack

- Python 3
- Streamlit (UI)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)
- Statsmodels (forecasting)
- Groq Python SDK (LLM narrative)
- python-dotenv (environment variable loading)

## Project Structure

```
main.py                # Streamlit app entrypoint
app/
  analyzer.py          # Data loading, profiling, plotting, forecast, Groq integration
  __init__.py
requirements.txt
README.md
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Add Groq key in `.env`:

```env
GROQ_API_KEY=your_key_here
```

4. Run the app:

```bash
streamlit run main.py
```

5. Open the local Streamlit URL and upload your dataset.

## Important Notes

- No long-term persistence is implemented in this version.
- The app handles unsupported file types with a clear error.
- Forecasting is best-effort and only appears for suitable time-series data.
