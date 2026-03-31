# Data Insight Studio (Streamlit)

Streamlit app to upload a CSV or Excel file, generate quick statistical insights, simple charts, and a short AI-written narrative via Groq. No data is persisted; everything stays in memory for the request.

## Features
- Upload CSV/XLS/XLSX via Streamlit file uploader
- Summary stats and missingness report
- Correlation heatmap and a sample distribution plot
- Lightweight 7-day forecast when a datetime column and numeric target exist (Exponential Smoothing)
- Optional Groq-powered narrative (set `GROQ_API_KEY`)

## Quickstart
1. Install dependencies (use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Groq API key (for AI narrative):
   ```bash
   # Option A: .env file (already supported)
   echo GROQ_API_KEY=your_key_here > .env

   # Option B: environment variable
   set GROQ_API_KEY=your_key_here   # PowerShell
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```
4. Open the displayed local URL and upload a CSV/Excel file.

## Notes
- Forecasting uses daily resampling and Exponential Smoothing; if no suitable datetime/target columns exist, the forecast block is omitted.
- Only the first few rows and aggregated metadata are sent to Groq; raw files are not stored.
- Supported file types: `.csv`, `.xls`, `.xlsx`.
