# Pool Practice Analyzer

Simple Streamlit app for pool players who want to upload a CSV or Excel file with:

- `date`
- `attempt`
- `balls_potted`

The app shows:

- histograms with mean and median reference lines
- all-time, yearly, monthly, and rolling 12-month breakdowns
- summary statistics including mean, median, standard deviation, quartiles, 95th percentile, IQR, min, max, range, total balls potted, and number of attempts

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or with the included virtual environment pattern used during setup here:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Input format

Required columns:

- `date`
- `attempt`
- `balls_potted`

The app accepts `.csv`, `.xlsx`, and `.xls` files.

## Sample data

The app includes built-in sample datasets and also lets users download example CSV files directly from the sidebar.

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Keep `requirements.txt` at the repo root so Streamlit installs dependencies automatically.
