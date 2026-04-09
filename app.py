from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import random
import numpy as np
import pandas as pd
import streamlit as st

MPL_CONFIG_DIR = Path(__file__).parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


plt.style.use("fivethirtyeight")

REQUIRED_COLUMNS = ["date", "attempt", "balls_potted"]
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "jamesplayspool-logo.png"
YOUTUBE_URL = "https://www.youtube.com/channel/UCo4gZG2ly_pMf5w249OPsNg"
PAYPAL_URL = "https://paypal.me/jamesplayspool"


@dataclass(frozen=True)
class TimeSlice:
    label: str
    frame: pd.DataFrame


def generate_sample_balls_potted(rng: random.Random, adjustment: float = 0.0) -> int:
    # Base distribution targets a mean near 43, a standard deviation near 12,
    # and a slight skew toward lower values while staying within 0 to 75.
    base_mean = 44.0 + adjustment
    base_sd = 11.5
    shape = (base_mean / base_sd) ** 2
    scale = (base_sd**2) / base_mean

    if rng.random() < 0.014:
        sample = rng.randint(0, 8)
    else:
        sample = rng.gammavariate(shape, scale)

    return int(max(0, min(75, round(sample))))


def generate_sample_dataset(kind: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if kind == "consistent":
        rng = random.Random(42)
        current = date(2024, 1, 1)
        end = date(2025, 3, 31)
        day_index = 0
        while current <= end:
            if current.weekday() in {1, 3, 5}:
                attempts = rng.randint(2, 5)
                daily_adjustment = ((day_index % 7) - 3) * 0.12
                for attempt in range(1, attempts + 1):
                    attempt_adjustment = (attempt - (attempts + 1) / 2) * 0.2
                    balls = generate_sample_balls_potted(rng, daily_adjustment + attempt_adjustment)
                    rows.append(
                        {"date": current.isoformat(), "attempt": attempt, "balls_potted": balls}
                    )
            current += timedelta(days=1)
            day_index += 1
    else:
        rng = random.Random(7)
        current = date(2024, 1, 1)
        end = date(2025, 3, 31)
        while current <= end:
            if current.weekday() in {0, 2, 4, 6}:
                attempts = rng.randint(1, 5)
                seasonal_adjustment = (
                    (current.month in {5, 6, 7, 8}) * 1.0
                    - (current.month in {11, 12, 1, 2}) * 0.8
                    - (current.month == 10) * 0.7
                    + 1.2
                )
                for attempt in range(1, attempts + 1):
                    attempt_adjustment = (attempt - 3) * 0.25
                    balls = generate_sample_balls_potted(rng, seasonal_adjustment + attempt_adjustment)
                    rows.append(
                        {"date": current.isoformat(), "attempt": attempt, "balls_potted": balls}
                    )
            current += timedelta(days=1)

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    normalized = raw_df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "Your file is missing required columns: "
            + ", ".join(missing_columns)
            + ". Expected columns are date, attempt, balls_potted."
        )

    normalized = normalized[REQUIRED_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["attempt"] = pd.to_numeric(normalized["attempt"], errors="coerce")
    normalized["balls_potted"] = pd.to_numeric(normalized["balls_potted"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "attempt", "balls_potted"])

    normalized["attempt"] = normalized["attempt"].astype(int)
    normalized["balls_potted"] = normalized["balls_potted"].astype(float)
    normalized = normalized.sort_values(["date", "attempt"]).reset_index(drop=True)
    return normalized


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["year"] = enriched["date"].dt.year
    enriched["month_period"] = enriched["date"].dt.to_period("M")
    enriched["month_label"] = enriched["month_period"].astype(str)
    return enriched


def compute_summary(series: pd.Series) -> pd.DataFrame:
    clean = series.dropna().astype(float)
    if clean.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    mean_value = clean.mean()
    std_value = clean.std(ddof=1)

    stats = [
        ("Attempts", len(clean)),
        ("Total Balls Potted", clean.sum()),
        ("Mean", mean_value),
        ("Median", clean.median()),
        ("Standard Deviation", std_value if not np.isnan(std_value) else 0.0),
        ("Minimum", clean.min()),
        ("25th Percentile", q1),
        ("75th Percentile", q3),
        ("95th Percentile", clean.quantile(0.95)),
        ("Maximum", clean.max()),
        ("Interquartile Range", q3 - q1),
        ("Range", clean.max() - clean.min()),
        ("Coefficient of Variation", (std_value / mean_value) if mean_value else np.nan),
    ]

    summary_df = pd.DataFrame(stats, columns=["Metric", "Value"])
    return summary_df


def summary_by_group(df: pd.DataFrame, group_column: str, label_name: str) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_column):
        stats_df = compute_summary(group_df["balls_potted"])
        record = {label_name: str(group_value)}
        for _, row in stats_df.iterrows():
            record[row["Metric"]] = row["Value"]
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    grouped_summary = pd.DataFrame(rows)
    return grouped_summary.sort_values(label_name).reset_index(drop=True)


def build_rolling_windows(df: pd.DataFrame) -> list[TimeSlice]:
    windows: list[TimeSlice] = []
    month_ends = pd.period_range(
        df["date"].min().to_period("M"),
        df["date"].max().to_period("M"),
        freq="M",
    )

    for period in month_ends:
        window_end = period.end_time.normalize()
        window_start = (period - 11).start_time.normalize()
        mask = (df["date"] >= window_start) & (df["date"] <= window_end)
        window_df = df.loc[mask].copy()
        if not window_df.empty:
            label = f"{window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}"
            windows.append(TimeSlice(label=label, frame=window_df))

    return windows


def make_histogram(series: pd.Series, title: str):
    clean = series.dropna().astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clean, bins="auto", edgecolor="black", alpha=0.85)
    ax.axvline(clean.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {clean.mean():.2f}")
    ax.axvline(clean.median(), color="#1f77b4", linestyle="-.", linewidth=2, label=f"Median: {clean.median():.2f}")
    ax.set_title(title)
    ax.set_xlabel("Balls Potted")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


def make_download_bytes(df: pd.DataFrame, file_type: str) -> bytes:
    if file_type == "csv":
        return df.to_csv(index=False).encode("utf-8")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def render_sidebar_branding() -> None:
    if LOGO_PATH.exists():
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <a href="{YOUTUBE_URL}" target="_blank">
                    <img
                        src="data:image/png;base64,{encoded_logo}"
                        alt="JamesPlaysPool logo"
                        style="width: 100%; max-width: 240px; border-radius: 12px;"
                    />
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.sidebar.link_button("Watch on YouTube", YOUTUBE_URL, use_container_width=True)
    st.sidebar.link_button("Tip via PayPal", PAYPAL_URL, use_container_width=True)
    st.sidebar.divider()


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Data Overview")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Rows", f"{len(df):,}")
    overview_columns[1].metric("Days Tracked", f"{df['date'].dt.date.nunique():,}")
    overview_columns[2].metric("First Date", df["date"].min().strftime("%Y-%m-%d"))
    overview_columns[3].metric("Last Date", df["date"].max().strftime("%Y-%m-%d"))

    if (df["attempt"] < 1).any() or (df["attempt"] > 5).any():
        st.warning("Some attempt values fall outside the expected 1 to 5 range.")

    duplicates = df.duplicated(subset=["date", "attempt"]).sum()
    if duplicates:
        st.warning(
            f"There are {duplicates} duplicate date/attempt rows. The app still analyzes them, "
            "but you may want to clean the file."
        )


def render_summary_block(title: str, frame: pd.DataFrame) -> None:
    st.markdown(f"### {title}")
    summary_df = compute_summary(frame["balls_potted"])
    st.dataframe(
        summary_df.style.format({"Value": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    histogram = make_histogram(frame["balls_potted"], f"{title} Distribution")
    st.pyplot(histogram)
    plt.close(histogram)


def render_grouped_section(df: pd.DataFrame, group_column: str, label_name: str, title: str) -> None:
    grouped_values = sorted(df[group_column].astype(str).unique())
    st.markdown(f"### {title}")
    selected_label = st.selectbox(
        f"Choose a {label_name.lower()}",
        grouped_values,
        key=f"select_{group_column}",
    )
    filtered = df[df[group_column].astype(str) == selected_label]
    render_summary_block(f"{title}: {selected_label}", filtered)
    st.markdown(f"#### {title} Summary Table")
    grouped_summary = summary_by_group(df, group_column, label_name)
    st.dataframe(grouped_summary.style.format(precision=2), use_container_width=True, hide_index=True)


def render_rolling_section(df: pd.DataFrame) -> None:
    rolling_windows = build_rolling_windows(df)
    st.markdown("### Rolling 12 Months")
    if not rolling_windows:
        st.info("Not enough data to build rolling windows.")
        return

    labels = [window.label for window in rolling_windows]
    selected_label = st.selectbox("Choose a rolling 12-month window", labels, key="rolling_window_select")
    selected_window = next(window for window in rolling_windows if window.label == selected_label)
    render_summary_block(f"Rolling 12 Months: {selected_label}", selected_window.frame)

    summary_rows = []
    for window in rolling_windows:
        record = {"Window": window.label}
        stats_df = compute_summary(window.frame["balls_potted"])
        for _, row in stats_df.iterrows():
            record[row["Metric"]] = row["Value"]
        summary_rows.append(record)

    st.markdown("#### Rolling 12-Month Summary Table")
    st.dataframe(pd.DataFrame(summary_rows).style.format(precision=2), use_container_width=True, hide_index=True)


def build_sidebar() -> tuple[pd.DataFrame | None, str]:
    render_sidebar_branding()
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    sample_options = {
        "None": None,
        "Consistent practice example": "consistent",
        "Variable practice example": "variable",
    }
    selected_sample = st.sidebar.selectbox("Or load a sample dataset", list(sample_options.keys()))

    if uploaded_file is not None:
        data_source = f"Uploaded file: {uploaded_file.name}"
        return load_uploaded_file(uploaded_file), data_source

    sample_key = sample_options[selected_sample]
    if sample_key is not None:
        data_source = f"Built-in sample: {selected_sample}"
        return generate_sample_dataset(sample_key), data_source

    return None, "No file selected"


def main() -> None:
    st.set_page_config(page_title="Pool Practice Analyzer", layout="wide")
    st.title("Pool Practice Analyzer")
    st.write(
        "Upload a spreadsheet with `date`, `attempt`, and `balls_potted` columns to get "
        "simple histograms and summary statistics for your pool practice."
    )

    raw_df, data_source = build_sidebar()

    st.sidebar.markdown("### Sample File Templates")
    sample_downloads = {
        "sample_pool_log_consistent.csv": generate_sample_dataset("consistent"),
        "sample_pool_log_variable.csv": generate_sample_dataset("variable"),
    }
    for file_name, sample_df in sample_downloads.items():
        st.sidebar.download_button(
            label=f"Download {file_name}",
            data=make_download_bytes(sample_df, "csv"),
            file_name=file_name,
            mime="text/csv",
        )

    if raw_df is None:
        st.info("Start by uploading a CSV or Excel file, or load one of the sample datasets from the sidebar.")
        return

    try:
        df = prepare_dataframe(raw_df)
    except ValueError as error:
        st.error(str(error))
        return

    if df.empty:
        st.error("No valid rows were found after parsing the file.")
        return

    df = add_time_columns(df)

    st.caption(data_source)
    render_overview(df)

    with st.expander("Preview parsed data", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    tabs = st.tabs(["All Time", "Yearly", "Monthly", "Rolling 12 Months"])

    with tabs[0]:
        render_summary_block("All Time", df)

    with tabs[1]:
        render_grouped_section(df, "year", "Year", "Yearly")

    with tabs[2]:
        render_grouped_section(df, "month_label", "Month", "Monthly")

    with tabs[3]:
        render_rolling_section(df)


if __name__ == "__main__":
    main()
