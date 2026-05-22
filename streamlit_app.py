from __future__ import annotations

import io
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from hvac_v3_engine import (
    load_comparison_file,
    compute_metrics,
    monthly_error_summary,
    yearly_error_summary,
    train_residual_correction,
    apply_monthly_bias_correction,
)

st.set_page_config(page_title="Improved ROM Validation Bundle", layout="wide")
st.title("Reduced-Order-Model-ROM--Degradation: Improved Validation Bundle")
st.caption("DesignBuilder comparison, monthly error, and residual correction to reduce daily MAPE/CVRMSE.")

with st.expander("Required input format", expanded=False):
    st.write(
        "Upload CSV/TXT/XLSX with at least three columns: date, DesignBuilder/reference energy, and ROM/model energy. "
        "Optional columns can improve correction: outdoor_temp, solar_radiation, occupancy_factor."
    )

uploaded = st.file_uploader("Upload validation data", type=["csv", "txt", "tsv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a comparison file to start. The included example CSV can be used for testing.")
    st.stop()

try:
    df = load_comparison_file(uploaded)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

st.success(f"Loaded {len(df):,} records from {df['date'].min().date()} to {df['date'].max().date()}.")

# Original validation
metrics_original = compute_metrics(df["reference"], df["rom_original"]).as_dict()
monthly_original = monthly_error_summary(df, pred_col="rom_original")
yearly_original = yearly_error_summary(df, pred_col="rom_original")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Aggregate error", f"{metrics_original['aggregate_pct_error']:.2f}%")
k2.metric("Daily MAPE", f"{metrics_original['mape_pct']:.2f}%")
k3.metric("Daily CVRMSE", f"{metrics_original['cvrmse_pct']:.2f}%")
k4.metric("Monthly MAPE", f"{monthly_original['monthly_abs_pct_error'].mean():.2f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Daily data", "Monthly validation", "Error reduction", "Export"])

with tab1:
    preview = df.copy()
    preview["error"] = preview["rom_original"] - preview["reference"]
    preview["pct_error"] = preview["error"] / preview["reference"] * 100
    st.dataframe(preview.head(200), use_container_width=True)

with tab2:
    st.subheader("Monthly error summary")
    st.dataframe(monthly_original, use_container_width=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly_original["month"], monthly_original["monthly_pct_error"], marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Monthly percentage error (%)")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.subheader("Correction layer to reduce daily error")
    method = st.radio(
        "Correction method",
        ["ML residual correction", "Monthly bias correction"],
        help="ML residual correction uses calendar/weather/occupancy/thermal-lag features. Monthly bias correction applies month-specific reference/model ratios."
    )
    holdout = st.checkbox("Use last year as holdout validation", value=True)
    train_mask = None
    if holdout:
        last_year = df["date"].dt.year.max()
        train_mask = df["date"].dt.year < last_year
        st.info(f"Training/calibration years: < {last_year}; test/holdout year: {last_year}.")
    if st.button("Run correction"):
        if method == "Monthly bias correction":
            corrected, ratio_table = apply_monthly_bias_correction(df, train_mask=train_mask)
            pred_col = "rom_monthly_bias_corrected"
            st.write("Monthly correction factors")
            st.dataframe(ratio_table, use_container_width=True)
        else:
            corrected, meta = train_residual_correction(df, train_mask=train_mask)
            pred_col = "rom_residual_corrected"
            st.write(f"Correction method used: {meta.get('method')}")

        eval_mask = ~train_mask if holdout and train_mask is not None else pd.Series(True, index=df.index)
        before = compute_metrics(corrected.loc[eval_mask, "reference"], corrected.loc[eval_mask, "rom_original"]).as_dict()
        after = compute_metrics(corrected.loc[eval_mask, "reference"], corrected.loc[eval_mask, pred_col]).as_dict()
        comparison = pd.DataFrame([before, after], index=["Before correction", "After correction"])
        st.dataframe(comparison, use_container_width=True)
        monthly_after = monthly_error_summary(corrected, pred_col=pred_col)
        st.write("Corrected monthly error")
        st.dataframe(monthly_after, use_container_width=True)
        st.session_state["corrected"] = corrected
        st.session_state["pred_col"] = pred_col
        st.session_state["monthly_after"] = monthly_after

with tab4:
    st.subheader("Download outputs")
    monthly_csv = monthly_original.to_csv(index=False).encode("utf-8")
    yearly_csv = yearly_original.to_csv(index=False).encode("utf-8")
    st.download_button("Download monthly_error_original.csv", monthly_csv, "monthly_error_original.csv")
    st.download_button("Download yearly_error_original.csv", yearly_csv, "yearly_error_original.csv")
    if "corrected" in st.session_state:
        corrected = st.session_state["corrected"]
        pred_col = st.session_state["pred_col"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("daily_original_and_corrected.csv", corrected.to_csv(index=False))
            zf.writestr("monthly_error_corrected.csv", st.session_state["monthly_after"].to_csv(index=False))
            metrics = pd.DataFrame([
                compute_metrics(corrected["reference"], corrected["rom_original"]).as_dict(),
                compute_metrics(corrected["reference"], corrected[pred_col]).as_dict(),
            ], index=["original", "corrected"])
            zf.writestr("metrics_original_vs_corrected.csv", metrics.to_csv())
        st.download_button("Download corrected validation ZIP", buffer.getvalue(), "corrected_validation_outputs.zip")
