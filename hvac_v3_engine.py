"""
Improved ROM validation and residual-correction engine.
Project: Reduced-Order-Model-ROM--Degradation

This module does not replace the physical HVAC degradation ROM. It adds a
calibration/validation layer to reduce short-term daily error against a
DesignBuilder/EnergyPlus-style reference while preserving the ROM as the
physical baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


@dataclass
class MetricBlock:
    n: int
    reference_total: float
    model_total: float
    aggregate_pct_error: float
    mae: float
    rmse: float
    mape_pct: float
    cvrmse_pct: float
    nmbe_pct: float
    r2: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "reference_total": self.reference_total,
            "model_total": self.model_total,
            "aggregate_pct_error": self.aggregate_pct_error,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape_pct": self.mape_pct,
            "cvrmse_pct": self.cvrmse_pct,
            "nmbe_pct": self.nmbe_pct,
            "r2": self.r2,
        }


def _safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    den = np.where(np.abs(den) < eps, np.nan, den)
    return num / den


def compute_metrics(reference: Iterable[float], prediction: Iterable[float]) -> MetricBlock:
    """Compute validation metrics using reference as the denominator."""
    y = np.asarray(reference, dtype=float)
    p = np.asarray(prediction, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        raise ValueError("No finite data points available for metric calculation.")
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.nanmean(np.abs(_safe_divide(e, y))) * 100)
    mean_y = float(np.mean(y))
    cvrmse = float(rmse / mean_y * 100) if abs(mean_y) > 1e-12 else np.nan
    nmbe = float(np.mean(e) / mean_y * 100) if abs(mean_y) > 1e-12 else np.nan
    agg = float((np.sum(p) - np.sum(y)) / np.sum(y) * 100) if abs(np.sum(y)) > 1e-12 else np.nan
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return MetricBlock(
        n=int(len(y)),
        reference_total=float(np.sum(y)),
        model_total=float(np.sum(p)),
        aggregate_pct_error=agg,
        mae=mae,
        rmse=rmse,
        mape_pct=mape,
        cvrmse_pct=cvrmse,
        nmbe_pct=nmbe,
        r2=r2,
    )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common DesignBuilder/ROM comparison column names."""
    raw_cols = list(df.columns)
    lower = {str(c).strip().lower().replace(" ", "_"): c for c in raw_cols}

    # Date column
    date_candidates = ["date/time", "date", "datetime", "time", "date_time"]
    reference_candidates = ["design_builder", "designbuilder", "reference", "actual", "measured", "total_design_builder"]
    model_candidates = ["model", "rom", "prediction", "predicted", "reduced_order_model", "total_model"]

    rename = {}
    for cand in date_candidates:
        key = cand.replace(" ", "_").lower()
        if key in lower:
            rename[lower[key]] = "date"
            break
    for cand in reference_candidates:
        key = cand.replace(" ", "_").lower()
        if key in lower:
            rename[lower[key]] = "reference"
            break
    for cand in model_candidates:
        key = cand.replace(" ", "_").lower()
        if key in lower:
            rename[lower[key]] = "rom_original"
            break

    # Special case for two identical Total columns after export.
    if len(raw_cols) >= 3 and "reference" not in rename.values() and "rom_original" not in rename.values():
        rename[raw_cols[1]] = "reference"
        rename[raw_cols[2]] = "rom_original"
    if len(raw_cols) >= 1 and "date" not in rename.values():
        rename[raw_cols[0]] = "date"

    out = df.rename(columns=rename).copy()
    required = {"date", "reference", "rom_original"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after parsing: {sorted(missing)}. Available: {list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["reference"] = pd.to_numeric(out["reference"], errors="coerce")
    out["rom_original"] = pd.to_numeric(out["rom_original"], errors="coerce")
    out = out.dropna(subset=["date", "reference", "rom_original"]).sort_values("date").reset_index(drop=True)
    return out


def load_comparison_file(path_or_buffer) -> pd.DataFrame:
    """Load CSV, TSV, TXT, or XLSX comparison data and standardize columns."""
    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_buffer)
    else:
        # Try tab-delimited two-row header format first, then normal CSV.
        try:
            df = pd.read_csv(path_or_buffer, sep="\t", skiprows=2, names=["date", "reference", "rom_original"])
        except Exception:
            df = pd.read_csv(path_or_buffer)
    return _standardize_columns(df)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for daily residual correction."""
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofyear"] = out["date"].dt.dayofyear
    out["dayofweek"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_weather_features(
    df: pd.DataFrame,
    temp_col: str = "outdoor_temp",
    solar_col: str = "solar_radiation",
    occupancy_col: str = "occupancy_factor",
    cooling_base: float = 24.0,
    heating_base: float = 18.0,
) -> pd.DataFrame:
    """Add optional weather and occupancy features if those columns exist.

    The function is safe when weather columns are absent; it simply creates
    neutral placeholders so the same pipeline can run on calendar-only data.
    """
    out = df.copy()
    if temp_col in out.columns:
        t = pd.to_numeric(out[temp_col], errors="coerce")
        out["CDD"] = np.maximum(t - cooling_base, 0)
        out["HDD"] = np.maximum(heating_base - t, 0)
        out["temp_lag1"] = t.shift(1).bfill()
        out["temp_lag2"] = t.shift(2).bfill()
        out["temp_roll3"] = t.rolling(3, min_periods=1).mean()
        # Simple temperature-dependent COP modifier: values >1 imply higher electricity
        # demand at high outdoor temperature. It is used as a feature, not as a blind
        # multiplier, so the residual model can learn whether it improves prediction.
        out["cop_modifier"] = np.where(t > cooling_base, 1.0 + 0.015 * (t - cooling_base), 1.0)
    else:
        out["CDD"] = 0.0
        out["HDD"] = 0.0
        out["temp_lag1"] = 0.0
        out["temp_lag2"] = 0.0
        out["temp_roll3"] = 0.0
        out["cop_modifier"] = 1.0
    if solar_col in out.columns:
        out["solar_feature"] = pd.to_numeric(out[solar_col], errors="coerce").fillna(0)
    else:
        out["solar_feature"] = 0.0
    if occupancy_col in out.columns:
        out["occupancy_feature"] = pd.to_numeric(out[occupancy_col], errors="coerce").fillna(1.0)
    else:
        out["occupancy_feature"] = np.where(out.get("is_weekend", 0) == 1, 0.3, 1.0)
    return out


def add_rom_lag_features(df: pd.DataFrame, rom_col: str = "rom_original") -> pd.DataFrame:
    """Add thermal-memory and model-shape features."""
    out = df.copy()
    s = pd.to_numeric(out[rom_col], errors="coerce")
    out["rom_lag1"] = s.shift(1).bfill()
    out["rom_lag2"] = s.shift(2).bfill()
    out["rom_lag7"] = s.shift(7).bfill()
    out["rom_roll3"] = s.rolling(3, min_periods=1).mean()
    out["rom_roll7"] = s.rolling(7, min_periods=1).mean()
    return out


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = add_calendar_features(df)
    out = add_weather_features(out)
    out = add_rom_lag_features(out)
    feature_cols = [
        "rom_original", "rom_lag1", "rom_lag2", "rom_lag7", "rom_roll3", "rom_roll7",
        "month", "dayofweek", "is_weekend", "dayofyear", "sin_doy", "cos_doy", "sin_month", "cos_month",
        "CDD", "HDD", "temp_lag1", "temp_lag2", "temp_roll3", "cop_modifier", "solar_feature", "occupancy_feature",
    ]
    return out, feature_cols


def monthly_error_summary(df: pd.DataFrame, pred_col: str = "rom_original", reference_col: str = "reference") -> pd.DataFrame:
    out = df.copy()
    out["month_period"] = out["date"].dt.to_period("M").astype(str)
    out["error"] = out[pred_col] - out[reference_col]
    out["ape_pct"] = np.abs(out["error"] / out[reference_col]) * 100
    grouped = out.groupby("month_period").agg(
        year=("date", lambda x: int(x.iloc[0].year)),
        month_no=("date", lambda x: int(x.iloc[0].month)),
        days=("date", "count"),
        reference_kWh=(reference_col, "sum"),
        model_kWh=(pred_col, "sum"),
        daily_MAPE_pct=("ape_pct", "mean"),
    ).reset_index().rename(columns={"month_period": "month"})
    grouped["error_kWh"] = grouped["model_kWh"] - grouped["reference_kWh"]
    grouped["monthly_pct_error"] = grouped["error_kWh"] / grouped["reference_kWh"] * 100
    grouped["monthly_abs_pct_error"] = grouped["monthly_pct_error"].abs()
    grouped["reference_MWh"] = grouped["reference_kWh"] / 1000
    grouped["model_MWh"] = grouped["model_kWh"] / 1000
    grouped["error_MWh"] = grouped["error_kWh"] / 1000
    return grouped


def yearly_error_summary(df: pd.DataFrame, pred_col: str = "rom_original", reference_col: str = "reference") -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["error"] = out[pred_col] - out[reference_col]
    out["ape_pct"] = np.abs(out["error"] / out[reference_col]) * 100
    grouped = out.groupby("year").agg(
        days=("date", "count"),
        reference_kWh=(reference_col, "sum"),
        model_kWh=(pred_col, "sum"),
        daily_MAPE_pct=("ape_pct", "mean"),
    ).reset_index()
    grouped["error_kWh"] = grouped["model_kWh"] - grouped["reference_kWh"]
    grouped["pct_error"] = grouped["error_kWh"] / grouped["reference_kWh"] * 100
    grouped["reference_MWh"] = grouped["reference_kWh"] / 1000
    grouped["model_MWh"] = grouped["model_kWh"] / 1000
    grouped["error_MWh"] = grouped["error_kWh"] / 1000
    return grouped


def apply_monthly_bias_correction(
    df: pd.DataFrame,
    train_mask: Optional[pd.Series] = None,
    pred_col: str = "rom_original",
    reference_col: str = "reference",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correct the ROM by month-specific reference/model energy ratios.

    Use this when the ROM annual total is good but monthly seasonality is shifted.
    For manuscript validation, estimate the ratios using training years only and
    test on a held-out year.
    """
    out = df.copy()
    out["month"] = out["date"].dt.month
    if train_mask is None:
        train_mask = pd.Series(True, index=out.index)
    train = out.loc[train_mask].copy()
    ratio = train.groupby("month").apply(lambda g: g[reference_col].sum() / max(g[pred_col].sum(), 1e-9))
    ratio_table = ratio.rename("monthly_correction_factor").reset_index()
    out = out.merge(ratio_table, on="month", how="left")
    out["monthly_correction_factor"] = out["monthly_correction_factor"].fillna(1.0)
    out["rom_monthly_bias_corrected"] = out[pred_col] * out["monthly_correction_factor"]
    return out, ratio_table


def train_residual_correction(
    df: pd.DataFrame,
    train_mask: Optional[pd.Series] = None,
    pred_col: str = "rom_original",
    reference_col: str = "reference",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train an ML residual model: corrected = ROM + predicted(reference-ROM).

    Uses scikit-learn HistGradientBoostingRegressor. If scikit-learn is not
    installed, falls back to monthly bias correction.
    """
    out, feature_cols = prepare_features(df)
    if train_mask is None:
        # default: all years except the last year are training, last year is test
        last_year = out["date"].dt.year.max()
        train_mask = out["date"].dt.year < last_year
    train_mask = pd.Series(train_mask, index=out.index).fillna(False)
    if train_mask.sum() < 30:
        raise ValueError("Too few training records for residual correction.")
    out["residual"] = out[reference_col] - out[pred_col]
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        )
        model.fit(out.loc[train_mask, feature_cols], out.loc[train_mask, "residual"])
        out["predicted_residual"] = model.predict(out[feature_cols])
        out["rom_residual_corrected"] = out[pred_col] + out["predicted_residual"]
        meta = {"method": "HistGradientBoostingRegressor residual correction", "features": feature_cols, "model": model}
    except Exception as exc:
        out, ratio_table = apply_monthly_bias_correction(out, train_mask=train_mask, pred_col=pred_col, reference_col=reference_col)
        out["predicted_residual"] = out["rom_monthly_bias_corrected"] - out[pred_col]
        out["rom_residual_corrected"] = out["rom_monthly_bias_corrected"]
        meta = {"method": "monthly bias correction fallback", "features": [], "ratio_table": ratio_table, "warning": str(exc)}
    return out, meta


def run_validation_workflow(
    input_path,
    output_dir: str | Path = "outputs",
    holdout_last_year: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run full validation workflow and save CSV outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_comparison_file(input_path)
    df["error_original"] = df["rom_original"] - df["reference"]
    df["pct_error_original"] = df["error_original"] / df["reference"] * 100
    df["abs_pct_error_original"] = df["pct_error_original"].abs()
    train_mask = None
    if holdout_last_year:
        train_mask = df["date"].dt.year < df["date"].dt.year.max()
    corrected, meta = train_residual_correction(df, train_mask=train_mask)
    corrected["error_corrected"] = corrected["rom_residual_corrected"] - corrected["reference"]
    corrected["pct_error_corrected"] = corrected["error_corrected"] / corrected["reference"] * 100
    corrected["abs_pct_error_corrected"] = corrected["pct_error_corrected"].abs()

    original_metrics = pd.DataFrame([compute_metrics(df["reference"], df["rom_original"]).as_dict()])
    corrected_metrics = pd.DataFrame([compute_metrics(corrected["reference"], corrected["rom_residual_corrected"]).as_dict()])
    original_metrics.insert(0, "case", "original")
    corrected_metrics.insert(0, "case", "corrected")
    metrics = pd.concat([original_metrics, corrected_metrics], ignore_index=True)

    monthly_original = monthly_error_summary(df, pred_col="rom_original")
    monthly_corrected = monthly_error_summary(corrected, pred_col="rom_residual_corrected")
    yearly_original = yearly_error_summary(df, pred_col="rom_original")
    yearly_corrected = yearly_error_summary(corrected, pred_col="rom_residual_corrected")

    corrected.to_csv(output_dir / "daily_original_and_corrected.csv", index=False)
    metrics.to_csv(output_dir / "validation_metrics_original_vs_corrected.csv", index=False)
    monthly_original.to_csv(output_dir / "monthly_error_original.csv", index=False)
    monthly_corrected.to_csv(output_dir / "monthly_error_corrected.csv", index=False)
    yearly_original.to_csv(output_dir / "yearly_error_original.csv", index=False)
    yearly_corrected.to_csv(output_dir / "yearly_error_corrected.csv", index=False)

    return {
        "daily": corrected,
        "metrics": metrics,
        "monthly_original": monthly_original,
        "monthly_corrected": monthly_corrected,
        "yearly_original": yearly_original,
        "yearly_corrected": yearly_corrected,
        "meta": pd.DataFrame([{"method": meta.get("method"), "features": ", ".join(meta.get("features", []))}]),
    }
