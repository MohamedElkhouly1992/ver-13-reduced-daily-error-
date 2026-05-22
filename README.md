# Improved ROM Deployment Bundle

Project: **Reduced-Order-Model-ROM--Degradation**

This bundle adds a validation and error-reduction layer to the HVAC reduced-order model. It is designed to reduce daily mismatch against a DesignBuilder/EnergyPlus-style reference while preserving the ROM as the physical baseline model.

## Main improvements included

1. Monthly error calculation from daily data
2. Daily MAPE, RMSE, CVRMSE, NMBE and aggregate percentage error
3. Calendar features: month, day of week, weekend, day of year, cyclic seasonality
4. Optional weather features: CDD, HDD, lagged temperature, rolling temperature, solar radiation, and temperature-dependent COP modifier
5. Optional occupancy feature: occupancy factor
6. Thermal-lag features from the ROM output: lag 1, lag 2, lag 7, rolling 3-day and rolling 7-day values
7. Monthly bias correction
8. ML residual correction: corrected energy = ROM energy + predicted residual
9. Streamlit app for deployment
10. CLI script for batch validation

## Files

- `streamlit_app.py` — deployable Streamlit interface
- `hvac_v3_engine.py` — validation, feature engineering, monthly error and correction engine
- `run_validation_pipeline.py` — command-line runner
- `requirements.txt` — Python dependencies
- `example_designbuilder_daily_with_errors.csv` — previous data with daily errors
- `monthly_error_summary.csv` — monthly error from the previous data
- `yearly_error_summary.csv` — annual summary from the previous data
- `overall_validation_metrics.csv` — overall metrics from the previous data

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Run from command line

```bash
python run_validation_pipeline.py --input example_designbuilder_daily_with_errors.csv --output outputs
```

## Recommended validation use

For a manuscript, do not train and test the residual correction on the same year without stating it clearly. Use the default holdout approach:

- train/calibrate on earlier years
- test on the final year

This gives a more defensible estimate of daily-error reduction.

## Input format

Required columns:

| Column meaning | Accepted names |
|---|---|
| Date | `date`, `Date/Time`, `datetime` |
| Reference | `Design Builder`, `DesignBuilder`, `reference`, `measured` |
| ROM output | `Model`, `ROM`, `prediction`, `rom_original` |

Optional columns that improve daily accuracy:

- `outdoor_temp`
- `solar_radiation`
- `occupancy_factor`

## Manuscript wording

The improved framework uses the physical ROM as the baseline predictor and applies a residual-correction layer to account for day-to-day variability not captured by the simplified formulation. Calendar, occupancy, weather, thermal-lag and monthly bias features are used to reduce daily MAPE and CVRMSE while retaining the ROM as the interpretable physical core.
