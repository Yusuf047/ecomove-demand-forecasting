![EcoMove — Hourly Rental Demand](docs/banner.svg)

# EcoMove — Hourly Rental Demand

**Python · pandas · scikit-learn · Plotly · Streamlit**

An end-to-end demand-planning case study: investigate hourly rental patterns, compare predictive models, and explore historical scenarios through an interactive Streamlit frontend.

**Business question:** how can time, weather, and calendar context help anticipate rental demand for fleet placement, charging, and staffing?

## What you can explore

- Interactive demand overview with month, day-type, and weather filters.
- Hourly scenario planner with a 24-hour comparison and CSV export.
- Mean, Ridge, and Random Forest models evaluated on time-ordered partitions.
- Fixed validity checks, training-only imputation, and explicit exclusion of target components.
- Model comparison, holdout predictions, residual charts, and a visible data-quality audit.

## Run locally

```bash
git clone https://github.com/Yusuf047/ecomove-demand-forecasting.git
cd ecomove-demand-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use Python **3.14**. On Windows,
activate with `.venv\Scripts\activate`. The first app run trains and caches the models;
subsequent interactions reuse them. No API keys are required.

## Measured results

**Selected on validation:** Random forest. Lower MAE/RMSE is better; higher R² is better.

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Mean baseline | 154.475 | 189.329 | -0.014 |
| Ridge regression | 118.690 | 181.318 | 0.070 |
| Random forest | 63.674 | 91.138 | 0.765 |

Complete dates are split chronologically: earliest 60% train, next 20% validation, latest 20% test. Validation MAE selects the model. Test dates never enter fitting or selection. All displayed metrics are regenerated from the checked-in data and code.

Dataset: **8,734** usable observations.
Exact split counts, package versions, and the dataset SHA-256 are in
[`reports/summary.json`](reports/summary.json).
Results differ from the initial exploration because preprocessing and evaluation have been strengthened.

## Business interpretation

The dashboard makes peak-hour and weather patterns inspectable and provides a reproducible baseline for demand-planning discussions. It does **not** demonstrate reduced costs or stockouts.

The observations are from 2012 and cover one year. The scenario planner is not a live forecast, and the app has no live weather feed. The target column is `Rental_Bikes_Total`; the fictional mobility case is presented as rentals without claiming verified scooter operations. Cross-city transfer and later-year performance are untested.

## Project structure

| Location | Purpose |
| --- | --- |
| `app.py` | Streamlit frontend |
| `src/pipeline.py` | Validation, transformations, models, and metrics |
| `notebooks/analysis.ipynb` | Reproducible walkthrough using the same pipeline |
| `data/` | Case-study CSV and data notes |
| `reports/` | Generated metrics, predictions, and experiment metadata |
| `tests/` | Pipeline checks and Streamlit interaction tests |
| `docs/` | Methodology, data dictionary, and deployment instructions |
| `docs/ci-template.yml` | Optional GitHub Actions configuration |

## Reproduce and test

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python evaluate.py
```

See the [analysis notebook](notebooks/analysis.ipynb), [methodology](docs/methodology.md),
[data dictionary](docs/data_dictionary.md), and [deployment guide](docs/deployment.md).
The app is configured for local use and Streamlit deployment; hosting is a separate step.
Automated checks have been run locally. CI is provided as an optional template and is not enabled on this repository.

## Data provenance

The supplied adapted case-study data is included for reproducibility. Identifiers are
excluded from all model inputs; manager codes, record IDs, and session colours are omitted from the distributed CSV. It is not a live feed. No upstream open-data licence was supplied;
this repository makes no new ownership or licensing claim over the underlying dataset.
See [data notes](data/README.md).

## Author

Mohammad Yusuf Mohiuddin — Business Analytics
