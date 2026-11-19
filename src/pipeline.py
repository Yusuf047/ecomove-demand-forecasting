"""Reproducible, time-ordered evaluation of hourly rental demand."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Rental_Bikes_Total"
DATA_PATH = ROOT / "data" / "ecomove.csv"
NUMERIC = ["Hour", "Month_sin", "Month_cos", "Weekday", "Temperature_Celsius",
           "Humidity_Percent", "Wind_Speed_kph"]
CATEGORICAL = ["Weather_Outcast", "Public_Holiday", "Business_Day"]
FEATURES = NUMERIC + CATEGORICAL
WEATHER = ["Clear", "Mist and Cloudy", "Light Rain or Snow", "Heavy Rain or Snow"]
RANGES = {"Temperature_Celsius": (-40, 55), "Humidity_Percent": (0, 100),
          "Wind_Speed_kph": (0, 150)}


def load_data(path=DATA_PATH):
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_data(raw):
    """Apply fixed validity rules only; learned imputation stays in the pipeline."""
    required = {"Date", "Hour", TARGET, *CATEGORICAL, *RANGES}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    valid = (df["Date"].notna() & df["Hour"].between(0, 23)
             & df["Hour"].mod(1).eq(0) & np.isfinite(df[TARGET]) & df[TARGET].ge(0))
    audit = {"input_rows": len(df), "invalid_date_hour_or_target": int((~valid).sum())}
    df = df.loc[valid].copy()
    df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    audit["duplicate_hours"] = int(df.duplicated("Timestamp").sum())
    df = df.drop_duplicates("Timestamp", keep="first").sort_values("Timestamp")
    for col, (low, high) in RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = ~values.between(low, high)
        audit[f"{col}_missing_or_invalid"] = int(invalid.sum())
        df[col] = values.mask(invalid)
    df["Weather_Outcast"] = df["Weather_Outcast"].astype("string").str.strip()
    df["Weather_Outcast"] = df["Weather_Outcast"].where(df["Weather_Outcast"].isin(WEATHER), np.nan)
    for col in ["Public_Holiday", "Business_Day"]:
        df[col] = df[col].astype("string").str.strip().str.lower().map({"yes": "Yes", "no": "No"})
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.day_of_week
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    audit["usable_rows"] = len(df)
    return df.reset_index(drop=True), audit


def split_data(df):
    """Split by complete dates: earliest 60%, next 20%, latest 20%."""
    days = np.sort(df["Date"].unique())
    if len(days) < 15:
        raise ValueError("At least 15 distinct dates are required for temporal validation.")
    a, b = days[int(len(days) * .6)], days[int(len(days) * .8)]
    return (df[df.Date < a].copy(), df[(df.Date >= a) & (df.Date < b)].copy(),
            df[df.Date >= b].copy())


def make_pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", prep), ("model", model)])


def predict(model, frame):
    """Counts cannot be negative; the same clipping is used in all evaluations."""
    return np.maximum(0, model.predict(frame[FEATURES]))


def regression_metrics(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def run_experiment(raw=None):
    df, audit = prepare_data(load_data() if raw is None else raw)
    train, validation, test = split_data(df)
    estimators = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=10),
        "Random forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=3,
                                                max_features=.9, random_state=42, n_jobs=2),
    }
    models, validation_rows = {}, []
    for name, estimator in estimators.items():
        model = make_pipeline(estimator)
        model.fit(train[FEATURES], train[TARGET])
        models[name] = model
        validation_rows.append({"Model": name, **regression_metrics(validation[TARGET], predict(model, validation))})
    validation_metrics = pd.DataFrame(validation_rows).sort_values("MAE").reset_index(drop=True)
    selected = validation_metrics.iloc[0]["Model"]
    # Test is evaluated only after the selection rule has been applied to validation.
    test_rows = [{"Model": name, **regression_metrics(test[TARGET], predict(model, test))}
                 for name, model in models.items()]
    predictions = test[["Timestamp", TARGET]].rename(columns={TARGET: "Actual"}).copy()
    predictions["Predicted"] = predict(models[selected], test)
    predictions["Residual"] = predictions.Actual - predictions.Predicted
    return {"data": df, "audit": audit, "train": train, "validation": validation, "test": test,
            "models": models, "selected": selected, "validation_metrics": validation_metrics,
            "test_metrics": pd.DataFrame(test_rows), "predictions": predictions}


def scenario_frame(date, hour, temperature, humidity, wind, weather, holiday, business_day):
    row = {"Date": pd.Timestamp(date).strftime("%d/%m/%Y"), "Hour": hour,
           "Temperature_Celsius": temperature, "Humidity_Percent": humidity,
           "Wind_Speed_kph": wind, "Weather_Outcast": weather, "Public_Holiday": holiday,
           "Business_Day": business_day, TARGET: 0}
    frame, _ = prepare_data(pd.DataFrame([row]))
    if frame.empty:
        raise ValueError("The scenario date or hour is invalid.")
    return frame
