import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.pipeline import FEATURES, TARGET, load_data, make_pipeline, prepare_data, scenario_frame, split_data


def test_dates_are_day_first_and_weather_errors_become_missing():
    raw = load_data().head(3).copy()
    raw.loc[0, "Date"] = "2/3/2012"
    raw.loc[0, "Humidity_Percent"] = 150
    raw.loc[0, "Temperature_Celsius"] = -99
    raw.loc[0, "Wind_Speed_kph"] = -20
    clean, audit = prepare_data(raw)
    row = clean.loc[clean.Date.eq(pd.Timestamp("2012-03-02"))].iloc[0]
    assert np.isnan(row.Humidity_Percent) and np.isnan(row.Temperature_Celsius)
    assert np.isnan(row.Wind_Speed_kph)
    assert audit["Humidity_Percent_missing_or_invalid"] == 1


def test_chronological_partitions_do_not_share_days():
    data, _ = prepare_data(load_data())
    train, validation, test = split_data(data)
    assert train.Date.max() < validation.Date.min() < test.Date.min()
    assert validation.Date.max() < test.Date.min()
    assert len(train) + len(validation) + len(test) == len(data)
    assert not {TARGET, "Registered_Member_Count", "Casual_User_Count", "RecordID"} & set(FEATURES)


def test_preprocessing_does_not_learn_from_test_values():
    data, _ = prepare_data(load_data())
    train, _, test = split_data(data)
    model = make_pipeline(DummyRegressor()).fit(train[FEATURES], train[TARGET])
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    statistics = imputer.statistics_.copy()
    changed = test.copy()
    changed["Temperature_Celsius"] = 999999
    changed["Weather_Outcast"] = "Unseen category"
    assert np.isfinite(model.predict(changed[FEATURES])).all()
    np.testing.assert_array_equal(imputer.statistics_, statistics)


def test_inference_features_match_training_schema():
    frame = scenario_frame("2012-07-16", 17, 24, 55, 12, "Clear", "No", "Yes")
    assert frame[FEATURES].notna().all().all()
    assert frame.Weekday.iloc[0] == 0
