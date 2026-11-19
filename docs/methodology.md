### Business question

How can hourly demand patterns support fleet placement, charging, and staffing discussions? EcoMove is a fictional mobility business. The supplied sample contains hourly rental observations from 2012. The source target is named `Rental_Bikes_Total`; this project retains that name and uses the neutral term **rentals** rather than claiming verified scooter operations.

### Data preparation

Dates use an explicit day/month/year format. Month and weekday features are derived from those dates. Impossible weather readings become missing values: temperature outside −40 to 55 °C, humidity outside 0–100%, and wind outside 0–150 km/h. These are fixed screening rules, not thresholds learned from the test set. Duplicate date/hour records keep the first observation. Invalid dates, hours, or targets are excluded. An audit records every rule's count.

Numeric medians and categorical encodings are fitted on the training partition only. Missing values are not filled using future records. Holiday/business-day labels are normalised. Identifiers, manager codes, session colours, and registered/casual rental counts are excluded from predictors. The latter counts form the target and would leak the answer.

### Validation design

Complete dates are ordered and split into the earliest 60% for training, the next 20% for validation, and the latest 20% for testing. The exact row counts and date ranges appear in the model evidence page. Three fixed candidates are compared: mean baseline, Ridge regression, and Random Forest. Validation mean absolute error selects the model. Hyperparameters are fixed in source, not tuned on the holdout.

The fitted models remain trained on the training partition; neither validation nor test data is used for refitting. Test metrics are reported after model selection. Negative count predictions are clipped to zero consistently during validation, test evaluation, and inference. Metrics include MAE, RMSE, and R²; a negative R² is possible and means worse performance than the test-set mean benchmark.

### Interpretation and limits

The frontend offers a historical scenario comparison, not a current demand forecast. Weather inputs are observed historical measurements, while a live deployment would need weather forecasts whose errors may worsen performance. Only one year is available, so later-season performance may suffer when earlier months are used for training. A single chronological holdout is not a substitute for multi-year backtesting.

Differences from the initial exploratory analysis include explicit day-first parsing, validity checks, training-only imputation, removal of non-operational fields, chronological evaluation, and an additional nonlinear model. Reported results are regenerated for this implementation; old report scores are not reused.

Potential benefits such as fewer stockouts or lower operating cost are hypotheses. There is no deployment experiment or measured business impact in this dataset. Before operational use, collect more years and local observations, compare rolling-origin backtests, validate forecast-weather inputs, and measure the actual service outcomes.
