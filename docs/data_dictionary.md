# Data dictionary

| Field | Meaning | Use |
| --- | --- | --- |
| `Date`, `Hour` | Day/month/year and 0–23 hour | Temporal split and calendar features |
| `Temperature_Celsius` | Air temperature | Predictor; fixed validity screen |
| `Humidity_Percent` | Relative humidity, 0–100 | Predictor; fixed validity screen |
| `Wind_Speed_kph` | Wind speed | Predictor; fixed validity screen |
| `Weather_Outcast` | Four weather categories | Categorical predictor |
| `Public_Holiday`, `Business_Day` | Yes/no context | Categorical predictors |
| `Rental_Bikes_Total` | Observed hourly count | Regression target |
| `Registered_Member_Count`, `Casual_User_Count` | Rental-count components | Excluded from predictors |
| `Year`, `Month`, `Week_Day`, `Season`, `Feeling_Temperature_Celsius` | Additional source descriptors | Retained as source context; not model inputs |

The source schema is validated in `src/pipeline.py`. See methodology for cleaning rules.
