"""Interactive EcoMove demand-planning dashboard."""

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import ROOT, TARGET, WEATHER, predict, run_experiment, scenario_frame

st.set_page_config(page_title="EcoMove | Demand intelligence", page_icon="🚲", layout="wide")
st.markdown("""<style>
.block-container {padding-top:4rem; max-width:1350px;}
h1 {letter-spacing:-.045em;}
[data-testid="stMetric"] {background:#f1f7f5; border:1px solid #dce9e4; border-radius:12px; padding:18px;}
</style>""", unsafe_allow_html=True)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#087F6D", "#437EA0", "#EDA95A", "#64748B"]


@st.cache_resource(show_spinner="Preparing the demand models…")
def experiment():
    return run_experiment()


try:
    result = experiment()
except (ValueError, FileNotFoundError) as error:
    st.error(f"The dataset could not be loaded: {error}")
    st.stop()

df = result["data"]
with st.sidebar:
    st.markdown("## EcoMove")
    st.caption("DEMAND INTELLIGENCE")
    page = st.radio("Workspace", ["Demand overview", "Scenario planner", "Model evidence", "Data & methodology"])
    st.divider()
    st.caption("Historical mobility case study · 2012")
    st.caption("Python · scikit-learn · Streamlit")

st.caption("ECOMOVE / OPERATIONS ANALYTICS")
st.title({"Demand overview": "Understand the rhythm of demand.",
          "Scenario planner": "Plan an hourly rental scenario.",
          "Model evidence": "See what the model can support.",
          "Data & methodology": "From source data to a tested model."}[page])

if page == "Demand overview":
    st.write("Explore rental patterns across time and weather to inform fleet and staffing discussions.")
    a, b, c = st.columns([1, 1, 2])
    months = a.multiselect("Month", list(range(1, 13)), default=list(range(1, 13)),
                           format_func=lambda m: date(2012, m, 1).strftime("%b"))
    day_type = b.selectbox("Business day", ["All", "Yes", "No"])
    weather = c.multiselect("Weather", WEATHER, default=WEATHER)
    filtered = df[df.Month.isin(months) & df.Weather_Outcast.isin(weather)].copy()
    if day_type != "All":
        filtered = filtered[filtered.Business_Day.eq(day_type)]
    if filtered.empty:
        st.info("No records match these filters. Select a month and weather condition to continue.")
        st.stop()
    hourly = filtered.groupby("Hour", as_index=False)[TARGET].mean()
    peak = int(hourly.loc[hourly[TARGET].idxmax(), "Hour"])
    metrics = st.columns(4)
    metrics[0].metric("Hours observed", f"{len(filtered):,}")
    metrics[1].metric("Total rentals", f"{filtered[TARGET].sum():,.0f}")
    metrics[2].metric("Average per hour", f"{filtered[TARGET].mean():.0f}")
    metrics[3].metric("Peak average hour", f"{peak:02d}:00")
    st.subheader("Hourly demand profile")
    profile = filtered.groupby(["Hour", "Business_Day"], as_index=False)[TARGET].mean()
    fig = px.line(profile, x="Hour", y=TARGET, color="Business_Day", markers=True,
                  labels={TARGET: "Average rentals", "Business_Day": "Business day"})
    fig.update_xaxes(dtick=2)
    st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Seasonal pattern")
        monthly = filtered.groupby("Month", as_index=False)[TARGET].mean()
        monthly["Month"] = monthly.Month.map(lambda m: date(2012, int(m), 1).strftime("%b"))
        st.plotly_chart(px.bar(monthly, x="Month", y=TARGET, labels={TARGET: "Average rentals"}), width="stretch")
    with right:
        st.subheader("Weather context")
        grouped = filtered.groupby("Weather_Outcast", as_index=False).agg(Average=(TARGET, "mean"), Hours=(TARGET, "size"))
        st.plotly_chart(px.bar(grouped, x="Average", y="Weather_Outcast", orientation="h",
                              hover_data=["Hours"], labels={"Weather_Outcast": "", "Average": "Average rentals"}), width="stretch")
    st.caption("Charts describe associations in the supplied historical sample. They do not establish weather effects or operational savings.")
    st.download_button("Download filtered observations", filtered.drop(columns=["Timestamp"]).to_csv(index=False),
                       "ecomove_filtered.csv", "text/csv")

elif page == "Scenario planner":
    st.write("Use the selected model to compare conditions within this historical case study.")
    st.info("This is a historical scenario tool, not a live forecast. Inputs are supplied assumptions; there is no live weather feed.")
    with st.form("scenario"):
        left, middle, right = st.columns(3)
        scenario_date = left.date_input("Scenario date", date(2012, 7, 16), min_value=date(2012, 1, 1), max_value=date(2012, 12, 31))
        hour = left.slider("Hour", 0, 23, 17)
        temperature = middle.slider("Temperature (°C)", -10, 45, 24)
        humidity = middle.slider("Humidity (%)", 0, 100, 55)
        wind = right.slider("Wind (km/h)", 0, 100, 12)
        weather = right.selectbox("Weather condition", WEATHER)
        holiday = left.selectbox("Public holiday", ["No", "Yes"])
        business = middle.selectbox("Business day", ["Yes", "No"])
        submitted = st.form_submit_button("Estimate demand", type="primary")
    if submitted:
        frame = scenario_frame(scenario_date, hour, temperature, humidity, wind, weather, holiday, business)
        model = result["models"][result["selected"]]
        estimate = float(predict(model, frame)[0])
        stats = result["test_metrics"].set_index("Model").loc[result["selected"]]
        a, b = st.columns(2)
        a.metric("Estimated rentals in this hour", f"{estimate:,.0f}")
        b.metric("Holdout mean absolute error", f"{stats.MAE:.1f} rentals")
        st.caption("Average holdout error is context, not a prediction interval or a guarantee for this scenario.")
        daily = pd.concat([scenario_frame(scenario_date, h, temperature, humidity, wind, weather, holiday, business)
                           for h in range(24)], ignore_index=True)
        daily["Estimated rentals"] = predict(model, daily)
        st.plotly_chart(px.line(daily, x="Hour", y="Estimated rentals", markers=True), width="stretch")
        st.caption(f"Model: {result['selected']}. The daily curve holds weather and day flags constant. Changing an input is not a causal estimate.")
        st.download_button("Download scenario", daily[["Hour", "Estimated rentals"]].to_csv(index=False), "demand_scenario.csv", "text/csv")

elif page == "Model evidence":
    st.write("Models are selected by validation MAE. The latest dates are reserved for a final holdout check.")
    st.success(f"Selected on validation: {result['selected']}")
    rows = []
    for name in ["train", "validation", "test"]:
        part = result[name]
        rows.append({"Split": name.title(), "Rows": len(part), "Start": str(part.Date.min().date()), "End": str(part.Date.max().date())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        st.subheader("Validation comparison")
        st.dataframe(result["validation_metrics"].round(3), hide_index=True, width="stretch")
    with b:
        st.subheader("Holdout comparison")
        st.dataframe(result["test_metrics"].round(3), hide_index=True, width="stretch")
    predictions = result["predictions"]
    a, b = st.columns(2)
    with a:
        st.subheader("Actual vs predicted")
        fig = px.scatter(predictions, x="Actual", y="Predicted", opacity=.35)
        high = float(max(predictions.Actual.max(), predictions.Predicted.max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=high, y1=high, line=dict(color="#E09F3E", dash="dash"))
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Residual distribution")
        st.plotly_chart(px.histogram(predictions, x="Residual", nbins=45), width="stretch")
    st.caption("Residual = actual − predicted. Positive values indicate underprediction. No model is refitted on validation or test records.")
    st.download_button("Download holdout predictions", predictions.to_csv(index=False), "holdout_predictions.csv", "text/csv")

else:
    st.markdown((ROOT / "docs" / "methodology.md").read_text())
    st.subheader("Data-quality audit")
    st.dataframe(pd.DataFrame(result["audit"].items(), columns=["Check", "Count"]), hide_index=True, width="stretch")

st.divider()
st.caption("EcoMove · Reproducible business analytics portfolio · Historical case study")
