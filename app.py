
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# -------------------------------
# 1. PAGE SETUP
# -------------------------------

st.set_page_config(
    page_title="DonorPulse",
    page_icon="📊",
    layout="wide"
)




st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0B1020;
    color: #E5E7EB;
}

/* MAIN CONTENT */
[data-testid="stAppViewContainer"] {
    background-color: #0B1020;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* HEADINGS */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

/* NORMAL TEXT */
p, label, .stMarkdown {
    color: #CBD5E1;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

/* KPI CARDS */
div[data-testid="stMetric"] {
    background-color: #171F35;
    border: 1px solid #293653;
    padding: 18px 14px;
    min-height: 125px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* KPI LABELS */
div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 14px;
}

/* KPI VALUES */
div[data-testid="stMetricValue"] {
    color: #A78BFA !important;
    font-size: clamp(20px, 2vw, 32px);
    white-space: nowrap;
}

/* BUTTONS */
.stButton button {
    background-color: #6D5DFB;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #5145CD;
    color: white;
}

/* INPUT FIELDS */
.stSelectbox div[data-baseweb="select"] {
    background-color: #171F35;
}

/* DATA TABLES */
[data-testid="stDataFrame"] {
    background-color: #171F35;
    border: 1px solid #293653;
    border-radius: 12px;
}

/* SECTION SPACING */
[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

</style>
""", unsafe_allow_html=True)





BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "donorpulse_synthetic_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "conversion_model.pkl"
RECOMMENDATIONS_PATH = BASE_DIR / "data" / "final_donor_recommendations.csv"


# -------------------------------
# 2. LOAD DATA
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_recommendations():
    return pd.read_csv(RECOMMENDATIONS_PATH)

df = load_data()
recommendations = load_recommendations()
model = joblib.load(MODEL_PATH)


# -------------------------------
# 3. SIDEBAR
# -------------------------------

st.sidebar.title("DonorPulse")
page = st.sidebar.radio(
    "Navigation",
    ["Campaign Overview", "Campaign Health",
        "Anomaly Detection",
        "Conversion Prediction","Explainable AI","Follow-up Recommendations"
    ]
)


# -------------------------------
# 4. CAMPAIGN OVERVIEW
# -------------------------------

if page == "Campaign Overview":

    st.title("📊 Campaign Overview")

    st.write(
        "Monitor fundraising performance and understand "
        "how your campaigns are performing."
    )

    # Calculate important metrics

    total_interactions = len(df)
    total_donors = df["donor_id"].nunique()
    total_campaigns = df["campaign_id"].nunique()

    conversion_rate = df["converted"].mean() * 100

    total_donations = df["donation_amount"].sum()

    # Display KPI cards

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Interactions", f"{total_interactions:,}")
    col2.metric("Unique Donors", f"{total_donors:,}")
    col3.metric("Campaigns", total_campaigns)
    col4.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    col5.metric(
    "Total Donations",
    f"₹{total_donations / 100000:.2f} L",
    help=f"Full amount: ₹{total_donations:,.0f}"
)

    st.divider()

    # -------------------------------
    # CAMPAIGN PERFORMANCE
    # -------------------------------

    st.subheader("Campaign Performance")

    campaign_data = df.groupby("campaign_name").agg(
        Total_Interactions=("interaction_id", "count"),
        Conversion_Rate=("converted", "mean"),
        Total_Donations=("donation_amount", "sum")
    ).reset_index()

    campaign_data["Conversion_Rate"] *= 100

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.bar(
            campaign_data,
            x="campaign_name",
            y="Conversion_Rate",
            title="Conversion Rate by Campaign",
            labels={
                "campaign_name": "Campaign",
                "Conversion_Rate": "Conversion Rate (%)"
            },
            template="plotly_white",
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.bar(
            campaign_data,
            x="campaign_name",
            y="Total_Donations",
            title="Total Donations by Campaign",
            labels={
                "campaign_name": "Campaign",
                "Total_Donations": "Donations (₹)"
            },
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # -------------------------------
    # MONTHLY PERFORMANCE
    # -------------------------------

    st.subheader("Monthly Conversion Trend")

    df["interaction_date"] = pd.to_datetime(df["interaction_date"])

    monthly_data = df.copy()

    monthly_data["Month"] = (
        monthly_data["interaction_date"].dt.to_period("M").astype(str)
    )

    monthly_conversion = monthly_data.groupby("Month").agg(
        Conversion_Rate=("converted", "mean")
    ).reset_index()

    monthly_conversion["Conversion_Rate"] *= 100

    fig3 = px.line(
        monthly_conversion,
        x="Month",
        y="Conversion_Rate",
        markers=True,
        title="Monthly Conversion Rate",
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # -------------------------------
    # CAMPAIGN DATA TABLE
    # -------------------------------

    st.subheader("Campaign Summary")

    st.dataframe(
        campaign_data.sort_values(
            by="Conversion_Rate",
            ascending=False
        ),
        use_container_width=True
    )

    st.caption(
        "Note: DonorPulse currently uses synthetic data for demonstration."
    )
    
# -------------------------------
# 5. CAMPAIGN HEALTH
# -------------------------------

elif page == "Campaign Health":

    st.title("🎯 Campaign Health")
    st.write(
        "Analyse fundraising campaign performance and identify "
        "campaigns that may require attention."
    )

    # Calculate campaign metrics
    campaign_health = df.groupby(
        ["campaign_id", "campaign_name"]
    ).agg(
        total_interactions=("interaction_id", "count"),
        conversion_rate=("converted", "mean"),
        total_donations=("donation_amount", "sum"),
        duplicate_rate=("duplicate_contact", "mean")
    ).reset_index()

    campaign_health["conversion_rate"] *= 100
    campaign_health["duplicate_rate"] *= 100

    # Campaign selection
    selected_campaign = st.selectbox(
        "Select a campaign",
        campaign_health["campaign_name"].unique()
    )

    selected_data = campaign_health[
        campaign_health["campaign_name"] == selected_campaign
    ].iloc[0]

    st.divider()

    # Display selected campaign metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Interactions",
        f"{selected_data['total_interactions']:,}"
    )

    col2.metric(
        "Conversion Rate",
        f"{selected_data['conversion_rate']:.2f}%"
    )

    col3.metric(
        "Total Donations",
        f"₹{selected_data['total_donations']:,.0f}"
    )

    st.divider()

    # Conversion comparison
    st.subheader("Campaign Conversion Comparison")

    fig1 = px.bar(
        campaign_health,
        x="campaign_name",
        y="conversion_rate",
        title="Conversion Rate Across Campaigns",
        labels={
            "campaign_name": "Campaign",
            "conversion_rate": "Conversion Rate (%)"
        },
        template="plotly_white"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Donation comparison
    st.subheader("Donation Performance")

    fig2 = px.bar(
        campaign_health,
        x="campaign_name",
        y="total_donations",
        title="Total Donations Across Campaigns",
        labels={
            "campaign_name": "Campaign",
            "total_donations": "Total Donations (₹)"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Duplicate contact analysis
    st.subheader("Duplicate Contact Rate")

    fig3 = px.bar(
        campaign_health,
        x="campaign_name",
        y="duplicate_rate",
        title="Duplicate Contact Rate by Campaign",
        labels={
            "campaign_name": "Campaign",
            "duplicate_rate": "Duplicate Rate (%)"
        },
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Full campaign table
    st.subheader("Complete Campaign Health Report")

    st.dataframe(
        campaign_health.sort_values(
            by="conversion_rate",
            ascending=True
        ),
        use_container_width=True
    )

    st.caption(
        "These metrics are calculated from synthetic interaction data. "
        "A low conversion rate is a signal for investigation, not proof "
        "that a campaign is failing."
    )
    
# -------------------------------
# 6. ANOMALY DETECTION
# -------------------------------

elif page == "Anomaly Detection":

    st.title("🚨 Anomaly Detection")

    st.write(
        "Identify unusual patterns in agent performance, "
        "duplicate contacts and missed follow-ups."
    )

    # Prepare dates
    anomaly_df = df.copy()

    anomaly_df["interaction_date"] = pd.to_datetime(
        anomaly_df["interaction_date"]
    )

    anomaly_df["Month"] = (
        anomaly_df["interaction_date"].dt.to_period("M").astype(str)
    )

    anomaly_df["connected"] = (
        anomaly_df["contact_status"] == "Connected"
    ).astype(int)

    # Agent monthly performance
    agent_monthly = anomaly_df.groupby(
        ["agent_id", "Month"]
    ).agg(
        total_calls=("interaction_id", "count"),
        conversion_rate=("converted", "mean"),
        duplicate_rate=("duplicate_contact", "mean"),
        invalid_contacts=(
            "contact_status",
            lambda x: (x == "Invalid Contact").mean()
        ),
        required_followups=("followup_required", "sum"),
        completed_followups=(
            "followup_completed",
            "sum"
        )
    ).reset_index()

    agent_monthly["conversion_rate"] *= 100
    agent_monthly["duplicate_rate"] *= 100
    agent_monthly["invalid_contacts"] *= 100

    # Correct follow-up completion calculation
    followup_data = anomaly_df[
        anomaly_df["followup_required"] == 1
    ].groupby(
        ["agent_id", "Month"]
    ).agg(
        required_followups=("interaction_id", "count"),
        completed_followups=("followup_completed", "sum")
    ).reset_index()

    followup_data["followup_completion"] = (
        followup_data["completed_followups"]
        / followup_data["required_followups"]
    ) * 100

    # Merge follow-up metrics
    agent_monthly = agent_monthly.drop(
        columns=["required_followups", "completed_followups"]
    ).merge(
        followup_data[
            ["agent_id", "Month", "followup_completion"]
        ],
        on=["agent_id", "Month"],
        how="left"
    )

    # -------------------------------
    # KPI CARDS
    # -------------------------------

    total_duplicates = int(
        anomaly_df["duplicate_contact"].sum()
    )

    invalid_count = int(
        (anomaly_df["contact_status"] == "Invalid Contact").sum()
    )

    pending_followups = int(
        (
            (anomaly_df["followup_required"] == 1)
            & (anomaly_df["followup_completed"] == 0)
        ).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Duplicate Contacts", f"{total_duplicates:,}")
    col2.metric("Invalid Contacts", f"{invalid_count:,}")
    col3.metric("Pending Follow-ups", f"{pending_followups:,}")

    st.divider()

    # -------------------------------
    # FILTERS
    # -------------------------------

    st.subheader("Filter Agent Performance")

    selected_agent = st.selectbox(
        "Select Agent",
        ["All Agents"] + sorted(
            agent_monthly["agent_id"].unique().tolist()
        )
    )

    if selected_agent == "All Agents":
        filtered_data = agent_monthly.copy()
    else:
        filtered_data = agent_monthly[
            agent_monthly["agent_id"] == selected_agent
        ]

    # -------------------------------
    # CHARTS
    # -------------------------------

    st.subheader("Monthly Agent Conversion Rate")

    fig1 = px.line(
        filtered_data,
        x="Month",
        y="conversion_rate",
        color="agent_id",
        markers=True,
        title="Agent Conversion Rate Over Time",
        template="plotly_white"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Duplicate Contact Rate")

    fig2 = px.line(
        filtered_data,
        x="Month",
        y="duplicate_rate",
        color="agent_id",
        markers=True,
        title="Monthly Duplicate Contact Rate",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Invalid Contact Rate")

    fig3 = px.line(
        filtered_data,
        x="Month",
        y="invalid_contacts",
        color="agent_id",
        markers=True,
        title="Monthly Invalid Contact Rate",
        template="plotly_white"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Follow-up Completion Rate")

    fig4 = px.line(
        filtered_data,
        x="Month",
        y="followup_completion",
        color="agent_id",
        markers=True,
        title="Monthly Follow-up Completion Rate",
        template="plotly_white"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -------------------------------
    # UNUSUAL PATTERN FLAGS
    # -------------------------------

    st.subheader("⚠️ Potential Risk Flags")

    duplicate_threshold = 15
    invalid_threshold = 20
    followup_threshold = 60

    flagged_agents = agent_monthly[
        (agent_monthly["duplicate_rate"] > duplicate_threshold)
        | (agent_monthly["invalid_contacts"] > invalid_threshold)
        | (agent_monthly["followup_completion"] < followup_threshold)
    ].copy()

    if len(flagged_agents) > 0:

        st.warning(
            f"{len(flagged_agents)} agent-month records "
            "cross at least one risk threshold."
        )

        def explain_risk(row):

            reasons = []

            if row["duplicate_rate"] > duplicate_threshold:
                reasons.append("High duplicate-contact rate")

            if row["invalid_contacts"] > invalid_threshold:
                reasons.append("High invalid-contact rate")

            if (
                pd.notna(row["followup_completion"])
                and row["followup_completion"] < followup_threshold
            ):
                reasons.append("Low follow-up completion")

            return ", ".join(reasons)

        flagged_agents["Risk_Reason"] = flagged_agents.apply(
            explain_risk,
            axis=1
        )

        st.dataframe(
            flagged_agents.sort_values(
                by="duplicate_rate",
                ascending=False
            ),
            use_container_width=True
        )

    else:

        st.success("No agent-month records crossed the selected thresholds.")

    st.caption(
        "Risk flags use illustrative thresholds and are not proof of "
        "misconduct or poor performance. Review the underlying records "
        "before taking action."
    )
    
# -------------------------------
# 7. CONVERSION PREDICTION
# -------------------------------

elif page == "Conversion Prediction":

    st.title("🤖 Donor Conversion Prediction")

    st.write(
        "Use the trained Logistic Regression model to estimate "
        "the likelihood of conversion for an interaction."
    )

    # Prepare model features
    features = [
        "age_group",
        "city_tier",
        "previous_donor",
        "previous_donations",
        "total_previous_amount",
        "campaign_type",
        "attempt_number",
        "call_duration_sec",
        "contact_status",
        "disposition",
        "followup_required",
        "days_since_last_contact",
        "duplicate_contact"
    ]

    # Select an interaction
    interaction_index = st.selectbox(
        "Select an interaction",
        df.index,
        format_func=lambda x: (
            f"Interaction {df.loc[x, 'interaction_id']} "
            f"| Donor {df.loc[x, 'donor_id']}"
        )
    )

    selected_row = df.loc[[interaction_index]]

    # Predict conversion probability
    prediction_probability = model.predict_proba(
        selected_row[features]
    )[0][1] * 100

    # Assign priority
    if prediction_probability >= 60:
        priority = "High"
    elif prediction_probability >= 30:
        priority = "Medium"
    else:
        priority = "Low"

    st.divider()

    # Display prediction
    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Conversion Probability",
        f"{prediction_probability:.2f}%"
    )

    col2.metric(
        "Priority Level",
        priority
    )

    st.progress(
        int(prediction_probability)
    )

    st.divider()

    # Display interaction information
    st.subheader("Interaction Details")

    details = {
        "Donor ID": selected_row["donor_id"].iloc[0],
        "Campaign": selected_row["campaign_name"].iloc[0],
        "Agent": selected_row["agent_id"].iloc[0],
        "Contact Status": selected_row["contact_status"].iloc[0],
        "Disposition": selected_row["disposition"].iloc[0],
        "Previous Donor": selected_row["previous_donor"].iloc[0],
        "Follow-up Required": selected_row["followup_required"].iloc[0],
        "Duplicate Contact": selected_row["duplicate_contact"].iloc[0]
    }

    st.dataframe(
        pd.DataFrame(
            details.items(),
            columns=["Attribute", "Value"]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Show actual outcome for demonstration
    st.subheader("Actual Outcome (Historical Data)")

    actual_outcome = selected_row["converted"].iloc[0]

    if actual_outcome == 1:
        st.success("This interaction resulted in a conversion.")
    else:
        st.info("This interaction did not result in a conversion.")

    st.caption(
        "The actual outcome is shown only for historical evaluation. "
        "It is not used as an input to the prediction. This model uses "
        "post-interaction information, so it should not be interpreted "
        "as a prediction made before the call."
    )

    st.warning(
        "Predictions are based on synthetic training data. "
        "A predicted probability is a model estimate, not a guarantee "
        "of an actual donation."
    )
    
# -------------------------------
# 8. EXPLAINABLE AI
# -------------------------------

elif page == "Explainable AI":

    import shap
    import numpy as np
    import scipy.sparse

    st.title("🔍 Explainable AI")
    st.write(
        "Understand which factors influence the model's "
        "conversion predictions using SHAP."
    )

    features = [
        "age_group",
        "city_tier",
        "previous_donor",
        "previous_donations",
        "total_previous_amount",
        "campaign_type",
        "attempt_number",
        "call_duration_sec",
        "contact_status",
        "disposition",
        "followup_required",
        "days_since_last_contact",
        "duplicate_contact"
    ]

    # Select interaction
    interaction_index = st.selectbox(
        "Select an interaction to explain",
        df.index,
        key="shap_interaction",
        format_func=lambda x: (
            f"Interaction {df.loc[x, 'interaction_id']} "
            f"| Donor {df.loc[x, 'donor_id']}"
        )
    )

    # Prepare model components
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X = df[features]

    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    # Use a smaller sample for faster calculations
    background = X_transformed[:500]

    if scipy.sparse.issparse(background):
        background = background.toarray()

    # Create SHAP explainer
    explainer = shap.LinearExplainer(
        classifier,
        background
    )

    # Transform selected interaction
    selected_X = preprocessor.transform(
        df.loc[[interaction_index], features]
    )

    if scipy.sparse.issparse(selected_X):
        selected_X = selected_X.toarray()

    shap_values = explainer(selected_X)

    # Get SHAP values
    values = shap_values.values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": values
    })

    # Sort by absolute influence
    explanation_df["Importance"] = (
        explanation_df["SHAP Value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    # Display feature contributions
    st.subheader("Top Factors Influencing This Prediction")

    fig = px.bar(
        explanation_df.sort_values("SHAP Value"),
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="SHAP Value",
        color_continuous_midpoint=0,
        title="SHAP Feature Contributions",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "Positive SHAP values push the model's prediction "
        "towards conversion. Negative values push it away "
        "from conversion."
    )

    # Show explanation table
    st.subheader("Detailed Explanation")

    st.dataframe(
        explanation_df[
            ["Feature", "SHAP Value"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "SHAP values explain this Logistic Regression model's "
        "output in log-odds space. They are not percentage-point "
        "changes in conversion probability. Correlated inputs can "
        "share or complicate feature contributions."
    )

    st.warning(
        "These explanations describe model behaviour on synthetic "
        "data. They do not establish that a feature causes a donor "
        "to convert."
    )
    
# -------------------------------
# 9. FOLLOW-UP RECOMMENDATIONS
# -------------------------------

elif page == "Follow-up Recommendations":

    st.title("📞 Follow-up Recommendations")

    st.write(
        "Identify eligible donors, prioritise follow-ups "
        "and provide recommended actions to fundraising teams."
    )

    # Load recommendation data
    rec = recommendations.copy()

    # -------------------------------
    # KPI CARDS
    # -------------------------------

    total_eligible = len(rec)

    high_priority = (
        rec["priority"] == "High"
    ).sum()

    medium_priority = (
        rec["priority"] == "Medium"
    ).sum()

    low_priority = (
        rec["priority"] == "Low"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Eligible Donors", f"{total_eligible:,}")
    col2.metric("High Priority", f"{high_priority:,}")
    col3.metric("Medium Priority", f"{medium_priority:,}")
    col4.metric("Low Priority", f"{low_priority:,}")

    st.divider()

    # -------------------------------
    # FILTERS
    # -------------------------------

    st.subheader("Find Donors")

    col1, col2 = st.columns(2)

    with col1:
        selected_priority = st.selectbox(
            "Filter by Priority",
            ["All", "High", "Medium", "Low"]
        )

    with col2:
        selected_campaign = st.selectbox(
            "Filter by Campaign",
            ["All"] + sorted(
                rec["campaign_name"].dropna().unique().tolist()
            )
        )

    filtered_rec = rec.copy()

    if selected_priority != "All":
        filtered_rec = filtered_rec[
            filtered_rec["priority"] == selected_priority
        ]

    if selected_campaign != "All":
        filtered_rec = filtered_rec[
            filtered_rec["campaign_name"] == selected_campaign
        ]

    # -------------------------------
    # RECOMMENDATION TABLE
    # -------------------------------

    st.subheader("Recommended Donor Actions")

    display_columns = [
        "donor_id",
        "campaign_name",
        "agent_id",
        "conversion_probability",
        "priority",
        "recommended_action",
        "recommendation_reason",
        "recommendation_score"
    ]

    st.dataframe(
        filtered_rec[display_columns].sort_values(
            by="recommendation_score",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"Showing {len(filtered_rec):,} donors."
    )

    st.divider()

    # -------------------------------
    # SELECT A DONOR
    # -------------------------------

    st.subheader("Donor Recommendation Details")

    if len(filtered_rec) > 0:

        selected_donor = st.selectbox(
            "Select a donor",
            filtered_rec["donor_id"].tolist()
        )

        donor_info = filtered_rec[
            filtered_rec["donor_id"] == selected_donor
        ].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Donor ID:**", donor_info["donor_id"])
            st.write("**Campaign:**", donor_info["campaign_name"])
            st.write("**Priority:**", donor_info["priority"])

        with col2:
            st.write(
                "**Conversion Probability:**",
                f"{donor_info['conversion_probability']:.2f}%"
            )

            st.write(
                "**Recommended Action:**",
                donor_info["recommended_action"]
            )

        st.info(
            "Why this donor was selected: "
            + str(donor_info["recommendation_reason"])
        )

    else:

        st.warning("No donors match the selected filters.")

    st.divider()

    # -------------------------------
    # DOWNLOAD RECOMMENDATIONS
    # -------------------------------

    st.subheader("Export Recommendations")

    csv = filtered_rec.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Donor List",
        data=csv,
        file_name="donorpulse_recommendations.csv",
        mime="text/csv"
    )

    st.caption(
        "Recommendation scores are heuristic ranking scores, "
        "not probabilities. Review contact eligibility and donor "
        "history before taking action. This demonstration uses "
        "synthetic data."
    )


