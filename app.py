import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# Page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# Simple styling
st.markdown(
    """
    <style>

    /* Page */
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main text */
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container p,
    .block-container label,
    .block-container span {
        color: #1f2937 !important;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #2454c6, #2867e8);
        padding: 22px 24px;
        border-radius: 11px;
        margin-bottom: 24px;
    }

    .header-box h1 {
        color: white !important;
        font-size: 28px;
        margin: 0 0 6px 0;
    }

    .header-box p {
        color: #e5edff !important;
        margin: 0;
        font-size: 13px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9dee8;
        border-radius: 9px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    /* Inputs */
    .stNumberInput input {
        background-color: white !important;
        color: #111827 !important;
    }

    /* Info box */
    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Button */
    .stButton button {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
}

.stButton button p,
.stButton button span {
    color: white !important;
}

    /* Expander */
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #d9dee8;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load model files
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

with open(MODEL_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open(MODEL_DIR / "feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


# Header
st.markdown(
    """
    <div class="header-box">
        <h1>💳 Credit Card Fraud Detection</h1>
        <p>
            Analyze credit card transactions and predict whether
            a transaction is likely to be fraudulent.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Dataset overview
st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "283,726")

with col2:
    st.metric("Normal Transactions", "283,253")

with col3:
    st.metric("Fraud Transactions", "473")

with col4:
    st.metric("Fraud Rate", "0.17%")


st.write("")


# Analytics
st.subheader("📊 Transaction Analytics")

left, right = st.columns(2)

with left:

    st.write("**Transaction Distribution**")

    distribution = pd.DataFrame(
        {
            "Transactions": [283253, 473]
        },
        index=["Normal", "Fraud"]
    )

    st.bar_chart(
        distribution,
        height=240
    )


with right:

    st.write("**Average Transaction Amount**")

    amount_data = pd.DataFrame(
        {
            "Average Amount": [88.41, 123.87]
        },
        index=["Normal", "Fraud"]
    )

    st.bar_chart(
        amount_data,
        height=240
    )


st.divider()


# Prediction
st.subheader("🔍 Transaction Prediction")

st.info(
    "Enter the transaction values and click Predict Fraud."
)


col1, col2 = st.columns(2)

with col1:
    time_value = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

with col2:
    amount_value = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


# Advanced features
with st.expander("V1 - V28 Features"):

    st.caption(
        "These are anonymized features from the original dataset."
    )

    values = {}

    for start in range(1, 29, 4):

        cols = st.columns(4)

        for index, col in enumerate(cols):

            feature_number = start + index

            with col:
                values[f"V{feature_number}"] = st.number_input(
                    f"V{feature_number}",
                    value=0.0,
                    format="%.5f",
                    key=f"v{feature_number}"
                )


st.write("")


# Prediction button
if st.button(
    "🔎 Predict Fraud",
    use_container_width=True
):

    input_data = {
        "Time": time_value
    }

    input_data.update(values)

    input_data["Amount"] = amount_value

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    probability_percent = probability * 100


    st.divider()

    st.subheader("Prediction Result")

    result1, result2, result3 = st.columns(3)

    with result1:

        if prediction == 1:
            st.error("🚨 Fraud Transaction")
        else:
            st.success("✅ Normal Transaction")

    with result2:

        st.metric(
            "Fraud Probability",
            f"{probability_percent:.2f}%"
        )

    with result3:

        if probability < 0.20:
            risk = "Low"
        elif probability < 0.50:
            risk = "Medium"
        else:
            risk = "High"

        st.metric(
            "Risk Level",
            risk
        )

    st.progress(
        float(probability),
        text=f"Fraud Probability: {probability_percent:.2f}%"
    )


st.divider()


# Model information
st.subheader("🤖 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Algorithm**")
    st.write("Random Forest")

with col2:
    st.write("**Features**")
    st.write("30")

with col3:
    st.write("**ROC-AUC**")
    st.write("0.9246")

with col4:
    st.write("**PR-AUC**")
    st.write("0.7958")


st.caption(
    "This project is developed for educational and analytical purposes."
)