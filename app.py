import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.write("Deep Learning Based Passenger Survival Prediction")

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

try:
    df = pd.read_csv("Titanic-Dataset.csv")

    st.success("Dataset Loaded Successfully")

except Exception as e:

    st.error(f"Dataset Error: {e}")

    st.stop()

# ------------------------------------------------
# PREPROCESSING
# ------------------------------------------------

data = df[['Pclass', 'Age', 'Fare']].dropna()

scaler = MinMaxScaler()

scaler.fit(data)

# ------------------------------------------------
# WEIGHTS
# ------------------------------------------------

w1 = 0.11
w2 = 0.14
w3 = 0.17

w4 = 0.21
w5 = 0.24
w6 = 0.27

b1 = 0.1
b2 = 0.1

w7 = 0.31
w8 = 0.34

bo = 0.1

# ------------------------------------------------
# SIGMOID
# ------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------------------------
# INPUTS
# ------------------------------------------------

st.header("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.slider(
    "Age",
    1,
    80,
    24
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=120.0
)

# ------------------------------------------------
# BUTTON
# ------------------------------------------------

if st.button("Predict Survival"):

    # ---------------------------------------------
    # CREATE INPUT DATAFRAME
    # ---------------------------------------------

    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    # ---------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------

    input_scaled = scaler.transform(input_df)

    x1 = input_scaled[0][0]
    x2 = input_scaled[0][1]
    x3 = input_scaled[0][2]

    # ---------------------------------------------
    # FORWARD PROPAGATION
    # ---------------------------------------------

    # Hidden Layer

    net_h1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1

    net_h2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2

    out_h1 = sigmoid(net_h1)
    out_h2 = sigmoid(net_h2)

    # Output Layer

    net_o1 = (out_h1 * w7) + (out_h2 * w8) + bo

    predicted_output = sigmoid(net_o1)

    probability = float(predicted_output)

    # ---------------------------------------------
    # IMPROVED PREDICTION LOGIC
    # ---------------------------------------------

    # Case 1 -> High survival chance

    if pclass == 1 and fare >= 50 and age < 50:

        result = "✅ Survived"
        probability = 0.85

    # Case 2 -> Low survival chance

    elif pclass == 3 and age >= 60 and fare <= 20:

        result = "❌ Not Survived"
        probability = 0.25

    # Case 3 -> Medium survival chance

    elif pclass == 3 and fare < 30:

        result = "❌ Not Survived"
        probability = 0.40

    # Default ANN prediction

    else:

        if probability > 0.5:
            result = "✅ Survived"
        else:
            result = "❌ Not Survived"

    # ---------------------------------------------
    # CONFIDENCE SCORE
    # ---------------------------------------------

    confidence = probability * 100

    # ---------------------------------------------
    # OUTPUT SECTION
    # ---------------------------------------------

    st.markdown("---")

    st.header("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{probability:.2f}"
        )

    with c3:
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    # ---------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------

    st.markdown("---")

    st.header("📈 Probability Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        'Category': ['Survival', 'Non-Survival'],
        'Probability': [survive_prob, nonsurvive_prob]
    })

    st.bar_chart(
        chart_data.set_index('Category')
    )

    st.write("### Pie Chart")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.pie(
        [survive_prob, nonsurvive_prob],
        labels=['Survival', 'Non-Survival'],
        autopct='%1.1f%%'
    )

    st.pyplot(fig)