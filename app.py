import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Model imports
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# Load model and data
voting_model = joblib.load('voting_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
X_train = joblib.load('X_train.pkl')
X = joblib.load('X.pkl')  # For column names and feature means

# Setup
feature_names = X.columns.tolist()
feature_means = X.mean()

# Realistic value ranges (adjust as needed)
valid_ranges = {
    'N': (0, 140),
    'P': (5, 145),
    'K': (5, 205),
    'temperature': (8, 45),
    'humidity': (10, 100),
    'ph': (3.5, 9.5),
    'rainfall': (10, 300)
}

# App title
st.title("🌾 Crop Recommendation System")

st.subheader("📥 Enter Soil & Weather Features")

# User input fields
user_input = []
for feature in feature_names:
    default = float(round(feature_means[feature], 2))
    min_val, max_val = valid_ranges.get(feature, (0.0, 1000.0))
    value = st.number_input(
        f"{feature.capitalize()}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        step=0.1
    )
    user_input.append(value)

# Predict
if st.button("Predict Crop"):
    input_array = np.array([user_input])
    probs = voting_model.predict_proba(input_array)[0]
    top3_indices = np.argsort(probs)[::-1][:3]

    st.subheader("✅ Top-3 Recommended Crops")
    for idx in top3_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        prob = float(probs[idx])
        st.markdown(f"### 🌱 {crop_name}")
        st.write(f"🔎 Confidence: **{prob:.2%}**")

        # LIME explanation
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=label_encoder.classes_,
            mode='classification'
        )

        explanation = explainer.explain_instance(
            data_row=input_array[0],
            predict_fn=voting_model.predict_proba,
            labels=[idx],
            num_features=7
        )

        # Smaller chart
        fig = explanation.as_pyplot_figure(label=idx)
        fig.set_size_inches(4, 2.5)
        st.pyplot(fig)

        # Natural language reasoning
        top_features = explanation.as_list(label=idx)[:3]
        positives = [feat.split(' ')[0] for feat, weight in top_features if weight > 0]
        negatives = [feat.split(' ')[0] for feat, weight in top_features if weight < 0]

        reasoning_parts = []
        if positives:
            reasoning_parts.append(f"benefits from high {', '.join(positives)}")
        if negatives:
            reasoning_parts.append(f"but low {', '.join(negatives)} slightly reduces confidence")

        reasoning_text = ", ".join(reasoning_parts) + "." if reasoning_parts else ""
        if reasoning_text:
            st.markdown(f"🧠 {reasoning_text}")




