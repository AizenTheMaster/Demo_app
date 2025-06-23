# combine_app.py
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# 🧠 Load models
diabetes_model = pickle.load(open('diabetes_model1.sav', 'rb'))
heart_model_data = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinson_model_data = pickle.load(open('parkinsons_model.sav', 'rb'))

# 🎯 Extract model + scaler from dicts
heart_model = heart_model_data['model']
heart_scaler = heart_model_data['scaler']
parkinson_model = parkinson_model_data['model']
parkinson_scaler = parkinson_model_data['scaler']

# 🌈 Custom CSS
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
.stApp {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #0f4c75;
}
.stButton > button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
div[data-testid="stForm"] {
    background-color: #f0f8ff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# 🎛️ Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="🧬 Disease Detection",
        options=["🩸 Diabetes", "❤️ Heart", "🧍 Parkinson's"],
        icons=["droplet-half", "heart-pulse", "person"],
        default_index=0
    )

# =================== 🩸 Diabetes ===================
if selected == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("Pregnancies", min_value=0)
            bp = st.number_input("Blood Pressure")
            insulin = st.number_input("Insulin")
            dpf = st.number_input("Diabetes Pedigree Function")
        with col2:
            glucose = st.number_input("Glucose")
            skin = st.number_input("Skin Thickness")
            bmi = st.number_input("BMI")
            age = st.number_input("Age")

        if st.form_submit_button("🚀 Predict"):
            input_data = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
            prediction = diabetes_model.predict(input_data)
            st.success("🟢 No Diabetes Detected" if prediction[0] == 0 else "🔴 Person is Diabetic")

# =================== ❤️ Heart ===================
elif selected == "❤️ Heart":
    st.title("❤️ Heart Disease Prediction")
    with st.form("heart_form"):
        age = st.number_input('Age')
        sex = st.selectbox('Sex', ['Female', 'Male'])
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        trestbps = st.number_input('Resting BP')
        chol = st.number_input('Cholesterol')
        fbs = st.selectbox('Fasting Sugar >120 mg/dl', [0, 1])
        restecg = st.selectbox('Resting ECG', [0, 1, 2])
        thalach = st.number_input('Max Heart Rate')
        exang = st.selectbox('Exercise Induced Angina', [0, 1])
        oldpeak = st.number_input('ST Depression', format="%.1f")
        slope = st.selectbox('Slope of ST Segment', [0, 1, 2])
        ca = st.selectbox('Major Vessels (0-3)', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', [1, 2, 3])

        if st.form_submit_button("🚀 Predict"):
            sex_val = 1 if sex == 'Male' else 0
            input_data = np.array([age, sex_val, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            input_scaled = heart_scaler.transform(input_data)
            prediction = heart_model.predict(input_scaled)
            st.success("🟢 Heart is Healthy" if prediction[0] == 0 else "🔴 Heart Disease Detected")

# =================== 🧍 Parkinson's ===================
elif selected == "🧍 Parkinson's":
    st.title("🧍 Parkinson’s Disease Prediction")
    with st.form("parkinsons_form"):
        st.caption("📌 Enter 22 acoustic and signal parameters:")

        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)')
            fhi = st.number_input('MDVP:Fhi(Hz)')
            flo = st.number_input('MDVP:Flo(Hz)')
            jitter_percent = st.number_input('Jitter (%)')
            jitter_abs = st.number_input('Jitter (Abs)')
            rap = st.number_input('RAP')
            ppq = st.number_input('PPQ')
            ddp = st.number_input('DDP')
            shimmer = st.number_input('Shimmer')
            shimmer_db = st.number_input('Shimmer (dB)')
            apq3 = st.number_input('APQ3')
        with col2:
            apq5 = st.number_input('APQ5')
            apq = st.number_input('APQ')
            dda = st.number_input('DDA')
            nhr = st.number_input('NHR')
            hnr = st.number_input('HNR')
            rpde = st.number_input('RPDE')
            dfa = st.number_input('DFA')
            spread1 = st.number_input('Spread1')
            spread2 = st.number_input('Spread2')
            d2 = st.number_input('D2')
            ppe = st.number_input('PPE')

        if st.form_submit_button("🚀 Predict"):
            input_data = np.array([
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]).reshape(1, -1)
            input_scaled = parkinson_scaler.transform(input_data)
            prediction = parkinson_model.predict(input_scaled)
            st.success("🟢 No Parkinson’s Detected" if prediction[0] == 0 else "🔴 Parkinson’s Detected")

# 📎 Footer
st.markdown("""<hr><center><b>Made with ❤️ by Badmosh</b></center>""", unsafe_allow_html=True)
