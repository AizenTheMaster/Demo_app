# -*- coding: utf-8 -*-
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# 📦 Load Models
diabetes_model = pickle.load(open('diabetes_model1.sav', 'rb'))
heart_model_data = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinsons_model.sav', 'rb'))

heart_model = heart_model_data['model']
heart_scaler = heart_model_data['scaler']
parkinson_model_obj = parkinson_model['model']
parkinson_scaler = parkinson_model['scaler']

# 🌐 Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title='💉 Multiple Disease Prediction',
        options=['🩸 Diabetes', '❤️ Heart Disease', '🧍 Parkinson’s'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# 🧠 Centered Layout Helper
def centered_input(prompt):
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        return st.number_input(prompt)

# ✅ Diabetes Page
if selected == '🩸 Diabetes':
    st.title("🩸 Diabetes Prediction")

    with st.form("diabetes_form"):
        st.subheader("🧾 Enter Patient Info")
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input("Pregnancies", min_value=0)
            BloodPressure = st.number_input("Blood Pressure")
            Insulin = st.number_input("Insulin")
            DiabetesPedigreeFunction = st.number_input("Pedigree Function")
        with col2:
            Glucose = st.number_input("Glucose Level")
            SkinThickness = st.number_input("Skin Thickness")
            BMI = st.number_input("BMI")
            Age = st.number_input("Age")

        submitted = st.form_submit_button("🔍 Predict Diabetes")
        if submitted:
            input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                   Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
            prediction = diabetes_model.predict(input_data)

            st.success("🟢 No Diabetes Detected" if prediction[0] == 0 else "🔴 Diabetic")

# ❤️ Heart Page
elif selected == '❤️ Heart Disease':
    st.title("❤️ Heart Disease Prediction")

    with st.form("heart_form"):
        st.subheader("🧾 Enter Heart Data")
        age = st.number_input('Age')
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        trestbps = st.number_input('Resting BP')
        chol = st.number_input('Cholesterol')
        fbs = st.selectbox('Fasting Sugar >120', [0, 1])
        restecg = st.selectbox('ECG Results', [0, 1, 2])
        thalach = st.number_input('Max Heart Rate')
        exang = st.selectbox('Exercise Induced Angina', [0, 1])
        oldpeak = st.number_input('Oldpeak', format="%.1f")
        slope = st.selectbox('Slope of ST', [0, 1, 2])
        ca = st.selectbox('Major Vessels Colored', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', [1, 2, 3])

        submitted = st.form_submit_button("🔍 Predict Heart Condition")
        if submitted:
            input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            input_scaled = heart_scaler.transform(input_data)
            prediction = heart_model.predict(input_scaled)

            st.success("🟢 Heart is Healthy" if prediction[0] == 0 else "🔴 Heart Disease Detected")

# 🧍 Parkinson Page
elif selected == '🧍 Parkinson’s':
    st.title("🧍 Parkinson’s Prediction")

    with st.form("parkinson_form"):
        st.subheader("🧾 Enter Voice & Signal Features")

        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)')
            fhi = st.number_input('MDVP:Fhi(Hz)')
            flo = st.number_input('MDVP:Flo(Hz)')
            jitter_percent = st.number_input('MDVP:Jitter(%)')
            jitter_abs = st.number_input('MDVP:Jitter(Abs)')
            rap = st.number_input('MDVP:RAP')
            ppq = st.number_input('MDVP:PPQ')
            ddp = st.number_input('Jitter:DDP')
            shimmer = st.number_input('MDVP:Shimmer')
            shimmer_db = st.number_input('Shimmer(dB)')
            apq3 = st.number_input('Shimmer:APQ3')
        with col2:
            apq5 = st.number_input('Shimmer:APQ5')
            apq = st.number_input('MDVP:APQ')
            dda = st.number_input('Shimmer:DDA')
            nhr = st.number_input('NHR')
            hnr = st.number_input('HNR')
            rpde = st.number_input('RPDE')
            dfa = st.number_input('DFA')
            spread1 = st.number_input('Spread1')
            spread2 = st.number_input('Spread2')
            d2 = st.number_input('D2')
            ppe = st.number_input('PPE')

        submitted = st.form_submit_button("🔍 Predict Parkinson's")
        if submitted:
            input_data = np.array([
                fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]).reshape(1, -1)

            scaled_input = parkinson_scaler.transform(input_data)
            prediction = parkinson_model_obj.predict(scaled_input)

            st.success("🟢 No Parkinson's Detected" if prediction[0] == 0 else "🔴 Parkinson's Detected")

# 🧾 Footer
st.markdown("""<hr><center>Made with ❤️ by Badmosh</center>""", unsafe_allow_html=True)
