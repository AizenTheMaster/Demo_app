# -*- coding: utf-8 -*-
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


diabetes_model = pickle.load(open('diabetes_model1.sav', 'rb'))
heart_model_data = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# Handle dict-based heart model
heart_model = heart_model_data['model']
heart_scaler = heart_model_data['scaler']

# 📍 Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Made By Badmosh',
        ['🩸 Diabetes Prediction', '❤️ Heart Disease Prediction', '🧍 Parkinson’s Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# 🚨 Diabetes Page
if selected == '🩸 Diabetes Prediction':
    st.title('🩸 Diabetes Prediction System')

    # Input fields
    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure value')
    SkinThickness = st.number_input('Skin Thickness value')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI value')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    Age = st.number_input('Age of the Person')

    if st.button('Predict Diabetes'):
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness,
                               Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        prediction = diabetes_model.predict(input_data)

        st.success("🔴 Person is Diabetic" if prediction[0] == 1 else "🟢 Person is Not Diabetic")

# ❤️ Heart Page
elif selected == '❤️ Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction System')

    # Input fields
    age = st.number_input('Age')
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholesterol (mg/dl)')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved')
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.number_input('ST Depression (Oldpeak)', format="%.1f")
    slope = st.selectbox('Slope of ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0–3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

    if st.button('Predict Heart Condition'):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        input_scaled = heart_scaler.transform(input_data)
        prediction = heart_model.predict(input_scaled)

        st.success("🔴 Person has Heart Disease" if prediction[0] == 1 else "🟢 Person does NOT have Heart Disease")

# 🧍 Parkinson's Page
elif selected == '🧍 Parkinson’s Prediction':
    st.title("🧍 Parkinson’s Disease Prediction System")

    # 👉 22 Parkinson's features (collect in same order as model was trained)
    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    jitter_percent = st.number_input('MDVP:Jitter(%)')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)')
    rap = st.number_input('MDVP:RAP')
    ppq = st.number_input('MDVP:PPQ')
    ddp = st.number_input('Jitter:DDP')
    shimmer = st.number_input('MDVP:Shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)')
    apq3 = st.number_input('Shimmer:APQ3')
    apq5 = st.number_input('Shimmer:APQ5')
    apq = st.number_input('MDVP:APQ')
    dda = st.number_input('Shimmer:DDA')
    nhr = st.number_input('NHR')
    hnr = st.number_input('HNR')
    rpde = st.number_input('RPDE')
    dfa = st.number_input('DFA')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    d2 = st.number_input('D2')
    ppe = st.number_input('PPE')

    # ✅ Load model & scaler from dict
    model = parkinson_model['model']
    scaler = parkinson_model['scaler']

    if st.button("Predict Parkinson's"):
        # 🧠 Prepare input data
        input_data = np.array([
            fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]).reshape(1, -1)

        # 🔄 Scale the input
        scaled_input = scaler.transform(input_data)

        # 🔍 Predict
        prediction = model.predict(scaled_input)

        # 🎯 Output
        st.success("🔴 Person has Parkinson's Disease" if prediction[0] == 1 else "🟢 Person does NOT have Parkinson's Disease")
