import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_dt.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Osteoporosis - Decision Tree")

age = st.number_input("Age", 18, 100, 30)

gender_label = st.selectbox("Gender", ["Female","Male"])
hormonal_label = st.selectbox("Hormonal Changes", ["Normal","Postmenopausal"])
family_label = st.selectbox("Family History", ["No","Yes"])
race_label = st.selectbox("Race", ["African American","Asian","Caucasian"])
weight_label = st.selectbox("Body Weight", ["Normal","Underweight"])
calcium_label = st.selectbox("Calcium Intake", ["Adequate","Low"])
vitd_label = st.selectbox("Vitamin D", ["Insufficient","Sufficient"])
activity_label = st.selectbox("Physical Activity", ["Active","Sedentary"])
smoke_label = st.selectbox("Smoking", ["No","Yes"])
alcohol_label = st.selectbox("Alcohol", ["None","Moderate"])
medical_label = st.selectbox("Medical Condition", ["Hyper","None","RA"])
meds_label = st.selectbox("Medication", ["Corticosteroids","None"])
fracture_label = st.selectbox("Prior Fracture", ["No","Yes"])

# Mapping ke angka sesuai training
gender = 0 if gender_label=="Female" else 1
hormonal = 0 if hormonal_label=="Normal" else 1
family = 0 if family_label=="No" else 1
race_map = {"African American":0,"Asian":1,"Caucasian":2}
race = race_map[race_label]
weight = 0 if weight_label=="Normal" else 1
calcium = 0 if calcium_label=="Adequate" else 1
vitd = 0 if vitd_label=="Insufficient" else 1
activity = 0 if activity_label=="Active" else 1
smoke = 0 if smoke_label=="No" else 1
alcohol = 0 if alcohol_label=="None" else 1
medical_map = {"Hyper":0,"None":1,"RA":2}
medical = medical_map[medical_label]
meds = 0 if meds_label=="Corticosteroids" else 1
fracture = 0 if fracture_label=="No" else 1

data = np.array([[age,gender,hormonal,family,race,weight,calcium,vitd,
                  activity,smoke,alcohol,medical,meds,fracture]])

data_scaled = scaler.transform(data)

if st.button("Prediksi"):
    hasil = model.predict(data_scaled)
    if hasil[0] == 1:
        st.error("Berisiko Osteoporosis")
    else:
        st.success("Tidak Berisiko Osteoporosis")
