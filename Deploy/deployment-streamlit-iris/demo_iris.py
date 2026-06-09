import streamlit as st
import joblib

model = joblib.load("iris.joblib")

st.title("Iris Flower Classification")

sl = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
pl = st.number_input("Petal Length", 0.0, 10.0, 1.4)
pw = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])[0]

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(f"Prediksi: {classes[pred]}")
