import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config first
st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")

# Set background image URL
background_image_url = "https://atl.images.passionperformance.ca/content/photos/52/56/525636-dodge-challenger-d-occasion-quelle-version-choisir.jpeg"

# Apply background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model & preprocessor
model = joblib.load(r"E:\New DS projects\Excelr\Cars24 Price Prediction\deployment\hist_gradient_boosting_model.pkl")
preprocessor = joblib.load(r"E:\New DS projects\Excelr\Cars24 Price Prediction\deployment\car_price_preprocessor.pkl")

st.title("🚗 Car Price Prediction App")
st.markdown("Predict the resale price of a car based on its features.")

# Sidebar for inputs
st.sidebar.header("Car Details Input")

year = st.sidebar.number_input("Year", min_value=1980, max_value=2025, value=2020)
kilometerdriven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=20000)
ownernumber = st.sidebar.selectbox("Owner Number", [1, 2, 3, 4])
benefits = st.sidebar.number_input("Benefits", min_value=0, value=0)
discountprice = st.sidebar.number_input("Discount Price", min_value=0, value=0)

isc24assured = st.sidebar.selectbox("C24 Assured?", ["No", "Yes"])
isc24assured = 1 if isc24assured == "Yes" else 0

make = st.sidebar.text_input("Make", "Toyota")
city = st.sidebar.text_input("City", "Mumbai")
registrationstate = st.sidebar.text_input("Registration State", "MH")
fueltype = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
bodytype = st.sidebar.selectbox("Body Type", ["Hatchback", "Sedan", "SUV", "MUV"])

if st.sidebar.button("Predict Price"):
    # Automatically calculate car_age
    car_age = 2025 - year

    # Create DataFrame from inputs
    input_df = pd.DataFrame([{
        'year': year,
        'kilometerdriven': kilometerdriven,
        'ownernumber': ownernumber,
        'benefits': benefits,
        'discountprice': discountprice,
        'car_age': car_age,
        'isc24assured': isc24assured,
        'make': make,
        'city': city,
        'registrationstate': registrationstate,
        'fueltype': fueltype,
        'transmission': transmission,
        'bodytype': bodytype
    }])

    # Transform input
    input_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_transformed)[0]
    
    st.markdown(
    f"""
    <div style="
        background-color: #28a745; 
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    ">
        💰 Estimated Price: ₹{prediction:,.2f}
    </div>
    """,
    unsafe_allow_html=True
    )
