import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Pro Car Price Prediction & AI Valuation Dashboard",
    page_icon="🚗",
    layout="wide",
)

# Safe Absolute Path Setup for Cloud & Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "car_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

# Title and Description
st.title("🚗 Pro Car Price Prediction & AI Valuation Dashboard")
st.markdown(
    "Advanced Indian Car Market Intelligence with Hotness Meter & AI Assistant"
)


# Load Dataset Safely
@st.cache_data
def load_data():
  if os.path.exists(DATA_PATH):
    return pd.read_csv(DATA_PATH)
  else:
    return None


# Load Model Safely
@st.cache_resource
def load_model():
  if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
      return pickle.load(f)
  else:
    return None


df = load_data()
model = load_model()

# Check if Dataset & Model Exist
if df is None or model is None:
  st.error(
      "Model or Dataset not found! Please ensure `car_data.csv` and"
      " `car_price_model.pkl` are in the repository."
  )
else:
  # Sidebar Inputs for Prediction
  st.sidebar.header("🔧 Configure Car Parameters")

  # Example input fields (Adjust based on your actual model features)
  car_brand = st.sidebar.selectbox(
      "Select Brand", df["Brand"].unique() if "Brand" in df.columns else ["Tata"]
  )
  year = st.sidebar.slider("Manufacturing Year", 2010, 2026, 2020)
  kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 200000, 30000)
  fuel_type = st.sidebar.selectbox(
      "Fuel Type",
      df["Fuel_Type"].unique() if "Fuel_Type" in df.columns else ["Petrol"],
  )

  # Prediction Button
  if st.sidebar.button("Predict Price"):
    st.success("🎉 Model loaded successfully! (Add your model prediction logic here)")

  # Main Dashboard Display
  st.subheader("📊 Market Overview")
  st.dataframe(df.head())