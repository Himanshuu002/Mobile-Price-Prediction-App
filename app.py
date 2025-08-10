import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("mobile_price_prediction.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Mobile Price Prediction", layout="centered")
st.title("Mobile Price Prediction App")

st.write("""
This app predicts the price range of a mobile phone based on its specifications.  
Fill in the details below and get an instant prediction.
""")

st.sidebar.header("Input Mobile Specifications")

def user_input_features():
    battery_power = st.sidebar.number_input("Battery Power (mAh)", min_value=500, max_value=5000, value=2000)
    blue = st.sidebar.selectbox("Bluetooth", [0, 1])
    clock_speed = st.sidebar.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    dual_sim = st.sidebar.selectbox("Dual SIM", [0, 1])
    fc = st.sidebar.number_input("Front Camera (MP)", min_value=0, max_value=20, value=5)
    four_g = st.sidebar.selectbox("4G", [0, 1])
    int_memory = st.sidebar.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=32)
    m_dep = st.sidebar.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
    mobile_wt = st.sidebar.number_input("Mobile Weight (g)", min_value=50, max_value=300, value=150)
    n_cores = st.sidebar.slider("Number of Cores", min_value=1, max_value=8, value=4)
    pc = st.sidebar.number_input("Primary Camera (MP)", min_value=0, max_value=50, value=12)
    px_height = st.sidebar.number_input("Pixel Height", min_value=0, max_value=3000, value=800)
    px_width = st.sidebar.number_input("Pixel Width", min_value=0, max_value=3000, value=1200)
    ram = st.sidebar.number_input("RAM (MB)", min_value=256, max_value=16384, value=2048)
    sc_h = st.sidebar.number_input("Screen Height (cm)", min_value=0, max_value=30, value=14)
    sc_w = st.sidebar.number_input("Screen Width (cm)", min_value=0, max_value=30, value=7)
    talk_time = st.sidebar.number_input("Talk Time (hours)", min_value=2, max_value=50, value=10)
    three_g = st.sidebar.selectbox("3G", [0, 1])
    touch_screen = st.sidebar.selectbox("Touch Screen", [0, 1])
    wifi = st.sidebar.selectbox("WiFi", [0, 1])

    data = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi
    }

    return pd.DataFrame(data, index=[0])

# Collect input
input_df = user_input_features()

# Scale inputs before prediction
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Price Range"):
    prediction = model.predict(input_scaled)
    prediction_labels = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    st.success(f"Predicted Price Range: {prediction_labels[prediction[0]]}")

