import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Paris Housing Price Predictor", layout="centered")
st.title("Paris Housing Price Predictor")

# === Upload model and scaler files ===
model_file = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"], key="model")
scaler_file = st.file_uploader("Upload Feature Scaler (.pkl)", type=["pkl"], key="scaler")

# === Load files once both are uploaded ===
if model_file is not None and scaler_file is not None:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

    st.success("Model and Scaler loaded successfully!")

    # === Input fields ===
    squareMeters = st.number_input("Area (Square Meters)", min_value=10, max_value=1000, value=50, step=10)
    numberOfRooms = st.slider("Number of Rooms", 1, 10, 3)
    hasYard = st.checkbox("Has Yard")
    hasPool = st.checkbox("Has Pool")
    floors = st.slider("Number of Floors", 1, 5, 1)
    cityCode = st.number_input("City Code", min_value=1000, max_value=99999, value=75000, step=100)
    cityPartRange = st.slider("City Part Range", 1, 10, 5)
    numPrevOwners = st.slider("Previous Owners", 0, 10, 1)
    made = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    isNewBuilt = st.checkbox("Newly Built")
    hasStormProtector = st.checkbox("Has Storm Protector")
    basement = st.slider("Basement Area (sq m)", 0, 300, 0)
    attic = st.slider("Attic Area (sq m)", 0, 200, 0)
    garage = st.slider("Garage Area (sq m)", 0, 300, 0)
    hasStorageRoom = st.checkbox("Has Storage Room")
    hasGuestRoom = st.checkbox("Has Guest Room")

    # === Collect input ===
    input_data = np.array([[
        squareMeters,
        numberOfRooms,
        int(hasYard),
        int(hasPool),
        floors,
        cityCode,
        cityPartRange,
        numPrevOwners,
        made,
        int(isNewBuilt),
        int(hasStormProtector),
        basement,
        attic,
        garage,
        int(hasStorageRoom),
        int(hasGuestRoom)
    ]])

    input_scaled = scaler.transform(input_data)

    # === Predict ===
    if st.button("Predict Price"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"Estimated Price: €{prediction:,.2f}")

else:
    st.warning("Please upload both the trained model and scaler to proceed.")
