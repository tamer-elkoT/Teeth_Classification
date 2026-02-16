import streamlit as st
import requests
from PIL import Image
import io

# The URL of your FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Configuration ---
st.set_page_config(page_title="Teeth Classification AI", page_icon="🦷", layout="centered")

st.title("🦷 Teeth Classification AI")
st.markdown("""
Upload a dental X-ray or image to instantly classify the tooth condition.
Powered by custom Deep Learning models.
""")

st.divider()

# --- User Inputs ---
# 1. Dropdown for model selection
model_name = st.selectbox(
    "🧠 Select AI Model",
    ("CNN", "EfficientNetB0"),
    help="Choose which neural network to use for the prediction."
)

# 2. File uploader for the image
uploaded_file = st.file_uploader(
    "📂 Upload a dental image (JPEG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

# --- Prediction Action ---
if uploaded_file is not None:
    # Display the uploaded image beautifully
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # The Predict Button
    if st.button("🔍 Predict Classification", use_container_width=True):
        
        # Show a loading spinner while waiting for the FastAPI server
        with st.spinner("Analyzing image..."):
            
            try:
                # 1. Package the File (The raw bytes)
                # Note: The key "image" MUST match the exact parameter name in your FastAPI endpoint!
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # 2. Package the Form Data (The text string)
                # Note: The key "model_name" MUST match your FastAPI endpoint!
                data = {"model_name": model_name}

                # 3. Send the HTTP POST request to the FastAPI server
                response = requests.post(API_URL, files=files, data=data)

                # 4. Check if the server responded successfully (Status Code 200)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the results
                    st.success("✅ Prediction Complete!")
                    st.metric(label="Predicted Class", value=result["prediction"])
                    st.info(f"Model Used: {result['model_used']}")
                    
                else:
                    # If FastAPI threw your 400 or 500 error, display it here!
                    error_detail = response.json().get("detail", "Unknown Error")
                    st.error(f"⚠️ Server Error {response.status_code}: {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Is your FastAPI server running?")