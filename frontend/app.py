import streamlit as st
import requests
from PIL import Image
import json
from streamlit_lottie import st_lottie

# --- Configuration & Constants ---
API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="Teeth Classification AI", page_icon="🦷", layout="wide")

# --- Helper Functions ---
def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Initialize Session State for History Dashboard
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2806/2806554.png", width=80) # Placeholder tooth icon
    st.title("Settings & Info")
    st.markdown("Configure your AI inference parameters here.")
    
    model_choice = st.selectbox(
        "🧠 Select AI Engine",
        ("CNN", "EfficientNetB0"),
        help="EfficientNetB0 generally provides higher accuracy but might take slightly longer."
    )
    
    st.divider()
    st.markdown("### About")
    st.info("This system uses deep learning to classify dental X-rays into 7 distinct categories. Upload a clear, cropped image for best results.")

# --- Main Dashboard ---
# Load a medical/AI animation
lottie_medical = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_5n8ox9di.json")

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🦷 Intelligent Dental Analysis")
    st.markdown("### Upload an image to detect and classify tooth conditions.")
with col2:
    if lottie_medical:
        st_lottie(lottie_medical, height=120, key="medical_anim")

st.divider()

# --- Tabs for Organization ---
tab1, tab2 = st.tabs(["🔍 Live Analysis", "📊 Session History"])

# TAB 1: Live Analysis
with tab1:
    upload_col, result_col = st.columns([1, 1], gap="large")
    
    with upload_col:
        st.subheader("1. Upload X-Ray")
        uploaded_file = st.file_uploader("Drag and drop a JPEG/PNG file here", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
            
            analyze_button = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    with result_col:
        st.subheader("2. Analysis Results")
        
        if uploaded_file is not None and analyze_button:
            with st.spinner("Processing image through Neural Network..."):
                try:
                    # Package and send to FastAPI
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"model_name": model_choice}
                    response = requests.post(API_URL, files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        
                        # Display stunning metrics
                        st.success("Analysis Complete!")
                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Predicted Class", prediction)
                        metric_col2.metric("Engine Used", result["model_used"])
                        
                        st.info("Diagnosis requires review by a certified dental professional.")
                        
                        # Save to session history
                        st.session_state.history.append({
                            "file": uploaded_file.name,
                            "prediction": prediction,
                            "model": result["model_used"]
                        })
                        
                    else:
                        error_detail = response.json().get("detail", "Unknown Error")
                        st.error(f"Server Error {response.status_code}: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to the Backend API. Ensure Uvicorn is running.")
        elif uploaded_file is None:
            st.info("👈 Upload an image on the left to see results here.")

# TAB 2: Session History
with tab2:
    st.subheader("Patient Session Log")
    if len(st.session_state.history) == 0:
        st.write("No analyses performed in this session yet.")
    else:
        # Display history as a clean, interactive table
        st.dataframe(st.session_state.history, use_container_width=True)