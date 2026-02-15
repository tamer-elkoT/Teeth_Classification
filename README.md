## 1. 🎯 The User Scenario (The "Why")
Imagine a dentist or a dental student using this application.

* **The Entry:** The user opens their web browser and navigates to the Streamlit application.
* **The Input:** They upload an X-ray or a standard photo of a single tooth.
* **The Choice:** They use a dropdown menu to select which AI brain they want to consult: the custom CNN or the heavier EfficientNetB0.
* **The Action:** They click a "Classify Tooth" button.
* **The Result:** A few milliseconds later, the screen updates to display the predicted class (e.g., "Canine") along with a success message.

## 2. 🏗️ The System Architecture (The Big Picture)

<p align="center">
  <img src="assets/architecture of teeth classification project.svg" alt="Teeth Classification Architecture" width="800"/>
</p>

The system is essentially three separate "layers" that talk to each other, housed inside two isolated Docker containers.
* **Container A - The View Layer (Streamlit):** This is the frontend. It knows absolutely nothing about TensorFlow, neural networks, or image preprocessing. Its only job is to draw the UI, collect the image from the user, and send it away.
* **Container B - The Controller & Model Layer (FastAPI):** This is the heavy lifter. It holds the API endpoints, the TensorFlow library, and the `.h5` model files.
* **The Bridge (Docker Network):** Because Container A and Container B are separate, they need a way to talk. Docker creates an internal network so Streamlit can send the image via an HTTP POST request to FastAPI, just like a website talking to a server.

## 3. 🔄 The System Flow (The Journey of the Image)
Here is the exact sequence of events when the user clicks "Predict":

1.  **Upload:** Streamlit captures the image file as raw bytes.
2.  **Transmission:** Streamlit packages these raw bytes and the user's model choice (e.g., "CNN") into a payload and shoots it over to FastAPI via the `/predict` endpoint.
3.  **Validation:** FastAPI receives the payload. It first checks: "Is this actually an image?" 
4.  **Preprocessing:** FastAPI hands the raw bytes to the `inference.py` script. The script opens the bytes, resizes them to the target size (e.g., 224x224), and normalizes the pixels (dividing by 255.0).
5.  **Prediction:** The preprocessed tensor is fed into the globally loaded model (retrieved via `model_loader.py`).
6.  **Translation:** The model outputs an array of probabilities. `numpy` finds the highest probability, and the `config.yaml` file translates that index (e.g., `1`) into a human-readable string (e.g., "CoS").
7.  **Return:** FastAPI packages that string into a JSON response `{"prediction": "CoS"}` and sends it back to Streamlit.
8.  **Display:** Streamlit unpacks the JSON and displays the word "CoS" on the screen.

## 📂 4. Project Architecture
```text
teeth_classification_project/
│
├── config/                 # Global configurations (paths, constants)
│   └── config.yaml
│
├── src/                    # === THE MODEL LAYER ===
│   ├── __init__.py
│   ├── model_loader.py     # Logic to load the .pth or .h5 file safely
│   ├── preprocessing.py    # Transformations (Resize, Normalization)
│   └── inference.py        # The core prediction engine
│
├── api/                    # === THE CONTROLLER LAYER ===
│   ├── main.py             # FastAPI entry point
│   ├── schemas.py          # Pydantic models for request/response validation
│   └── routers.py          # API endpoints (e.g., /predict)
│
├── frontend/               # === THE VIEW LAYER ===
│   └── app.py              # Streamlit application
│
├── docker/                 # Containerization
│   ├── Dockerfile.api      # Dockerfile for FastAPI
│   └── Dockerfile.ui       # Dockerfile for Streamlit
│
├── tests/                  # Unit tests
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt
└── README.md

```

# 🦷 Teeth Classification: Deployment Architecture & Roadmap

This repository contains the end-to-end machine learning pipeline and deployment architecture for a Teeth Classification system. The project is built iteratively using the **Feature Branch Workflow**, cleanly separating the AI models (CNN / EfficientNetB0), the FastAPI backend, and the Streamlit frontend, all orchestrated via Docker.

## 🚀 Development Roadmap

### 🏗️ Phase 1: Foundation (`main` branch)
* **Goal:** Set up the project scaffolding.
* **Deliverables:** * [ ] Initialize the empty folder structure.
    * [ ] Create an initial `requirements.txt`.
    * [ ] Setup `config/config.yaml` containing model paths and class names.

### 🧠 Phase 2: The Model Engine (`feature/model-engine`)
* **Goal:** Ensure the application can load models and process images perfectly without any web interfaces.
* **Deliverables:** * [ ] `src/model_loader.py`: Logic to load `.h5` files into memory globally.
    * [ ] `src/inference.py`: Functions to preprocess the image and run `model.predict()`.
* **Test:** * [ ] Write a unit test in `tests/test_model.py` that passes a dummy image to the functions and verifies it outputs a valid tooth class.

### 🔌 Phase 3: The API Bridge (`feature/api-bridge`)
* **Goal:** Wrap the model engine in a web server so it can receive data over the network.
* **Deliverables:**
    * [ ] `api/routers.py`: Implement the `/predict` endpoint logic.
    * [ ] `api/main.py`: Initialize the FastAPI application.
* **Test:** * [ ] Run the FastAPI server locally.
    * [ ] Test the endpoint using the built-in Swagger UI (`localhost:8000/docs`) by manually uploading an image.

### 🖥️ Phase 4: The User Interface (`feature/streamlit-ui`)
* **Goal:** Build the visual dashboard for the user.
* **Deliverables:**
    * [ ] `frontend/app.py`: Create the Streamlit layout with a file uploader and model selection dropdown.
    * [ ] Implement the `requests.post()` logic to send data to the local FastAPI server.
* **Test:** * [ ] Run both Streamlit and FastAPI locally.
    * [ ] Successfully make a prediction through the Streamlit UI.

### 🐳 Phase 5: Containerization (`feature/dockerization`)
* **Goal:** Package the entire system into isolated containers so it can run on any machine without requiring Python or TensorFlow installations.
* **Deliverables:**
    * [ ] `docker/Dockerfile.api`: Instructions to build the backend image.
    * [ ] `docker/Dockerfile.ui`: Instructions to build the frontend image.
    * [ ] `docker-compose.yml`: The orchestration script to boot up