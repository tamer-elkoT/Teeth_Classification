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