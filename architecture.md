
``` mermaid

graph TD
    %% Define Styles
    classDef container fill:#f9f9fa,stroke:#333,stroke-width:2px,rx:10px;
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef frontend fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef backend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef database fill:#eceff1,stroke:#607d8b,stroke-width:2px;

    %% The User
    User([🧑‍⚕️ Dentist / Student]):::user

    %% Container A: Frontend
    subgraph Container_A [🖥️ Container A: Streamlit UI]
        direction TB
        UI[frontend/app.py<br>Web Interface]:::frontend
    end

    %% Docker Network Bridge
    UI -- "2. POST /predict over<br>Docker Network" --> API

    %% Container B: Backend
    subgraph Container_B [⚙️ Container B: FastAPI & Models]
        direction TB
        API[api/routers.py<br>FastAPI Endpoint]:::backend
        Infer[src/inference.py<br>Preprocessing & Predict]:::model
        Loader[src/model_loader.py<br>Global Cache]:::model
        Config[config/config.yaml<br>Class Names]:::database
        Models[(Saved .h5 Models<br>CNN / EfficientNet)]:::database
        
        %% Internal Backend Flow
        API -- "3. Validates File<br>& Passes Bytes" --> Infer
        Loader -. "Loads once at startup" .-> Models
        Infer -. "Requests loaded model" .-> Loader
        Infer -- "4. Resizes & Normalizes<br>5. model.predict()" --> Infer
        Config -. "Translates Index<br>to String" .-> Infer
        Infer -- "6. Returns String<br>(e.g., 'CoS')" --> API
    end

    %% External Connections to User
    User -- "1. Uploads Image<br>& Chooses Model" --> UI
    API -- "7. Returns JSON Response" --> UI
    UI -- "8. Displays Result" --> User

    %% Apply Container Styles
    class Container_A,Container_B container;
    
    ```