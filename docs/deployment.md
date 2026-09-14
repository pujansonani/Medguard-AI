# MEDGUARD AI: Deployment & Infrastructure

## 1. Architecture Overview
MEDGUARD AI provides high-performance dual interfaces:
1. **Streamlit Analytics Portal (`app/streamlit_app.py`):** Interactive clinical analytics dashboard designed for researchers, bioinformaticians, and clinical investigators.
2. **FastAPI REST Backend (`api/main.py`):** High-throughput microservice exposing asynchronous predictions, SHAP attributions, multi-horizon trajectories, and telemetry feeds.
3. **Interactive Web Dashboard (`web/index.html`):** Clinera.ai-inspired responsive web application with 60 FPS live canvas waveforms (ECG Lead II + Arterial Blood Pressure) and 24h simulation controls.

---

## 2. Docker Deployment

### Building & Running with Docker Compose
```bash
# Build both backend and frontend services
docker compose build

# Launch full stack
docker compose up -d
```
- **Backend REST API:** `http://localhost:8000` (Swagger UI at `/docs`)
- **Streamlit Analytics:** `http://localhost:8501`

### Running Standalone Container
```bash
docker build -t medguard-ai:latest .
docker run -p 8000:8000 -p 8501:8501 medguard-ai:latest
```

---

## 3. Production Monitoring & Drift Detection
The API includes `/monitoring/drift` to measure:
- Population Stability Index (PSI) on incoming physiological batches.
- Prediction distribution shifts over time.
- Sensor missingness rate anomalies.
