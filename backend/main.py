"""
FastAPI Server Entrypoint.
Emergency Vehicle Route & Traffic Congestion Prediction Cloud System.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api_routes import router as api_router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="Emergency Vehicle Route & Congestion Predictor",
    description="A YOLO and LSTM-Based Cloud System for Predictive, Congestion-Aware Emergency Routing",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST Router
app.include_router(api_router)

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "emergency_routing_cloud"}

# Mount Frontend Static Assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_dashboard():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        
    @app.get("/app.js")
    def serve_js():
        return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        
    @app.get("/style.css")
    def serve_css():
        return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
