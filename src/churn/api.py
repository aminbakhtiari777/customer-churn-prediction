"""FastAPI application for customer churn inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from .schemas import CustomerFeatures, Prediction
from .service import ModelService

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_pipeline.joblib"))
DEFAULT_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))


def create_app(service: ModelService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.model_service is None:
            try:
                application.state.model_service = ModelService.from_path(
                    DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD
                )
            except (FileNotFoundError, ValueError) as exc:
                application.state.model_error = str(exc)
        yield

    application = FastAPI(
        title="Customer Churn MLOps API",
        version="1.0.0",
        description="Predicts customer churn probability from IBM Telco features.",
        lifespan=lifespan,
    )
    application.state.model_service = service
    application.state.model_error = None

    @application.get("/health")
    def health(request: Request) -> dict[str, str | bool | None]:
        loaded = request.app.state.model_service is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "detail": request.app.state.model_error,
        }

    @application.post("/predict", response_model=Prediction)
    def predict(payload: CustomerFeatures, request: Request) -> dict[str, float | bool]:
        model_service = request.app.state.model_service
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return model_service.predict(payload.model_dump())

    return application


app = create_app()
