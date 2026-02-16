"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.model import router as model_router
from src.api.routes.predict import router as predict_router
from src.api.routes.visualize import router as visualize_router
from src.api.state import ModelState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"


def create_app(weights_path: Path | str | None = None) -> FastAPI:
    app = FastAPI(title="MNIST Vision Explorer", version="0.1.0")
    app.state.model_state = ModelState(weights_path=weights_path)

    app.include_router(predict_router)
    app.include_router(visualize_router)
    app.include_router(model_router)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()


def run_server() -> None:
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
