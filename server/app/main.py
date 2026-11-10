from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import claims, documents, jobs
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Data Room Investigator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(documents.router)
    app.include_router(claims.router)
    app.include_router(jobs.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
