"""
FastAPI Application

Main application entry point with:
- CORS middleware
- Health endpoints
- API routes
- OpenTelemetry hooks (minimal)
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

# Configure root logger BEFORE any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
# Silence noisy loggers
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from src.api.routes import router
from src.config import get_settings
from src.storage.database import init_db, get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    app.state.start_time = datetime.now(timezone.utc)
    app.state.request_count = 0
    app.state.error_count = 0

    # Initialise PostgreSQL schema
    try:
        await init_db()
        logger.info("PostgreSQL tables created / verified")
    except Exception as exc:
        logger.warning("PostgreSQL init failed (will retry on first request): %s", exc)

    yield

    # Shutdown — dispose of the SQLAlchemy engine
    engine = get_engine()
    await engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title="Causeway API",
    description="Agentic Decision Support System with Causal Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    
    # Ensure state is initialized
    if not hasattr(app.state, 'request_count'):
        app.state.request_count = 0
        app.state.error_count = 0
        app.state.start_time = datetime.now(timezone.utc)
    
    try:
        response = await call_next(request)
        app.state.request_count += 1
        return response
    except Exception:
        app.state.error_count += 1
        raise


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Basic metrics endpoint."""
    uptime = datetime.now(timezone.utc) - app.state.start_time
    
    return {
        "uptime_seconds": uptime.total_seconds(),
        "request_count": app.state.request_count,
        "error_count": app.state.error_count,
    }


# Include API routes
app.include_router(router, prefix="/api/v1")
