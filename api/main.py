# api/main.py

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

from core.severity_engine import score_from_raw
from core.cost_predictor import predict_cost_from_engine_output
from core.groq_narrator import generate_narrative
from search.search_engine import search_by_text, search_by_image
from vision.vector_store import get_index_stats

load_dotenv()

app = FastAPI(
    title="VisionLens — Visual AI Damage Pricing API",
    description="CLIP-powered visual search + AI damage assessment for Spinny & Cars24",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    make:              str  = Field(..., example="Maruti Suzuki")
    model:             str  = Field(..., example="Swift")
    panel:             str  = Field(..., example="Front Bumper Panel")
    issues:            str  = Field(..., example="Scratch-Minor, Dent-Major")
    include_narrative: bool = Field(default=True)

class TextSearchRequest(BaseModel):
    query:     str  = Field(..., example="grey sedan with dents under 8 lakhs in Bengaluru")
    tenant_id: str  = Field(default="spinny_india_prod")
    top_k:     int  = Field(default=5, ge=1, le=20)

class HealthResponse(BaseModel):
    status:  str
    version: str
    message: str


# ── Health ────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status":  "healthy",
        "version": "2.0.0",
        "message": "VisionLens API is live"
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status":  "healthy",
        "version": "2.0.0",
        "message": "All systems operational"
    }


# ── V1 — Pricing Engine (Iteration 1) ────────────────────────────

@app.post("/v1/analyze")
def analyze_panel(req: AnalyzeRequest):
    try:
        engine_out = score_from_raw(req.issues, req.panel)
        cost_out   = predict_cost_from_engine_output(engine_out)
        narrative  = None

        if req.include_narrative:
            narrative = generate_narrative(
                panel=req.panel,
                issues=engine_out["issues_parsed"],
                damage_score=cost_out["damage_score"],
                cost_bucket=cost_out["cost_bucket"],
                cost_range=cost_out["cost_range"],
                make=req.make,
                model=req.model,
            )

        return {
            "vehicle":          {"make": req.make, "model": req.model},
            "panel":            req.panel,
            "issues_parsed":    engine_out["issues_parsed"],
            "damage_score":     cost_out["damage_score"],
            "cost_bucket":      cost_out["cost_bucket"],
            "cost_range":       cost_out["cost_range"],
            "cost_min":         cost_out["cost_min"],
            "cost_max":         cost_out["cost_max"],
            "priority":         cost_out["priority"],
            "narrative":        narrative,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── V2 — VisionLens Visual Search (Iteration 2) ──────────────────

@app.post("/v2/search/text")
def text_search(req: TextSearchRequest):
    """
    Natural language → CLIP → Pinecone → ranked vehicles with cost
    """
    try:
        result = search_by_text(
            query=req.query,
            tenant_id=req.tenant_id,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/search/image")
async def image_search(
    file:      UploadFile = File(...),
    tenant_id: str        = "spinny_india_prod",
    max_price: int        = None,
    city:      str        = None,
    top_k:     int        = 5
):
    """
    Upload vehicle image → CLIP → Pinecone → visually similar vehicles
    """
    try:
        image_bytes = await file.read()
        result = search_by_image(
            image_bytes=image_bytes,
            tenant_id=tenant_id,
            max_price=max_price,
            city=city,
            top_k=top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/index/stats")
def index_stats(tenant_id: str = "spinny_india_prod"):
    """
    Returns vector count and index health per tenant
    """
    try:
        return get_index_stats(tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/health")
def v2_health():
    stats = get_index_stats("spinny_india_prod")
    return {
        "status":         "healthy",
        "vectors_indexed": stats.get("total_vectors", 0),
        "index":          os.getenv("PINECONE_INDEX_NAME_V2", "visionlens-index"),
    }