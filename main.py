"""
FastAPI Backend

Endpoints:
  POST /chat       full guidance pipeline (main endpoint)
  POST /profile    user profiling only
  POST /roadmap    roadmap-focused guidance
  GET  /health     server working  check
  POST /reset-index  rebuild FAISS index after adding docs

"""

import sys
import os

# Ensure project root is on Python path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from graph.workflow import run_career_guidance
from graph.nodes import profile_node, DEFAULT_PROFILE
from rag.retriever import reset_index


#  App setup 

app = FastAPI(
    title        = "AI Career Guidance API",
    description  = "LangGraph + LangChain + Ollama powered career counsellor",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

# Allow Streamlit (any local origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


#  Request / Response schemas 

class UserInput(BaseModel):
    education: str  = Field(..., example="B.Sc. Computer Science, 2023")
    skills:    str  = Field(..., example="Python, SQL, basic machine learning")
    interests: str  = Field(..., example="Data science, AI, building products")
    problem:   str  = Field(..., example="I don't know which career path to take after graduation")


class ChatResponse(BaseModel):
    report:             str
    followup_questions: str
    profile:            dict
    intent:             str
    decision:           str


class ProfileResponse(BaseModel):
    level:  str
    domain: str
    goal:   str


#  Endpoints 
@app.get("/health", tags=["Utility"])
def health_check():
    """Quick liveness check."""
    return {"status": "ok", "service": "AI Career Guidance API"}


@app.post("/chat", response_model=ChatResponse, tags=["Career"])
def chat(user_input: UserInput):
    """
    Run the full LangGraph career guidance pipeline.

    Returns a structured report, follow-up questions, and profile metadata.
    """
    try:
        result = run_career_guidance(
            education = user_input.education,
            skills    = user_input.skills,
            interests = user_input.interests,
            problem   = user_input.problem,
        )
        return ChatResponse(
            report             = result.get("report", ""),
            followup_questions = result.get("followup_questions", ""),
            profile            = result.get("profile", DEFAULT_PROFILE),
            intent             = result.get("intent", "general"),
            decision           = result.get("decision", "roadmap"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/profile", response_model=ProfileResponse, tags=["Career"])
def get_profile(user_input: UserInput):
    """
    Run only the profiling node – returns level, domain, and goal without
    generating a full report (useful for quick profile checks).
    """
    try:
        state = {
            "education": user_input.education,
            "skills":    user_input.skills,
            "interests": user_input.interests,
            "problem":   user_input.problem,
        }
        updated_state = profile_node(state)
        profile       = updated_state.get("profile", DEFAULT_PROFILE)
        return ProfileResponse(**profile)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/roadmap", response_model=ChatResponse, tags=["Career"])
def get_roadmap(user_input: UserInput):
    """
    Roadmap-focused guidance endpoint.
    Overrides the problem field to emphasise roadmap generation.
    """
    try:
        # Inject roadmap intent hint into the problem statement
        enhanced_problem = f"[INTENT: roadmap] {user_input.problem}"
        result = run_career_guidance(
            education = user_input.education,
            skills    = user_input.skills,
            interests = user_input.interests,
            problem   = enhanced_problem,
        )
        return ChatResponse(
            report             = result.get("report", ""),
            followup_questions = result.get("followup_questions", ""),
            profile            = result.get("profile", DEFAULT_PROFILE),
            intent             = result.get("intent", "roadmap"),
            decision           = result.get("decision", "roadmap"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reset-index", tags=["Utility"])
def rebuild_index():
    """
    Delete and rebuild the FAISS vector index.
    Call this after adding new .txt / .pdf files to the data/ directory.
    """
    try:
        reset_index()
        return {"status": "Index reset successfully. Will rebuild on next query."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
