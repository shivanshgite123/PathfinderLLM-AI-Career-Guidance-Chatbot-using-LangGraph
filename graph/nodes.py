"""


Each function LangGraph node: it receives the full
graph state dict, does one focused task, and returns a partial
state update.

Node execution order (defined in workflow.py):
  input_node → intent_node → profile_node → rag_node
  → decision_node → output_node

"""

import json
import re
from typing import Any, Dict

from models.llm import get_llm
from rag.retriever import retrieve
from utils.prompts import (
    INTENT_PROMPT,
    PROFILE_PROMPT,
    CAREER_GUIDANCE_PROMPT,
    FOLLOWUP_PROMPT,
)


# Helpers


def _invoke_llm(prompt_template, **kwargs) -> str:
    """Format a PromptTemplate and call the LLM. Returns stripped text."""
    llm    = get_llm()
    chain  = prompt_template | llm
    result = chain.invoke(kwargs)
    # OllamaLLM returns a string directly; handle both str and AIMessage
    return str(result).strip() if result else ""



# Node 1 – Input Validation

def input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the four required user inputs are present.
    Provides friendly defaults if any field is missing.
    """
    required = ["education", "skills", "interests", "problem"]
    defaults = {
        "education":  "Not specified",
        "skills":     "Not specified",
        "interests":  "Not specified",
        "problem":    "Looking for career direction",
    }

    for field in required:
        if not state.get(field, "").strip():
            state[field] = defaults[field]

    # Compose a combined message for intent detection
    state["user_message"] = (
        f"Education: {state['education']}. "
        f"Skills: {state['skills']}. "
        f"Interests: {state['interests']}. "
        f"Problem: {state['problem']}."
    )

    print(f"[input_node] User message ready.")
    return state



# Node 2 – Intent Detection


VALID_INTENTS = {"roadmap", "skills_plan", "career_options", "general"}

def intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify what the user primarily wants.
    Falls back to 'general' on any parse error.
    """
    try:
        raw    = _invoke_llm(INTENT_PROMPT, user_message=state["user_message"])
        intent = raw.strip().lower().split()[0]          # first word only
        intent = intent if intent in VALID_INTENTS else "general"
    except Exception as exc:
        print(f"[intent_node] WARNING: {exc}")
        intent = "general"

    state["intent"] = intent
    print(f"[intent_node] Detected intent: {intent}")
    return state



# Node 3 – User Profiling


DEFAULT_PROFILE = {"level": "beginner", "domain": "Technology", "goal": "Build a fulfilling career"}

def profile_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to produce a structured JSON profile.
    Robust JSON extraction handles markdown code fences.
    """
    try:
        raw = _invoke_llm(
            PROFILE_PROMPT,
            education=state["education"],
            skills=state["skills"],
            interests=state["interests"],
            problem=state["problem"],
        )

        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Extract the first {...} block
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        profile = json.loads(match.group()) if match else DEFAULT_PROFILE

        # Validate keys
        for key in DEFAULT_PROFILE:
            if key not in profile or not profile[key]:
                profile[key] = DEFAULT_PROFILE[key]

    except Exception as exc:
        print(f"[profile_node] WARNING – using default profile: {exc}")
        profile = DEFAULT_PROFILE.copy()

    state["profile"] = profile
    print(f"[profile_node] Profile: {profile}")
    return state



# Node 4 – RAG Retrieval


def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a semantic search query from the user profile and retrieve
    relevant career knowledge from the FAISS vector store.
    """
    profile = state.get("profile", DEFAULT_PROFILE)
    query   = (
        f"{profile.get('domain', '')} career path "
        f"{profile.get('level', '')} "
        f"{profile.get('goal', '')} "
        f"{state.get('interests', '')}"
    )

    context = retrieve(query, k=4)
    state["context"] = context
    print(f"[rag_node] Retrieved {len(context)} characters of context.")
    return state



# Node 5 – Decision Node


def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the detected intent to a response strategy tag.
    The output_node uses this to emphasise different report sections.
    """
    mapping = {
        "roadmap":        "roadmap",
        "skills_plan":    "skills_plan",
        "career_options": "career_options",
        "general":        "roadmap",   # default to roadmap for vague queries
    }
    state["decision"] = mapping.get(state.get("intent", "general"), "roadmap")
    print(f"[decision_node] Decision strategy: {state['decision']}")
    return state



# Node 6 – Response Generation


def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the full structured career guidance report and follow-up questions.
    This is the most compute-intensive node.
    """
    profile = state.get("profile", DEFAULT_PROFILE)

    #  Main report
    try:
        report = _invoke_llm(
            CAREER_GUIDANCE_PROMPT,
            education  = state.get("education",  "Not specified"),
            skills     = state.get("skills",     "Not specified"),
            interests  = state.get("interests",  "Not specified"),
            problem    = state.get("problem",    "Not specified"),
            level      = profile.get("level",    "beginner"),
            domain     = profile.get("domain",   "Technology"),
            goal       = profile.get("goal",     "Build a career"),
            intent     = state.get("intent",     "general"),
            context    = state.get("context",    ""),
        )
    except Exception as exc:
        print(f"[output_node] ERROR generating report: {exc}")
        report = f"Error generating report: {exc}\n\nPlease ensure Ollama is running."

    state["report"] = report

    #  Follow-up questions
    try:
        # Use the first 300 chars of the report as a summary for the follow-up prompt
        summary = report[:300].replace("\n", " ")
        followups_raw = _invoke_llm(
            FOLLOWUP_PROMPT,
            domain         = profile.get("domain", "Technology"),
            goal           = profile.get("goal",   "Build a career"),
            report_summary = summary,
        )
        # Ensure it starts with "1." (LLM sometimes includes preamble)
        followups = "1." + followups_raw.split("1.")[-1] if "1." in followups_raw else followups_raw
    except Exception as exc:
        print(f"[output_node] WARNING generating follow-ups: {exc}")
        followups = "1. What specific skills should I focus on first?\n2. How long will this roadmap take?\n3. What certifications are most valuable?"

    state["followup_questions"] = followups
    print("[output_node] Report and follow-up questions generated.")
    return state
