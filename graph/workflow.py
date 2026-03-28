"""


Defines the LangGraph StateGraph workflow:

  START
    ↓
  input_node        validate / normalise user inputs
    ↓
  intent_node       classify user intent
    ↓
  profile_node      build JSON user profile
    ↓
  rag_node           retrieve career knowledge from FAISS
    ↓
  decision_node      pick response strategy
    ↓
  output_node       generate full structured report
    ↓
  END

"""

from typing import Any, Dict, TypedDict, Optional

from langgraph.graph import StateGraph, END

from graph.nodes import (
    input_node,
    intent_node,
    profile_node,
    rag_node,
    decision_node,
    output_node,
)


#  State Schema 
# TypedDict gives LangGraph type info, all fields are optional so nodes
# update incrementally.

class CareerState(TypedDict, total=False):
    # Inputs
    education:          str
    skills:             str
    interests:          str
    problem:            str
    # Intermediate
    user_message:       str
    intent:             str
    profile:            Dict[str, str]
    context:            str
    decision:           str
    # Outputs
    report:             str
    followup_questions: str


#  Graph Construction 

def build_graph() -> Any:
    """
    Compile and return the LangGraph workflow.

    Returns:
        A compiled LangGraph runnable (supports .invoke()).
    """
    graph = StateGraph(CareerState)

    # Register nodes
    graph.add_node("input",    input_node)
    graph.add_node("intent",   intent_node)
    graph.add_node("profile",  profile_node)
    graph.add_node("rag",      rag_node)
    graph.add_node("decision", decision_node)
    graph.add_node("output",   output_node)

    # Define edges (linear pipeline)
    graph.set_entry_point("input")
    graph.add_edge("input",    "intent")
    graph.add_edge("intent",   "profile")
    graph.add_edge("profile",  "rag")
    graph.add_edge("rag",      "decision")
    graph.add_edge("decision", "output")
    graph.add_edge("output",   END)

    return graph.compile()


# Convenience runner 

def run_career_guidance(
    education: str,
    skills:    str,
    interests: str,
    problem:   str,
) -> Dict[str, Any]:
    """
    End-to-end helper used by the FastAPI backend.

    Args:
        education:  User's educational background.
        skills:     Comma-separated current skills.
        interests:  Career interests / passions.
        problem:    The main career challenge or question.

    Returns:
        Final state dict with keys: report, followup_questions, profile, intent, decision.
    """
    workflow = build_graph()
    initial_state: CareerState = {
        "education": education,
        "skills":    skills,
        "interests": interests,
        "problem":   problem,
    }
    result = workflow.invoke(initial_state)
    return result
