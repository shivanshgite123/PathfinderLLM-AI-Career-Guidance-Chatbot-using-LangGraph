"""
utils/prompts.py
─────────────────────────────────────────────────────────────
Centralised prompt templates for every LangGraph node.
All templates are plain Python f-strings / PromptTemplate
objects so they are easy to tweak without touching logic code.
─────────────────────────────────────────────────────────────
"""

from langchain.prompts import PromptTemplate


# ── 1. Intent Detection ───────────────────────────────────────────────────────

INTENT_PROMPT = PromptTemplate(
    input_variables=["user_message"],
    template="""You are a career counselling assistant.
Classify the user's intent from their message.

User message: {user_message}

Respond with EXACTLY one of these labels (no explanation):
  roadmap         – user wants a step-by-step learning/career roadmap
  skills_plan     – user wants to know what skills to build
  career_options  – user wants to explore career paths or job options
  general         – anything else (greetings, vague questions, etc.)

Intent:""",
)


# ── 2. User Profiling ─────────────────────────────────────────────────────────

PROFILE_PROMPT = PromptTemplate(
    input_variables=["education", "skills", "interests", "problem"],
    template="""You are an expert career counsellor AI.
Analyse the candidate profile below and return a JSON object with exactly
three keys: "level", "domain", "goal".

Rules:
  • level  → "beginner" | "intermediate" | "advanced"
  • domain → the primary technical/professional domain (e.g. "Data Science",
             "Web Development", "Cloud Engineering", "Product Management")
  • goal   → a concise, actionable career goal sentence (max 20 words)

Candidate Profile:
  Education  : {education}
  Skills     : {skills}
  Interests  : {interests}
  Problem    : {problem}

Return ONLY valid JSON. Example:
{{"level": "beginner", "domain": "Data Science", "goal": "Land a junior data analyst role within 6 months"}}

JSON:""",
)


# ── 3. RAG-Augmented Career Guidance ─────────────────────────────────────────

CAREER_GUIDANCE_PROMPT = PromptTemplate(
    input_variables=[
        "education", "skills", "interests", "problem",
        "level", "domain", "goal",
        "intent", "context",
    ],
    template="""You are a world-class AI Career Coach.
Use the structured profile AND the retrieved career knowledge below to write
a comprehensive, actionable career guidance report.

══════════════════════════════════════════
CANDIDATE PROFILE
══════════════════════════════════════════
Education  : {education}
Skills     : {skills}
Interests  : {interests}
Problem    : {problem}
Level      : {level}
Domain     : {domain}
Goal       : {goal}
Intent     : {intent}

══════════════════════════════════════════
RETRIEVED CAREER KNOWLEDGE
══════════════════════════════════════════
{context}

══════════════════════════════════════════
INSTRUCTIONS
══════════════════════════════════════════
Write a detailed report using EXACTLY these section headers (keep the dashes):

--- USER ANALYSIS ---
Summarise the candidate's current situation, strengths, and gaps (3-5 sentences).

--- CAREER OPTIONS ---
List 3-5 specific job titles / career paths that fit this profile.
For each: title | why it fits | average salary range.

--- RECOMMENDED PATH ---
State the single best career path and explain why (2-3 sentences).

--- STEP-BY-STEP ROADMAP ---
Numbered list of 6-10 concrete milestones to reach the goal.

--- SKILLS TO LEARN ---
Bullet list: technical skills | why needed | free resource to learn it.

--- 30-DAY ACTION PLAN ---
Week 1 / Week 2 / Week 3 / Week 4 – specific daily/weekly tasks.

--- PROJECT IDEAS ---
3-5 hands-on projects to build a portfolio (name + one-line description).

--- FINAL ADVICE ---
2-3 sentences of motivating, personalised encouragement.

Write clearly, use plain language, and be specific to this candidate's profile.
""",
)


# ── 4. Follow-up Question Generator ──────────────────────────────────────────

FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=["domain", "goal", "report_summary"],
    template="""You are an AI career coach having a conversation with a student.
Based on the guidance already given, generate 3 smart follow-up questions
that would deepen their career planning.

Domain        : {domain}
Goal          : {goal}
Report summary: {report_summary}

Return ONLY a numbered list of 3 questions. No preamble.

1.""",
)
