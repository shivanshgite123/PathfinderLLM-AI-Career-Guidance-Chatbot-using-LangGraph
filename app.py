"""
app.py  –  Streamlit Frontend
─────────────────────────────────────────────────────────────
Run with:  streamlit run app.py
─────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = os.getenv("API_BASE_URL", "http://localhost:8000")
PAGE_ICON = "🚀"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Career Guidance Bot",
    page_icon  = PAGE_ICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: white !important;
    border-radius: 8px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.5);
}

/* ── Report card ── */
.report-card {
    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    border: 1px solid #667eea40;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── Section headers ── */
.section-title {
    color: #667eea;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #667eea40;
}

/* ── Profile badge ── */
.profile-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}
.badge-beginner     { background: #e8f5e9; color: #2e7d32; }
.badge-intermediate { background: #fff3e0; color: #e65100; }
.badge-advanced     { background: #fce4ec; color: #880e4f; }

/* ── Chat bubble ── */
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.9rem;
}
.chat-bot {
    background: #f8f9ff;
    border: 1px solid #e0e0ff;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.9rem;
    color: #1a1a2e;
}

/* ── Follow-up chips ── */
.followup-chip {
    display: inline-block;
    background: #f0f0ff;
    border: 1px solid #667eea50;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    margin: 0.3rem;
    font-size: 0.82rem;
    color: #4a4a8a;
    cursor: pointer;
}

/* ── Status indicators ── */
.status-ok   { color: #4caf50; font-weight: 600; }
.status-err  { color: #f44336; font-weight: 600; }

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: white;
    border: 1px solid #e0e0ff;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(102,126,234,0.08);
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #667eea; }
.metric-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_profile" not in st.session_state:
    st.session_state.last_profile = None


# ── Utility Functions ─────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_api(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            json    = payload,
            timeout = 180,   # LLM generation can be slow
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The LLM may be slow – try again.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Is FastAPI running on port 8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def render_report(report: str) -> None:
    """Parse and render the structured report with styled section headers."""
    sections = {
        "--- USER ANALYSIS ---":       ("👤 User Analysis",         "section-title"),
        "--- CAREER OPTIONS ---":       ("💼 Career Options",        "section-title"),
        "--- RECOMMENDED PATH ---":     ("🎯 Recommended Path",      "section-title"),
        "--- STEP-BY-STEP ROADMAP ---": ("🗺️ Step-by-Step Roadmap", "section-title"),
        "--- SKILLS TO LEARN ---":      ("🛠️ Skills to Learn",      "section-title"),
        "--- 30-DAY ACTION PLAN ---":   ("📅 30-Day Action Plan",    "section-title"),
        "--- PROJECT IDEAS ---":        ("💡 Project Ideas",         "section-title"),
        "--- FINAL ADVICE ---":         ("✨ Final Advice",           "section-title"),
    }

    output = report
    for marker, (label, css_class) in sections.items():
        output = output.replace(
            marker,
            f'\n<div class="{css_class}">{label}</div>\n',
        )

    st.markdown(f'<div class="report-card">{output}</div>', unsafe_allow_html=True)


def render_profile_badges(profile: dict) -> None:
    level  = profile.get("level",  "beginner")
    domain = profile.get("domain", "Technology")
    goal   = profile.get("goal",   "")

    badge_class = {
        "beginner":     "badge-beginner",
        "intermediate": "badge-intermediate",
        "advanced":     "badge-advanced",
    }.get(level, "badge-beginner")

    st.markdown(
        f'<span class="profile-badge {badge_class}">📊 {level.upper()}</span>'
        f'<span class="profile-badge" style="background:#e8eaff;color:#3949ab;">🏷️ {domain}</span>',
        unsafe_allow_html=True,
    )
    if goal:
        st.caption(f"🎯 Goal: {goal}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚀 AI Career Bot")
    st.markdown("*Powered by LangGraph + Ollama*")
    st.divider()

    # ── API status ────────────────────────────────────────────────────────────
    api_ok = check_api_health()
    status_html = (
        '<span class="status-ok">● API Online</span>'
        if api_ok else
        '<span class="status-err">● API Offline</span>'
    )
    st.markdown(status_html, unsafe_allow_html=True)

    st.divider()

    # ── User Input Form ───────────────────────────────────────────────────────
    st.markdown("### 📝 Your Profile")

    education = st.text_area(
        "🎓 Education",
        placeholder="e.g. B.Sc. Computer Science, 2023",
        height=80,
        key="education",
    )
    skills = st.text_area(
        "🛠️ Current Skills",
        placeholder="e.g. Python, SQL, basic ML, Excel",
        height=80,
        key="skills",
    )
    interests = st.text_area(
        "💡 Interests",
        placeholder="e.g. data science, AI, product building",
        height=80,
        key="interests",
    )
    problem = st.text_area(
        "❓ Your Challenge",
        placeholder="e.g. I don't know what career path suits me after graduation",
        height=100,
        key="problem",
    )

    st.divider()

    # ── Endpoint selector ─────────────────────────────────────────────────────
    endpoint_choice = st.selectbox(
        "🎯 Focus",
        options   = ["/chat", "/roadmap", "/profile"],
        index     = 0,
        help      = "/chat = full guidance | /roadmap = roadmap focus | /profile = quick profile",
    )

    submit = st.button("🚀 Get Career Guidance", use_container_width=True)

    st.divider()

    # ── Utility buttons ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_report  = None
            st.session_state.last_profile = None
            st.rerun()
    with col2:
        if st.button("🔄 Rebuild\nIndex", use_container_width=True):
            r = requests.post(f"{API_BASE}/reset-index", timeout=30)
            st.toast("Index rebuilt!" if r.ok else "Failed to rebuild index.")

    st.divider()
    st.caption("Built with LangGraph · LangChain · Ollama · FAISS · FastAPI · Streamlit")


# ── Main Panel ────────────────────────────────────────────────────────────────

st.markdown("# 🧭 AI Career Guidance Bot")
st.markdown(
    "*Helping you find structure, direction, and a clear path after education.*"
)

# ── Handle form submission ────────────────────────────────────────────────────
if submit:
    if not any([education.strip(), skills.strip(), interests.strip(), problem.strip()]):
        st.warning("⚠️ Please fill in at least one field before submitting.")
    elif not api_ok:
        st.error("🔌 Backend is not running. Start it with: `uvicorn main:app --reload`")
    else:
        payload = {
            "education": education or "Not specified",
            "skills":    skills    or "Not specified",
            "interests": interests or "Not specified",
            "problem":   problem   or "Looking for career direction",
        }

        # Show the user message in chat history
        user_msg = (
            f"**Education:** {payload['education']}  \n"
            f"**Skills:** {payload['skills']}  \n"
            f"**Interests:** {payload['interests']}  \n"
            f"**Problem:** {payload['problem']}"
        )
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.spinner("🤖 Analysing your profile and generating guidance… (this may take 30–90s)"):
            data = call_api(endpoint_choice, payload)

        if data:
            if endpoint_choice == "/profile":
                # Profile-only response
                profile_text = (
                    f"**Level:** {data.get('level', 'N/A')}  \n"
                    f"**Domain:** {data.get('domain', 'N/A')}  \n"
                    f"**Goal:** {data.get('goal', 'N/A')}"
                )
                st.session_state.chat_history.append({"role": "bot", "content": profile_text})
                st.session_state.last_profile = data
            else:
                st.session_state.last_report  = data.get("report",  "")
                st.session_state.last_profile = data.get("profile", {})
                st.session_state.chat_history.append({
                    "role":    "bot",
                    "content": "✅ Career guidance report generated! See the full report below.",
                    "meta": {
                        "intent":   data.get("intent",   ""),
                        "decision": data.get("decision", ""),
                    },
                })
                # Store follow-ups
                st.session_state["last_followups"] = data.get("followup_questions", "")

        st.rerun()


# ── Chat History ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            meta = msg.get("meta", {})
            label = ""
            if meta:
                label = (
                    f' <small style="opacity:0.6;">'
                    f'intent: {meta.get("intent","")} | '
                    f'strategy: {meta.get("decision","")}</small>'
                )
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}{label}</div>',
                unsafe_allow_html=True,
            )


# ── Full Report ───────────────────────────────────────────────────────────────
if st.session_state.last_report:
    st.markdown("---")

    # ── Profile summary row ───────────────────────────────────────────────────
    if st.session_state.last_profile:
        st.markdown("### 👤 Your Profile")
        render_profile_badges(st.session_state.last_profile)

    st.markdown("### 📋 Your Career Guidance Report")
    render_report(st.session_state.last_report)

    # ── Follow-up questions ───────────────────────────────────────────────────
    followups = st.session_state.get("last_followups", "")
    if followups:
        st.markdown("---")
        st.markdown("### 🙋 Follow-up Questions to Explore")
        st.info(followups)

    # ── Download button ───────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label     = "⬇️ Download Report as .txt",
        data      = st.session_state.last_report,
        file_name = "career_guidance_report.txt",
        mime      = "text/plain",
    )


# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.chat_history and not st.session_state.last_report:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value">🧠</div>
          <div class="metric-label">AI-Powered Analysis</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value">🗺️</div>
          <div class="metric-label">Step-by-Step Roadmap</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value">🔒</div>
          <div class="metric-label">100% Local & Private</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 👋 Welcome! Here's how to get started:

    1. **Fill in your profile** in the sidebar on the left
    2. **Choose a focus** (Full Guidance / Roadmap / Profile)
    3. **Click "Get Career Guidance"** and wait ~30–90 seconds
    4. **Explore your personalised report** with roadmap, skills, and 30-day plan

    > 💡 **Tip:** The more detail you provide, the better your guidance will be!

    ---
    #### 📌 Example Query
    - **Education:** B.Sc. Computer Science, 2023
    - **Skills:** Python, SQL, basic statistics
    - **Interests:** Machine learning, data visualisation, building AI tools
    - **Problem:** I graduated 6 months ago but don't know which data role to target
    """)
