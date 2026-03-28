# 🚀 AI Career Guidance Chatbot

> *Helping people find structure, mentorship, and career direction after formal education.*

A **production-ready, fully local** AI career counsellor built with:

| Layer | Technology |
|---|---|
| LLM | **Ollama** (Llama3 / Mistral) – runs 100% locally |
| Orchestration | **LangChain** |
| Workflow | **LangGraph** (stateful multi-node pipeline) |
| Vector DB / RAG | **FAISS** + Ollama Embeddings |
| Backend API | **FastAPI** |
| Frontend | **Streamlit** |

---

## 📁 Project Structure

```
ai_career_bot/
│
├── app.py                  ← Streamlit frontend (chat UI)
├── main.py                 ← FastAPI backend (REST API)
│
├── graph/
│   ├── __init__.py
│   ├── workflow.py         ← LangGraph StateGraph definition
│   └── nodes.py            ← All node functions (intent, profile, RAG, output)
│
├── rag/
│   ├── __init__.py
│   ├── loader.py           ← Document loader + text splitter
│   └── retriever.py        ← FAISS vector store builder & retriever
│
├── models/
│   ├── __init__.py
│   └── llm.py              ← Ollama LLM + Embeddings wrappers
│
├── utils/
│   ├── __init__.py
│   └── prompts.py          ← All LangChain PromptTemplates
│
├── data/
│   ├── career_knowledge.txt          ← Sample career knowledge base
│   ├── emerging_tech_careers.txt     ← Extended knowledge base
│   └── faiss_index/        ← Auto-created on first run
│
├── requirements.txt
└── README.md
```

---

## 🧠 LangGraph Workflow

```
START
  ↓
input_node      → Validate & normalise user inputs
  ↓
intent_node     → Classify intent: roadmap | skills_plan | career_options | general
  ↓
profile_node    → Build JSON: { "level": ..., "domain": ..., "goal": ... }
  ↓
rag_node        → Semantic search in FAISS vector store
  ↓
decision_node   → Map intent → response strategy
  ↓
output_node     → Generate full structured career report + follow-up questions
  ↓
END
```

---

## ▶️ Step-by-Step Setup

### 1. Prerequisites

- Python 3.10 or 3.11
- [Ollama](https://ollama.com/download) installed and running

### 2. Install Ollama & Pull Models

```bash
# Install Ollama (macOS)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM (choose one)
ollama pull llama3          # Recommended (~4.7GB)
ollama pull mistral         # Alternative (~4.1GB)

# Pull the embedding model (required for FAISS)
ollama pull nomic-embed-text   # Fast 768-dim embeddings (~274MB)

# Verify Ollama is running
ollama serve                   # Should output: Listening on 0.0.0.0:11434
```

### 3. Clone / Navigate to Project

```bash
cd ai_career_bot
```

### 4. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Start the FastAPI Backend

```bash
# From the ai_career_bot/ directory
uvicorn main:app --reload --port 8000
```

Open your browser at **http://localhost:8000/docs** to see the Swagger UI.

### 6. Start the Streamlit Frontend

```bash
# In a NEW terminal (keep FastAPI running)
source venv/bin/activate
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🖥️ Using the App

1. Fill in your **profile** in the left sidebar:
   - 🎓 Education (e.g. "B.Sc. Computer Science, 2023")
   - 🛠️ Skills (e.g. "Python, SQL, basic machine learning")
   - 💡 Interests (e.g. "AI, data science, building products")
   - ❓ Problem (e.g. "I don't know which career path to take")

2. Choose a **focus**:
   - `/chat` – Full career guidance report
   - `/roadmap` – Roadmap-emphasised guidance
   - `/profile` – Quick profile classification only

3. Click **"🚀 Get Career Guidance"** and wait ~30–90 seconds

4. **Download** your report as a `.txt` file

---

## 📌 Example Query

| Field | Example |
|---|---|
| Education | B.Sc. Computer Science, graduated 2023 |
| Skills | Python, SQL, pandas, basic machine learning, Git |
| Interests | Data science, AI engineering, LLMs, building tools |
| Problem | I've been job hunting for 6 months with no offers. I don't know if I should specialise in ML or data analysis or something else entirely. |

**Expected output sections:**
- `--- USER ANALYSIS ---`
- `--- CAREER OPTIONS ---`
- `--- RECOMMENDED PATH ---`
- `--- STEP-BY-STEP ROADMAP ---`
- `--- SKILLS TO LEARN ---`
- `--- 30-DAY ACTION PLAN ---`
- `--- PROJECT IDEAS ---`
- `--- FINAL ADVICE ---`

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Full career guidance pipeline |
| `POST` | `/profile` | User profiling only (fast) |
| `POST` | `/roadmap` | Roadmap-focused guidance |
| `POST` | `/reset-index` | Rebuild FAISS index after adding docs |

### Example API call (curl)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "education": "B.Sc. Computer Science, 2023",
    "skills": "Python, SQL, basic ML",
    "interests": "Data science, AI",
    "problem": "I need a career roadmap after graduation"
  }'
```

---

## 📂 Adding Your Own Knowledge Base

1. Drop `.txt` or `.pdf` files into the `data/` directory
2. Call the reset endpoint to rebuild the FAISS index:
   ```bash
   curl -X POST http://localhost:8000/reset-index
   ```
3. Or click **"🔄 Rebuild Index"** in the Streamlit sidebar

---

## 🔧 Configuration

Set environment variables to customise behaviour:

```bash
# .env (or export in shell)
OLLAMA_MODEL=llama3           # or mistral, llama3:instruct, etc.
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text  # embedding model for FAISS
API_BASE_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` on port 11434 | Run `ollama serve` in a terminal |
| `model not found` error | Run `ollama pull llama3` |
| Slow response (>2 min) | Normal for first query; FAISS index is being built |
| `Cannot connect to backend` in Streamlit | Start FastAPI first: `uvicorn main:app --reload` |
| Import errors | Ensure you're in the venv and ran `pip install -r requirements.txt` |

---

## 🏗️ Architecture Notes

- **LangGraph** manages state across nodes — each node receives the full state dict and returns a partial update
- **FAISS** index is persisted to `data/faiss_index/` on first build and loaded from disk on restarts
- **Graceful degradation**: if RAG retrieval fails, the LLM still generates from its training data
- **Built-in knowledge**: if no files exist in `data/`, the loader uses a hardcoded career knowledge base

---

## 📄 License

MIT License – free to use, modify, and distribute.
