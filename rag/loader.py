"""
rag/loader.py

Loads documents from the data/ directory and splits them into
chunks suitable for embedding.

Supported formats:
   .txt   plain text career guides
   .pdf   PDF career documents

"""

import os
import glob
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# Configuration 
DATA_DIR: str  = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNK_SIZE: int    = 800   # characters per chunk
CHUNK_OVERLAP: int = 150   # overlap to preserve context across chunks


def _load_txt_files(data_dir: str) -> List[Document]:
    """Load all .txt files from data_dir."""
    docs: List[Document] = []
    for path in glob.glob(os.path.join(data_dir, "*.txt")):
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
            print(f"  [loader] Loaded TXT: {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [loader] WARNING – could not load {path}: {exc}")
    return docs


def _load_pdf_files(data_dir: str) -> List[Document]:
    """Load all .pdf files from data_dir."""
    docs: List[Document] = []
    for path in glob.glob(os.path.join(data_dir, "*.pdf")):
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(f"  [loader] Loaded PDF: {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [loader] WARNING – could not load {path}: {exc}")
    return docs


def load_documents() -> List[Document]:
    """
    Load every supported document from the data/ directory.

    Returns:
        List of LangChain Document objects (chunked & ready for embedding).
    """
    abs_data_dir = os.path.abspath(DATA_DIR)
    print(f"[loader] Scanning data directory: {abs_data_dir}")

    raw_docs: List[Document] = []
    raw_docs.extend(_load_txt_files(abs_data_dir))
    raw_docs.extend(_load_pdf_files(abs_data_dir))

    if not raw_docs:
        # Always return at least the built-in fallback knowledge
        print("[loader] No external docs found – using built-in knowledge base.")
        raw_docs = _get_builtin_knowledge()

    # ── Split into chunks ─────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[loader] Total chunks ready for embedding: {len(chunks)}")
    return chunks


def _get_builtin_knowledge() -> List[Document]:
    """
    Hardcoded career knowledge used when no files exist in data/.
    This ensures the RAG pipeline always has something to retrieve from.
    """
    knowledge_base = [
        """
        DATA SCIENCE CAREER PATH
        Data Science is one of the most in-demand fields. Entry-level roles include Data Analyst,
        Junior Data Scientist, and ML Engineer. Key skills: Python, SQL, statistics, machine learning,
        data visualisation. Top tools: pandas, scikit-learn, TensorFlow, Tableau, Power BI.
        Certifications: Google Data Analytics, IBM Data Science, AWS ML Specialty.
        Salary range: $60k–$150k depending on experience and location.
        Career progression: Data Analyst → Data Scientist → Senior DS → ML Engineer → AI Lead.
        """,

        """
        WEB DEVELOPMENT CAREER PATH
        Web development splits into frontend, backend, and full-stack roles.
        Frontend: HTML, CSS, JavaScript, React, Vue, Angular.
        Backend: Node.js, Python (Django/FastAPI), Java (Spring), databases.
        Full-stack: combines both. DevOps knowledge (Docker, CI/CD) is a big plus.
        Salary range: $55k–$140k. Freelancing is highly viable.
        Portfolio projects are critical: build 3-5 real-world apps.
        """,

        """
        CLOUD & DEVOPS CAREER PATH
        Cloud computing is the backbone of modern tech. Key platforms: AWS, GCP, Azure.
        Core skills: Linux, networking, containers (Docker/Kubernetes), Infrastructure-as-Code (Terraform).
        Certifications: AWS Solutions Architect, CKA (Kubernetes), Azure Administrator.
        Roles: Cloud Engineer, DevOps Engineer, SRE (Site Reliability Engineer).
        Salary range: $80k–$180k. High demand globally.
        """,

        """
        PRODUCT MANAGEMENT CAREER PATH
        PMs bridge business and engineering. Skills: user research, roadmapping, agile/scrum,
        data analysis, stakeholder communication, A/B testing.
        Tools: Jira, Notion, Figma (basic), Amplitude, SQL (basic).
        Entry paths: transition from engineering, design, or business analysis.
        Certifications: AIPMM, Pragmatic Institute, Google PM Certificate.
        Salary range: $90k–$180k at senior levels.
        """,

        """
        CYBERSECURITY CAREER PATH
        Cybersecurity is critical and consistently understaffed. Roles include Security Analyst,
        Penetration Tester, SOC Analyst, Cloud Security Engineer.
        Core skills: networking, Linux, Python scripting, vulnerability assessment, SIEM tools.
        Certifications: CompTIA Security+, CEH, CISSP, OSCP (advanced).
        Entry point: CompTIA A+ → Network+ → Security+.
        Salary range: $70k–$160k. Bug bounty programmes offer additional income.
        """,

        """
        30-DAY BEGINNER ACTION PLAN (GENERIC)
        Week 1: Research your chosen domain. Set up learning accounts (Coursera, YouTube, GitHub).
                 Pick one beginner course and commit 1 hour/day.
        Week 2: Complete 50% of the course. Start a small personal project.
                 Join relevant Discord/Slack communities.
        Week 3: Finish the course. Document your project on GitHub.
                 Write a LinkedIn post about what you learned.
        Week 4: Apply for 5 internships or junior roles. Attend one virtual meetup.
                 Identify your next course/skill to build.
        """,

        """
        JOB SEARCH STRATEGIES
        Networking accounts for 70-80% of jobs filled. LinkedIn optimisation is essential:
        professional photo, keyword-rich headline, detailed experience section.
        Cold outreach: message 5 professionals per week in your target field.
        Resume tips: one page, quantified achievements, ATS-optimised keywords.
        Portfolio: GitHub for engineers, Behance for designers, case studies for PMs.
        Interview prep: LeetCode (engineers), mock interviews, STAR method for behavioural.
        """,

        """
        FREE LEARNING RESOURCES
        Programming & CS: freeCodeCamp, The Odin Project, CS50 (Harvard, free), MIT OpenCourseWare.
        Data Science & ML: fast.ai, Kaggle Learn, Google ML Crash Course.
        Cloud: AWS Free Tier + official docs, A Cloud Guru (free tier), YouTube channels.
        General: Coursera (audit for free), edX, Khan Academy, YouTube (CS Dojo, Fireship, Sentdex).
        Community: r/learnprogramming, Hashnode, Dev.to, Stack Overflow.
        """,

        """
        SOFT SKILLS FOR CAREER SUCCESS
        Communication: written and verbal clarity is valued more than pure technical skill at senior levels.
        Problem-solving: break large problems into smaller ones; document your thinking.
        Adaptability: technology evolves fast; commit to lifelong learning (1 hour/day minimum).
        Collaboration: open source contributions, pair programming, and code reviews build reputation.
        Self-management: use Notion/Trello for personal projects; treat your career like a product.
        """,
    ]

    return [
        Document(page_content=text.strip(), metadata={"source": "builtin_knowledge_base"})
        for text in knowledge_base
    ]
