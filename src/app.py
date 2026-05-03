"""
Streamlit UI for BIS Standards Recommendation Engine.
Run: streamlit run src/app.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.pipeline import run

st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page background ── */
.stApp {
    background: #f0f4f8;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a3c5e 0%, #2563a8 60%, #1e88e5 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    text-transform: uppercase;
}

/* ── Search card ── */
.search-card {
    background: white;
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 24px;
}
.search-card h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #1a3c5e;
    margin: 0 0 12px 0;
}

/* ── Textarea override ── */
textarea {
    border-radius: 10px !important;
    border: 1.5px solid #d1dce8 !important;
    font-size: 0.97rem !important;
    transition: border-color 0.2s;
}
textarea:focus {
    border-color: #2563a8 !important;
    box-shadow: 0 0 0 3px rgba(37,99,168,0.12) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3c5e, #2563a8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    background: white;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-left: 5px solid #2563a8;
    transition: box-shadow 0.2s;
}
.result-card:hover {
    box-shadow: 0 4px 20px rgba(37,99,168,0.13);
}
.result-card.rank-1 { border-left-color: #1a3c5e; }
.result-card.rank-2 { border-left-color: #2563a8; }
.result-card.rank-3 { border-left-color: #3b82f6; }
.result-card.rank-4 { border-left-color: #60a5fa; }
.result-card.rank-5 { border-left-color: #93c5fd; }

.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.result-id {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a3c5e;
}
.result-rank {
    background: #e8f0fb;
    color: #2563a8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.result-title {
    font-size: 0.92rem;
    color: #4b6080;
    font-style: italic;
    margin-bottom: 10px;
}
.result-rationale {
    font-size: 0.95rem;
    color: #2d3748;
    line-height: 1.6;
    background: #f7faff;
    border-radius: 8px;
    padding: 10px 14px;
}
.result-rationale span {
    font-weight: 600;
    color: #1a3c5e;
}
.score-pill {
    display: inline-block;
    background: #e8f0fb;
    color: #1a3c5e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.stat-chip {
    background: white;
    border-radius: 10px;
    padding: 10px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    font-size: 0.88rem;
    color: #4b6080;
    font-weight: 500;
}
.stat-chip b { color: #1a3c5e; }

/* ── Example hint ── */
.example-label {
    font-size: 0.83rem;
    color: #6b7a99;
    margin: 6px 0 14px 0;
    line-height: 1.5;
}
.example-label em {
    color: #2563a8;
    font-style: italic;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #8a9ab5;
    font-size: 0.82rem;
    padding: 24px 0 8px 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a3c5e !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #93c5fd !important;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ BIS Finder")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "This tool helps Indian Micro & Small Enterprises (MSEs) instantly find "
        "applicable Bureau of Indian Standards for their building material products."
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Enter your product description\n"
        "2. The engine searches 559 BIS standards\n"
        "3. Top matches are returned with rationale"
    )
    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown("BIS SP 21 (2005) — Summaries of Indian Standards for Building Materials")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("`all-MiniLM-L6-v2` (local embeddings)")
    st.markdown("`llama-3.3-70b` via Groq (rationale)")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🏆 BIS × Sigma Squad AI Hackathon</div>
    <h1>🏗️ BIS Standards Recommendation Engine</h1>
    <p>Instantly find applicable Bureau of Indian Standards for your building material product — powered by RAG + AI.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "run_search" not in st.session_state:
    st.session_state.run_search = False

# ── Search card ───────────────────────────────────────────────────────────────
st.markdown('<div class="search-card"><h3>Describe your product or compliance need</h3>', unsafe_allow_html=True)
st.markdown('<div class="example-label">Example: <em>We manufacture 33 Grade Ordinary Portland Cement and need to know which BIS standard covers the chemical and physical requirements for our product.</em></div>', unsafe_allow_html=True)

query = st.text_area(
    label="query_input",
    label_visibility="collapsed",
    value=st.session_state.query_text,
    placeholder="e.g. We are a small enterprise manufacturing Portland Pozzolana Cement using fly ash. Which BIS standard applies?",
    height=110,
)

col_btn, col_clear = st.columns([5, 1])
with col_btn:
    if st.button("🔍  Find Relevant Standards", type="primary", use_container_width=True):
        st.session_state.query_text = query
        st.session_state.run_search = True
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.query_text = ""
        st.session_state.run_search = False

st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.run_search:
    if not query.strip():
        st.warning("⚠️ Please enter a product description before searching.")
    else:
        with st.spinner("Searching 559 BIS standards..."):
            t0 = time.time()
            try:
                results = run(query.strip(), top_k=5)
                latency = round(time.time() - t0, 2)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                results = []
                latency = 0

        if results:
            # Stats bar
            st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-chip">📋 <b>{len(results)}</b> standards found</div>
                <div class="stat-chip">⚡ <b>{latency}s</b> response time</div>
                <div class="stat-chip">🗂️ Searched <b>559</b> BIS standards</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Recommended Standards")

            rank_colors = ["rank-1", "rank-2", "rank-3", "rank-4", "rank-5"]
            rank_labels = ["#1 Best Match", "#2", "#3", "#4", "#5"]

            for i, r in enumerate(results):
                rank_class = rank_colors[i] if i < len(rank_colors) else "rank-5"
                rank_label = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
                title = r["title"] if r["title"] else "BIS Standard"
                rationale = r["rationale"] if r["rationale"] else "Retrieved based on semantic similarity to your query."
                score_pct = round(r["score"] * 100, 1)

                st.markdown(f"""
                <div class="result-card {rank_class}">
                    <div class="result-header">
                        <div class="result-id">{r['standard_id']}</div>
                        <div class="result-rank">{rank_label}</div>
                    </div>
                    <div class="result-title">{title}</div>
                    <div class="result-rationale">
                        <span>Why it applies: </span>{rationale}
                    </div>
                    <div class="score-pill">Relevance: {score_pct}%</div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Data source: BIS SP 21 (2005) — Summaries of Indian Standards for Building Materials &nbsp;|&nbsp;
    Embeddings: all-MiniLM-L6-v2 (local) &nbsp;|&nbsp; LLM: Groq llama-3.3-70b-versatile
</div>
""", unsafe_allow_html=True)
