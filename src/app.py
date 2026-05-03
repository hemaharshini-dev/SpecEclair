"""
Streamlit UI for BIS Standards Recommendation Engine.
Run: streamlit run src/app.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import src.pipeline as pipeline

st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Performance Caching ──
@st.cache_resource
def get_pipeline():
    """
    Caches the pipeline resources (embedding model, index, LLM)
    so they don't reload on every interaction.
    """
    # Trigger first-time load of retriever and llm
    from src.retriever import _load_index
    _load_index()
    pipeline._get_llm()
    return pipeline.run

# ── Session State ──
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "run_search" not in st.session_state:
    st.session_state.run_search = False

def clear_search():
    st.session_state.query_text = ""
    st.session_state.run_search = False

def set_example(query):
    st.session_state.query_text = query
    st.session_state.run_search = True

# ── Custom CSS for minor tweaks ──
st.markdown("""
<style>
    /* Make metric labels less prominent */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #6b7a99 !important;
    }
    /* Hide the top padding of main block for a tighter layout */
    .block-container {
        padding-top: 2rem !important;
    }
    /* Style subheaders inside result cards */
    .result-header h3 {
        color: #1a3c5e !important;
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.title("🏗️ BIS Finder")
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "This tool helps Indian Micro & Small Enterprises (MSEs) instantly find "
        "applicable Bureau of Indian Standards for their building material products."
    )
    st.markdown(
        "**How it works**\n"
        "1. Enter a product description\n"
        "2. Search 559 BIS standards\n"
        "3. Get AI-powered rationale"
    )
    st.markdown("---")
    st.caption(
        "**Data Source:** BIS SP 21 (2005)\n\n"
        "**Embeddings:** all-MiniLM-L6-v2\n\n"
        "**LLM:** Groq llama-3.3-70b"
    )

# ── Main Content ──
st.title("🏗️ BIS Standards Recommendation Engine")
st.markdown("Instantly find applicable Bureau of Indian Standards (BIS) for your building material product using AI.")

st.markdown("---")

# ── Example Queries ──
st.markdown("##### ✨ Not sure what to search? Try an example:")
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("🧱 Portland Cement", use_container_width=True):
        set_example("We manufacture 33 Grade Ordinary Portland Cement and need to know which BIS standard covers the chemical and physical requirements.")
with ex2:
    if st.button("🏗️ Steel Rebars", use_container_width=True):
        set_example("High strength deformed steel bars and wires for concrete reinforcement.")
with ex3:
    if st.button("🏠 Clay Bricks", use_container_width=True):
        set_example("Common burnt clay building bricks used in masonry work.")

st.write("") # spacer

# ── Search Area ──
with st.container(border=True):
    st.markdown("##### 🔍 Describe your product or compliance need")
    
    query = st.text_area(
        "Product Description",
        placeholder="e.g., We are a small enterprise manufacturing Portland Pozzolana Cement using fly ash. Which BIS standard applies?",
        height=120,
        key="query_text",
        label_visibility="collapsed"
    )

    col1, col2, _ = st.columns([2, 1, 7])
    with col1:
        search_clicked = st.button("Find Relevant Standards", type="primary", use_container_width=True)
    with col2:
        st.button("Clear", on_click=clear_search, use_container_width=True)

if search_clicked:
    st.session_state.run_search = True

# ── Results Area ──
if st.session_state.run_search:
    if not st.session_state.query_text.strip():
        st.warning("⚠️ Please enter a product description to search.")
        st.session_state.run_search = False
    else:
        with st.spinner("Searching 559 BIS standards..."):
            t0 = time.time()
            try:
                # Use the cached pipeline function
                run_pipeline = get_pipeline()
                results = run_pipeline(st.session_state.query_text.strip(), top_k=5)
                latency = round(time.time() - t0, 2)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                results = []
                latency = 0

        if results:
            st.markdown("---")
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Standards Found", value=len(results))
            with m2:
                st.metric(label="Response Time", value=f"{latency}s")
            with m3:
                st.metric(label="Status", value="Optimized" if latency < 5 else "High Latency")
            
            st.markdown("### 📋 Recommended Standards")
            
            for i, r in enumerate(results):
                with st.container(border=True):
                    header_col, score_col = st.columns([4, 1])
                    with header_col:
                        st.markdown(f"<div class='result-header'><h3>{r['standard_id']}</h3></div>", unsafe_allow_html=True)
                        st.markdown(f"**{r.get('title', 'BIS Standard')}**")
                    with score_col:
                        score_pct = round(r['score'] * 100, 1)
                        st.metric(label="Relevance", value=f"{score_pct}%")
                    
                    st.info(f"**Why it applies:** {r.get('rationale', 'Retrieved based on semantic similarity to your query.')}")
