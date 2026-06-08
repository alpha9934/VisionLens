import os
import streamlit as st
import requests
import pandas as pd

#  FIX 1: Correctly check Streamlit Secrets first, then fall back to local environment
if "API_BASE_URL" in st.secrets:
    API_BASE_URL = st.secrets["API_BASE_URL"]
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="VisionLens — Visual AI Search",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.image("ui/Spinny_Logo.jpg", width=120)
with col2:
    st.title("🔍 VisionLens")
    st.markdown("### Visual AI Damage Detection & Pricing Engine")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    tenant_id = st.selectbox("Tenant", ["spinny_india_prod", "cars24_india_prod"])
    top_k = st.slider("Max Results", 1, 10, 5)
    st.divider()

    st.markdown("## 📊 Index Status")
    
    #  FIX 2: Pull API data cleanly and show metrics outside the conditional 
    # so they stay permanently visible on the screen
    try:
        r = requests.get(f"{API_BASE_URL}/v2/index/stats?tenant_id={tenant_id}", timeout=10)
        if r.status_code == 200:
            stats = r.json()
            total_vectors = stats.get("total_vectors", 0)
            dimension = stats.get("dimension", 512)
        else:
            total_vectors, dimension = "Error", "Error"
    except Exception:
        total_vectors, dimension = "Offline", "Offline"

    st.metric("Vectors Indexed", total_vectors)
    st.metric("Dimension", dimension)
    
    if st.button("🔄 Force Refresh"):
        st.rerun()
        
    st.divider()
    st.markdown("**Model:** CLIP ViT-B/32")
    st.markdown("**LLM:** LLaMA 3.1 8B (Groq)")
    st.markdown("**Vector DB:** Pinecone Serverless")
    st.markdown("**Dataset:** 48,635 Spinny panels")
