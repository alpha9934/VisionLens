import os
import streamlit as st
import requests

# ── 1. ENVIRONMENT & CONFIGURATION ─────────────────────────────────────────
if "API_BASE_URL" in st.secrets:
    API_BASE_URL = st.secrets["API_BASE_URL"]
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="VisionLens — Visual AI Search", page_icon="🔍", layout="wide")

# ── 2. HEADER REGION ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("ui/Spinny_Logo.jpg"):
        st.image("ui/Spinny_Logo.jpg", width=120)
with col2:
    st.title("🔍 VisionLens")
    st.markdown("### Visual AI Damage Detection & Pricing Engine")

st.divider()

# ── 3. SIDEBAR (SETTINGS & INDEX STATUS) ──────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    tenant_id = st.selectbox("Tenant", ["spinny_india_prod", "cars24_india_prod"])
    top_k = st.slider("Max Results", 1, 10, 5)
    st.divider()

    st.markdown("## 📊 Index Status")
    try:
        r = requests.get(f"{API_BASE_URL}/v2/index/stats?tenant_id={tenant_id}", timeout=10)
        stats = r.json() if r.status_code == 200 else {}
        st.metric("Vectors Indexed", stats.get("total_vectors", 0))
        st.metric("Dimension", stats.get("dimension", 512))
    except Exception:
        st.error("Backend Offline")
        
    if st.button("🔄 Refresh Data"):
        st.rerun()

# ── 4. TAB 1: TEXT SEARCH (With Callback Fix) ──────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔤 Text Search", "📷 Image Search", "🔧 Panel Analyzer"])

with tab1:
    st.subheader("Natural Language Vehicle Search")
    
    def set_query(val):
        st.session_state["main_search_input"] = val

    query = st.text_input("Search Query", key="main_search_input")

    col_ex, col_info = st.columns(2)
    with col_ex:
        for ex in ["car with dents on front bumper", "white SUV with scratches"]:
            st.button(ex, on_click=set_query, args=(ex,))

    if st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            r = requests.post(f"{API_BASE_URL}/v2/search/text", json={
                "query": query, "tenant_id": tenant_id, "top_k": top_k
            }, timeout=60)
            data = r.json()
            for res in data.get("results", []):
                st.write(f"Result: {res.get('vehicle_id')} - Score: {res.get('score')}")

# ══ Tab 2 — Image Search ════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload Image Search")
    uploaded = st.file_uploader("Upload image", type=["jpg", "png"])
    if st.button("📷 Search by Image") and uploaded:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        r = requests.post(f"{API_BASE_URL}/v2/search/image", files=files, params={"tenant_id": tenant_id})
        st.json(r.json())

# ══ Tab 3 — Panel Analyzer ══════════════════════════════════════════════════
with tab3:
    st.subheader("Panel Damage Assessment")
    model = st.text_input("Model")
    if st.button("🔧 Analyze"):
        r = requests.post(f"{API_BASE_URL}/v1/analyze", json={"model": model, "issues": "Dent-Major"})
        st.json(r.json())
