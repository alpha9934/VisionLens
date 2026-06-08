# ui/visionlens_app.py

import os

import streamlit as st
import requests
import pandas as pd




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
    if st.button("Refresh Stats"):
        r = requests.get(f"{API_BASE_URL}/v2/index/stats?tenant_id={tenant_id}")
        stats = r.json()
        st.metric("Vectors Indexed", stats.get("total_vectors", 0))
        st.metric("Dimension", stats.get("dimension", 512))
    st.divider()
    st.markdown("**Model:** CLIP ViT-B/32")
    st.markdown("**LLM:** LLaMA 3.1 8B (Groq)")
    st.markdown("**Vector DB:** Pinecone Serverless")
    st.markdown("**Dataset:** 48,635 Spinny panels")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔤 Text Search",
    "📷 Image Search",
    "🔧 Panel Analyzer"
])

# ══ Tab 1 — Text Search ══════════════════════════════════════════
with tab1:
    st.subheader("Natural Language Vehicle Search")
    st.caption("Search using plain English — VisionLens understands damage descriptions, price filters, and city preferences.")

    query = st.text_input(
        "Search Query",
        placeholder="e.g. grey sedan with dents under 8 lakhs in Bengaluru"
    )

    col1, col2 = st.columns(2)
    with col1:
        examples = [
            "car with dents on front bumper",
            "white SUV with scratches",
            "vehicle with cosmetic damage only",
            "car with major accident damage",
        ]
        st.markdown("**Quick Examples:**")
        for ex in examples:
            if st.button(ex, key=ex):
                query = ex

    with col2:
        st.markdown("**How it works:**")
        st.info(
            "1. Query parsed by GPT-4o-mini\n"
            "2. Visual description → CLIP embedding\n"
            "3. Pinecone semantic search\n"
            "4. Results enriched with damage scores"
        )

    if st.button("🔍 Search", type="primary", use_container_width=True, key="text_search"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Running visual search..."):
                try:
                    r = requests.post(f"{API_BASE_URL}/v2/search/text", json={
                        "query": query,
                        "tenant_id": tenant_id,
                        "top_k": top_k
                    }, timeout=60)
                    data = r.json()

                    if "error" in data:
                        st.error(data["error"])
                    else:
                        parsed = data.get("parsed", {})
                        st.divider()

                        # Parsed query info
                        p1, p2, p3 = st.columns(3)
                        p1.info(f"**Visual:** {parsed.get('visual_description', query)}")
                        p2.info(f"**Filters:** {parsed.get('filters', {})}")
                        p3.info(f"**Intent:** {parsed.get('search_intent', 'general')}")

                        st.markdown(f"### Results ({data.get('total_results', 0)} found)")

                        for i, result in enumerate(data.get("results", [])):
                            with st.expander(
                                f"#{i+1} — Vehicle {result.get('vehicle_id')} | "
                                f"{result.get('panel')} | "
                                f"Score: {result.get('score')} | "
                                f"{result.get('cost_bucket')}",
                                expanded=i == 0
                            ):
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    img_url = result.get("image_url", "")
                                    if img_url:
                                        st.image(img_url, use_container_width=True)

                                with c2:
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Similarity", result.get("score"))
                                    m2.metric("Damage Score", result.get("damage_score"))
                                    m3.metric("Cost Bucket", result.get("cost_bucket"))

                                    narrative = result.get("narrative", {})
                                    if narrative:
                                        st.markdown("**AI Summary:**")
                                        st.info(narrative.get("summary", "N/A"))
                                        st.markdown("**Recommended Action:**")
                                        st.success(narrative.get("recommended_action", "N/A"))

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

# ══ Tab 2 — Image Search ═════════════════════════════════════════
with tab2:
    st.subheader("Upload Image → Find Similar Damaged Vehicles")
    st.caption("Upload a photo of vehicle damage — VisionLens finds visually similar panels from Spinny's fleet.")

    uploaded = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    col1, col2 = st.columns(2)
    with col1:
        max_price = st.number_input("Max Price (₹)", min_value=0, value=0, step=50000)
        max_price = max_price if max_price > 0 else None
    with col2:
        city_filter = st.text_input("City Filter", placeholder="e.g. Bengaluru")
        city_filter = city_filter if city_filter else None

    if st.button("📷 Search by Image", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Encoding image with CLIP..."):
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(uploaded, caption="Your uploaded image", use_container_width=True)

                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    params = {"tenant_id": tenant_id, "top_k": top_k}
                    if max_price:
                        params["max_price"] = max_price
                    if city_filter:
                        params["city"] = city_filter

                    r = requests.post(
                        f"{API_BASE_URL}/v2/search/image",
                        files=files,
                        params=params,
                        timeout=60
                    )
                    data = r.json()

                    st.markdown(f"### Similar Vehicles ({data.get('total_results', 0)} found)")

                    for i, result in enumerate(data.get("results", [])):
                        with st.expander(
                            f"#{i+1} — {result.get('vehicle_id')} | "
                            f"{result.get('panel')} | "
                            f"Similarity: {result.get('score')}",
                            expanded=i == 0
                        ):
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                if result.get("image_url"):
                                    st.image(result["image_url"], use_container_width=True)
                            with c2:
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Similarity", result.get("score"))
                                m2.metric("Damage", result.get("damage_score"))
                                m3.metric("Bucket", result.get("cost_bucket"))
                                narrative = result.get("narrative", {})
                                if narrative:
                                    st.info(narrative.get("summary", "N/A"))

                except Exception as e:
                    st.error(f"Image search failed: {str(e)}")

# ══ Tab 3 — Panel Analyzer ═══════════════════════════════════════
with tab3:
    st.subheader("Panel Damage Assessment (Iteration 1 Engine)")

    col1, col2 = st.columns(2)
    with col1:
        make  = st.selectbox("Make", ["Maruti Suzuki", "Hyundai", "Honda", "Tata", "Mahindra", "Toyota", "Kia", "BMW"])
        model_name = st.text_input("Model", placeholder="Swift, Creta, City")
    with col2:
        panel = st.selectbox("Panel", [
            "Front Bumper Panel", "Rear Bumper Panel", "Bonnet Panel",
            "Dickey Door Panel", "Front Left Door Panel", "Front Right Door Panel",
            "Rear Left Door Panel", "Rear Right Door Panel",
            "Left Quarter Panel", "Right Quarter Panel", "Roof Panel"
        ])

    issues = st.multiselect("Damage Issues", [
        "Scratch-Minor", "Scratch-Major", "Dent-Minor", "Dent-Major",
        "Crack-Minor", "Crack-Major", "Rust-Minor", "Rust-Major",
        "Tear-Minor", "Tear-Major", "Replaced", "Broken", "Accident-Damage"
    ], default=["Scratch-Minor", "Dent-Major"])

    if st.button("🔧 Analyze", type="primary", use_container_width=True):
        if not model_name or not issues:
            st.warning("Please fill all fields.")
        else:
            with st.spinner("Analyzing..."):
                r = requests.post(f"{API_BASE_URL}/v1/analyze", json={
                    "make": make, "model": model_name,
                    "panel": panel, "issues": ", ".join(issues),
                    "include_narrative": True
                }, timeout=30)
                data = r.json()

                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Damage Score", f"{data['damage_score']} / 10")
                m2.metric("Cost Bucket", data["cost_bucket"])
                m3.metric("Est. Cost", data["cost_range"])
                priority = data["priority"]
                color = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🚨"}.get(priority, "⚪")
                m4.metric("Priority", f"{color} {priority}")

                st.progress(data["damage_score"] / 10)

                if data.get("narrative"):
                    n = data["narrative"]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(n.get("summary"))
                        st.warning(n.get("customer_impact"))
                    with c2:
                        st.error(n.get("inspector_note"))
                        st.success(n.get("recommended_action"))