# evaluator.py

import pandas as pd
import numpy as np
import requests
import time
import io
import os
from PIL import Image
from search.search_engine import search_by_image

# ── 1. Load Ground Truth ─────────────────────────────────────────
print("Loading Ground Truth Dataset...")
df = pd.read_csv('outputs/scored_dataset.csv')

# ── 2. AP@K Calculation ──────────────────────────────────────────
def calculate_average_precision_at_k(actual_panel, retrieved_panels, k=5):
    if not retrieved_panels:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(retrieved_panels[:k]):
        if p == actual_panel:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(retrieved_panels), k)

# ── 3. Test Batch Setup ──────────────────────────────────────────
sample_size = 100
sample_test_set = df.sample(n=sample_size, random_state=42)

ap_scores        = []
hits_at_k        = 0
successful_queries = 0
query_log        = []   # for detailed chart data

print(f"\nInitiating Live Vector Retrieval Evaluation ({sample_size} records)...\n")

for i, (index, row) in enumerate(sample_test_set.iterrows(), 1):
    actual_panel = row['panel']

    raw_urls = str(row['media_url']).split(',')
    image_url = None
    for url in raw_urls:
        url = url.strip()
        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_url = url
            break

    if not image_url:
        continue

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_bytes = response.content

        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()

        search_output = search_by_image(image_bytes=image_bytes, top_k=5)
        retrieved_results = search_output.get("results", [])
        retrieved_panels  = [r.get("panel") for r in retrieved_results[:5]]
        top_scores        = [r.get("score", 0) for r in retrieved_results[:5]]

        ap   = calculate_average_precision_at_k(actual_panel, retrieved_panels, k=5)
        hit  = actual_panel in retrieved_panels

        ap_scores.append(ap)
        if hit:
            hits_at_k += 1
        successful_queries += 1

        query_log.append({
            "query_id":      successful_queries,
            "actual_panel":  actual_panel,
            "top1_panel":    retrieved_panels[0] if retrieved_panels else "None",
            "top1_score":    top_scores[0] if top_scores else 0,
            "ap_score":      round(ap, 4),
            "hit":           hit,
            "retrieved":     ", ".join(retrieved_panels),
        })

        if i % 10 == 0:
            running_map = np.mean(ap_scores)
            print(f"  [{i}/{sample_size}] Running mAP@5: {running_map:.4f} | Hits: {hits_at_k}/{successful_queries}")

    except Exception as e:
        continue

    time.sleep(0.3)

# ── 4. Final Metrics ─────────────────────────────────────────────
mAP_5    = np.mean(ap_scores) if ap_scores else 0
recall_5 = hits_at_k / successful_queries if successful_queries > 0 else 0

print("\n" + "="*52)
print("  🚀 VISIONLENS RETRIEVAL METRICS (Live Index)")
print("="*52)
print(f"  Total Queries Evaluated : {successful_queries}")
print(f"  mAP@5 Score             : {mAP_5:.4f}")
print(f"  Recall@5 Score          : {recall_5:.4f} ({hits_at_k}/{successful_queries} hits)")
print("="*52)

# ── 5. Interactive Charts ─────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    log_df = pd.DataFrame(query_log)

    # Color palette
    BG       = "#0e1117"
    CARD     = "#161b22"
    RED      = "#ff4b4b"
    BLUE     = "#4b9eff"
    GREEN    = "#2ea043"
    YELLOW   = "#f0a500"
    GRID     = "#21262d"
    TEXT     = "#c9d1d9"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "mAP@5 Distribution",
            "Cumulative mAP@5 Over Queries",
            "AP@5 Score per Query",
            "Panel Hit Rate Breakdown"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # ── Chart 1: AP@5 Histogram ──────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=ap_scores,
            nbinsx=12,
            marker_color=RED,
            opacity=0.85,
            name="AP@5 Distribution",
            hovertemplate="AP Score: %{x:.3f}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    # ── Chart 2: Cumulative mAP@5 ───────────────────────────────
    cumulative_map = [np.mean(ap_scores[:i+1]) for i in range(len(ap_scores))]
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_map)+1)),
            y=cumulative_map,
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            name="Cumulative mAP@5",
            fill="tozeroy",
            fillcolor="rgba(75,158,255,0.1)",
            hovertemplate="Query #%{x}<br>mAP@5: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    # Final mAP line
    fig.add_hline(
        y=mAP_5,
        line_dash="dash",
        line_color=YELLOW,
        line_width=1.5,
        annotation_text=f"Final mAP@5: {mAP_5:.4f}",
        annotation_font_color=YELLOW,
        row=1, col=2
    )

    # ── Chart 3: Per-query AP scatter ───────────────────────────
    colors = [GREEN if h else RED for h in log_df["hit"]]
    fig.add_trace(
        go.Scatter(
            x=log_df["query_id"],
            y=log_df["ap_score"],
            mode="markers",
            marker=dict(
                color=colors,
                size=7,
                opacity=0.8,
                line=dict(width=0.5, color=GRID)
            ),
            name="Per-Query AP",
            customdata=np.stack([
                log_df["actual_panel"],
                log_df["top1_panel"],
                log_df["top1_score"],
                log_df["hit"].map({True: "✅ Hit", False: "❌ Miss"})
            ], axis=-1),
            hovertemplate=(
                "Query #%{x}<br>"
                "AP@5: %{y:.4f}<br>"
                "Actual Panel: %{customdata[0]}<br>"
                "Top-1 Result: %{customdata[1]}<br>"
                "Top-1 Score: %{customdata[2]:.4f}<br>"
                "Result: %{customdata[3]}<extra></extra>"
            )
        ),
        row=2, col=1
    )

    # ── Chart 4: Panel hit rate bar ─────────────────────────────
    panel_stats = log_df.groupby("actual_panel").agg(
        total=("hit", "count"),
        hits=("hit", "sum")
    ).reset_index()
    panel_stats["hit_rate"] = panel_stats["hits"] / panel_stats["total"]
    panel_stats = panel_stats.sort_values("hit_rate", ascending=True)

    fig.add_trace(
        go.Bar(
            x=panel_stats["hit_rate"],
            y=panel_stats["actual_panel"],
            orientation="h",
            marker=dict(
                color=panel_stats["hit_rate"],
                colorscale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
                showscale=False
            ),
            name="Panel Hit Rate",
            hovertemplate=(
                "%{y}<br>"
                "Hit Rate: %{x:.1%}<extra></extra>"
            )
        ),
        row=2, col=2
    )

    # ── Global Layout ────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>VisionLens — Retrieval Performance Dashboard</b><br>"
                f"<sup>mAP@5: {mAP_5:.4f} | Recall@5: {recall_5:.4f} | "
                f"Queries: {successful_queries} | Hits: {hits_at_k}</sup>"
            ),
            font=dict(color=TEXT, size=18),
            x=0.5
        ),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Arial"),
        showlegend=False,
        height=750,
        margin=dict(t=100, b=40, l=60, r=40),
    )

    # Grid styling for all subplots
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                gridcolor=GRID, gridwidth=0.5,
                zeroline=False, linecolor=GRID,
                row=row, col=col
            )
            fig.update_yaxes(
                gridcolor=GRID, gridwidth=0.5,
                zeroline=False, linecolor=GRID,
                row=row, col=col
            )

    # Subplot title colors
    for ann in fig.layout.annotations:
        ann.font.color = TEXT
        ann.font.size  = 13

    # Save interactive HTML
    fig.write_html("visionlens_performance_dashboard.html")
    print("\n[INFO] Interactive dashboard → visionlens_performance_dashboard.html")

    # Save static PNG
    try:
        fig.write_image("visionlens_performance_chart.png", scale=2)
        print("[INFO] Static chart → visionlens_performance_chart.png")
    except Exception:
        print("[INFO] Static PNG skipped (install kaleido: pip3 install kaleido)")

    # Open in browser automatically
    import webbrowser, pathlib
    webbrowser.open(pathlib.Path("visionlens_performance_dashboard.html").resolve().as_uri())

except ImportError:
    print("\n[INFO] Plotly not installed. Run: pip3 install plotly kaleido")
    print("       Falling back to matplotlib...")

    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('#0e1117')

    for ax in axes:
        ax.set_facecolor('#161b22')

    axes[0].hist(ap_scores, bins=10, color='#ff4b4b', edgecolor='#0e1117', alpha=0.85)
    axes[0].set_title('AP@5 Distribution', color='white', fontweight='bold')
    axes[0].set_xlabel('AP Score', color='#8b949e')
    axes[0].set_ylabel('Count', color='#8b949e')
    axes[0].grid(color='#21262d', linewidth=0.5)

    cumulative = [np.mean(ap_scores[:i+1]) for i in range(len(ap_scores))]
    axes[1].plot(cumulative, color='#4b9eff', linewidth=2)
    axes[1].axhline(y=mAP_5, color='#f0a500', linestyle='--', linewidth=1.5,
                    label=f'Final mAP@5: {mAP_5:.4f}')
    axes[1].set_title('Cumulative mAP@5', color='white', fontweight='bold')
    axes[1].set_xlabel('Query #', color='#8b949e')
    axes[1].legend(facecolor='#161b22', labelcolor='white')
    axes[1].grid(color='#21262d', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('visionlens_performance_chart.png', facecolor='#0e1117')
    print("[INFO] Chart saved → visionlens_performance_chart.png")