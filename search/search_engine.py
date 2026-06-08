# search/search_engine.py

import numpy as np
from vision.clip_engine import embed_text, embed_image_from_file
from vision.vector_store import search_vehicles
from search.query_parser import parse_query
from core.groq_narrator import generate_narrative


# ── Lever 2: Re-ranking ───────────────────────────────────────────

def rerank_results(
    matches: list[dict],
    target_damage_score: float = None,
    alpha: float = 0.7
) -> list[dict]:
    """
    Re-ranks results by combining cosine similarity + damage score proximity.
    alpha=0.7 → 70% visual similarity, 30% damage score match
    """
    if not matches or target_damage_score is None:
        return matches

    for match in matches:
        cosine = match.get("score", 0)
        damage = match.get("damage_score", 0)
        damage_proximity = max(0, 1 - abs(target_damage_score - damage) / 10.0)
        match["reranked_score"] = round(
            alpha * cosine + (1 - alpha) * damage_proximity, 4
        )

    return sorted(matches, key=lambda x: x["reranked_score"], reverse=True)


# ── Lever 3: Multi-image indexing aware search ────────────────────

def expand_query_vectors(text: str, variations: int = 3) -> list:
    """
    Lever 4: Query expansion — generates multiple CLIP embeddings
    for the same query and averages them for better recall.
    """
    base_templates = [
        "{text}",
        "vehicle with {text}",
        "car panel showing {text}",
    ]

    vectors = []
    for template in base_templates[:variations]:
        expanded = template.format(text=text)
        vec = embed_text(expanded)
        vectors.append(vec)

    # Average the vectors
    avg = np.mean(vectors, axis=0)
    # Re-normalize
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm

    return avg.tolist()


# ── Core Search Functions ─────────────────────────────────────────

def search_by_text(
    query: str,
    tenant_id: str = "spinny_india_prod",
    top_k: int = 5,
    use_reranking: bool = True,
    use_query_expansion: bool = True,
) -> dict:
    """
    Full text-to-image search pipeline.
    Query → parse → CLIP embed → Pinecone → rerank → results with cost

    Levers applied:
    - Lever 2: Damage score re-ranking
    - Lever 4: Query expansion (3 CLIP embeddings averaged)
    """
    # Step 1 — Parse query
    parsed = parse_query(query)
    visual_desc = parsed["visual_description"]
    filters = parsed["filters"]

    # Step 2 — CLIP embedding (with optional query expansion)
    if use_query_expansion:
        query_vector = expand_query_vectors(visual_desc)
    else:
        query_vector = embed_text(visual_desc)

    # Step 3 — Pinecone hybrid search
    # Fetch more than top_k for re-ranking headroom
    fetch_k = min(top_k * 3, 20)
    matches = search_vehicles(
        query_vector=query_vector,
        tenant_id=tenant_id,
        max_price=filters.get("max_price"),
        city=filters.get("city"),
        cost_bucket=filters.get("cost_bucket"),
        top_k=fetch_k
    )

    # Step 4 — Re-rank by damage score proximity
    if use_reranking and matches:
        avg_damage = sum(m.get("damage_score", 0) for m in matches) / len(matches)
        matches = rerank_results(matches, target_damage_score=avg_damage)

    # Trim to top_k after re-ranking
    matches = matches[:top_k]

    # Step 5 — Enrich results with narratives
    enriched = []
    for match in matches:
        narrative = generate_narrative(
            panel=match.get("panel", ""),
            issues=[],
            damage_score=match.get("damage_score", 0),
            cost_bucket=match.get("cost_bucket", ""),
            cost_range="",
            make="",
            model="",
        )
        enriched.append({**match, "narrative": narrative})

    return {
        "query":            query,
        "parsed":           parsed,
        "total_results":    len(enriched),
        "query_expanded":   use_query_expansion,
        "reranked":         use_reranking,
        "results":          enriched,
    }


def search_by_image(
    image_bytes: bytes,
    tenant_id: str = "spinny_india_prod",
    max_price: int = None,
    city: str = None,
    top_k: int = 5,
    use_reranking: bool = True,
) -> dict:
    """
    Image-to-image search pipeline.
    Uploaded image → CLIP embed → Pinecone → rerank → similar vehicles
    """
    # Step 1 — CLIP image embedding
    query_vector = embed_image_from_file(image_bytes)

    if query_vector is None:
        return {
            "error":   "Could not process uploaded image",
            "results": []
        }

    # Step 2 — Pinecone search
    fetch_k = min(top_k * 3, 20)
    matches = search_vehicles(
        query_vector=query_vector,
        tenant_id=tenant_id,
        max_price=max_price,
        city=city,
        top_k=fetch_k
    )

    # Step 3 — Re-rank
    if use_reranking and matches:
        avg_damage = sum(m.get("damage_score", 0) for m in matches) / len(matches)
        matches = rerank_results(matches, target_damage_score=avg_damage)

    matches = matches[:top_k]

    # Step 4 — Enrich with narratives
    enriched = []
    for match in matches:
        narrative = generate_narrative(
            panel=match.get("panel", ""),
            issues=[],
            damage_score=match.get("damage_score", 0),
            cost_bucket=match.get("cost_bucket", ""),
            cost_range="",
            make="",
            model="",
        )
        enriched.append({**match, "narrative": narrative})

    return {
        "search_type":   "image",
        "total_results": len(enriched),
        "reranked":      use_reranking,
        "results":       enriched,
    }


# ── Sanity Test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  VisionLens — Search Engine Test (All Levers)")
    print(f"{'='*60}\n")

    queries = [
        "car with dents on front bumper",
        "white SUV with scratches",
        "vehicle with cosmetic damage only",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        result = search_by_text(
            query=query,
            tenant_id="spinny_india_prod",
            top_k=3,
            use_reranking=True,
            use_query_expansion=True,
        )

        print(f"  Visual    : {result['parsed']['visual_description']}")
        print(f"  Expanded  : {result['query_expanded']}")
        print(f"  Reranked  : {result['reranked']}")
        print(f"  Results   : {result['total_results']}")

        for i, r in enumerate(result["results"]):
            print(f"  [{i+1}] {r['vehicle_id']} | "
                  f"{r['panel']} | "
                  f"cosine={r['score']} | "
                  f"reranked={r.get('reranked_score', 'N/A')} | "
                  f"{r['cost_bucket']}")
        print()