# vision/vector_store.py

import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME_V2", "visionlens-index"))


def upsert_vehicle(
    vehicle_id: str,
    tenant_id: str,
    image_url: str,
    vector: list[float],
    panel: str,
    damage_score: float,
    cost_bucket: str,
    price: int,
    city: str,
    img_index: int = 0
) -> bool:
    try:
        index.upsert(
            vectors=[{
                "id": f"{vehicle_id}_img_{img_index}",
                "values": vector,
                "metadata": {
                    "tenant_id":    tenant_id,
                    "vehicle_id":   vehicle_id,
                    "panel":        panel,
                    "damage_score": damage_score,
                    "cost_bucket":  cost_bucket,
                    "price":        price,
                    "city":         city,
                    "image_url":    image_url,
                }
            }],
            namespace=tenant_id
        )
        return True
    except Exception as e:
        print(f"  [Pinecone] Upsert error: {e}")
        return False


def search_vehicles(
    query_vector: list[float],
    tenant_id: str,
    max_price: int = None,
    city: str = None,
    cost_bucket: str = None,
    top_k: int = 5
) -> list[dict]:
    try:
        filters = {}
        if max_price:
            filters["price"] = {"$lte": max_price}
        if city:
            filters["city"] = {"$eq": city}
        if cost_bucket:
            filters["cost_bucket"] = {"$eq": cost_bucket}

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filters if filters else None,
            include_metadata=True,
            namespace=tenant_id
        )
        return [
            {
                "score":        round(m.score, 4),
                "vehicle_id":   m.metadata.get("vehicle_id"),
                "panel":        m.metadata.get("panel"),
                "damage_score": m.metadata.get("damage_score"),
                "cost_bucket":  m.metadata.get("cost_bucket"),
                "price":        m.metadata.get("price"),
                "city":         m.metadata.get("city"),
                "image_url":    m.metadata.get("image_url"),
            }
            for m in results.matches
        ]
    except Exception as e:
        print(f"  [Pinecone] Search error: {e}")
        return []


def get_index_stats(tenant_id: str = None) -> dict:
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors":    stats.total_vector_count,
            "dimension":        stats.dimension,
            "namespaces":       {k: v.vector_count for k, v in stats.namespaces.items()},
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("\n[TEST] Pinecone connection...")
    stats = get_index_stats()
    print(f"  Index stats: {stats}")
    print("✅ Vector store ready!")