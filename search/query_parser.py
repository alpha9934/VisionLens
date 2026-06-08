# search/query_parser.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a query parser for VisionLens — an AI-powered vehicle search engine for Spinny and Cars24.

Extract from the user query and return ONLY valid JSON, no markdown, no backticks:
{
  "visual_description": "what to visually search for in CLIP embedding space",
  "filters": {
    "max_price": <integer in rupees or null>,
    "city": "<city name or null>",
    "cost_bucket": "<one of: Cosmetic Fix, Standard Repair, Major Repair, Panel Replacement or null>"
  },
  "search_intent": "<one of: damage_search, price_search, visual_search, general>"
}

Rules:
- visual_description must describe visual appearance only (color, shape, damage type, panel)
- Convert lakh to rupees: 8 lakhs = 800000
- Normalize city names: bengaluru/bangalore → Bengaluru
- cost_bucket only if user explicitly mentions repair severity
- search_intent helps rank results appropriately

Examples:
Input: "grey sedan under 8 lakhs in Bengaluru with dents on front bumper"
Output: {"visual_description": "grey sedan dents front bumper", "filters": {"max_price": 800000, "city": "Bengaluru", "cost_bucket": null}, "search_intent": "damage_search"}

Input: "show me cars with only cosmetic damage"
Output: {"visual_description": "minor scratches cosmetic damage car panel", "filters": {"max_price": null, "city": null, "cost_bucket": "Cosmetic Fix"}, "search_intent": "damage_search"}

Input: "white SUV with scratch on rear bumper"
Output: {"visual_description": "white SUV scratch rear bumper panel", "filters": {"max_price": null, "city": null, "cost_bucket": null}, "search_intent": "visual_search"}
"""


def parse_query(raw_query: str) -> dict:
    """
    Parses natural language query into structured search parameters.
    Returns visual_description, filters, and search_intent.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": raw_query}
            ],
            temperature=0,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        # Fallback — treat entire query as visual description
        return {
            "visual_description": raw_query,
            "filters": {"max_price": None, "city": None, "cost_bucket": None},
            "search_intent": "general"
        }
    except Exception as e:
        print(f"  [QueryParser] Error: {e}")
        return {
            "visual_description": raw_query,
            "filters": {"max_price": None, "city": None, "cost_bucket": None},
            "search_intent": "general"
        }


if __name__ == "__main__":
    test_queries = [
        "grey sedan under 8 lakhs in Bengaluru with dents on front bumper",
        "white SUV with scratch on rear bumper",
        "show me cars with only cosmetic damage",
        "Honda City with major accident damage",
        "car under 5 lakhs in Delhi",
    ]

    print(f"\n{'='*60}")
    print("  VisionLens — Query Parser Test")
    print(f"{'='*60}\n")

    for q in test_queries:
        print(f"Query: {q}")
        result = parse_query(q)
        print(f"  Visual: {result['visual_description']}")
        print(f"  Filters: {result['filters']}")
        print(f"  Intent: {result['search_intent']}")
        print()