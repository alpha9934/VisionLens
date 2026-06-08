# core/groq_narrator.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
You are an expert automotive damage assessor for Spinny, India's leading used car platform.
Given a vehicle panel, list of damage issues, damage score (0-10), and cost bucket,
you must return ONLY a valid JSON object with NO markdown, NO backticks, NO preamble.

Return exactly this structure:
{
  "summary": "2-3 sentence plain English damage summary for a customer",
  "inspector_note": "1 sentence technical note for the refurbishment team",
  "customer_impact": "How this affects the buyer — resale value, safety, aesthetics",
  "recommended_action": "Specific repair action recommended",
  "confidence": "High / Medium / Low"
}
Rules:
- Write summary in simple language a car buyer understands
- Be specific about the panel and damage type
- Keep each field under 40 words
"""

def generate_narrative(
    panel: str,
    issues: list,
    damage_score: float,
    cost_bucket: str,
    cost_range: str,
    make: str = "",
    model: str = "",
) -> dict:
    issues_str = ", ".join(issues) if issues else "No issues detected"
    vehicle_str = f"{make} {model}".strip() or "Unknown Vehicle"

    user_prompt = f"""
Vehicle: {vehicle_str}
Panel: {panel}
Damage Issues: {issues_str}
Damage Score: {damage_score} / 10
Cost Bucket: {cost_bucket} ({cost_range})
Generate the damage assessment JSON.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "summary":            "Damage assessment unavailable.",
            "inspector_note":     "LLM parse error.",
            "customer_impact":    "Unknown",
            "recommended_action": "Manual inspection recommended.",
            "confidence":         "Low",
        }
    except Exception as e:
        return {
            "summary":            f"API error: {str(e)}",
            "inspector_note":     "Check GROQ_API_KEY in .env",
            "customer_impact":    "Unknown",
            "recommended_action": "Retry after fixing API connection.",
            "confidence":         "Low",
        }

if __name__ == "__main__":
    result = generate_narrative(
        panel="Front Bumper Panel",
        issues=["Scratch-Minor", "Dent-Major"],
        damage_score=0.71,
        cost_bucket="Cosmetic Fix",
        cost_range="₹1,000 – ₹3,000",
        make="Maruti Suzuki",
        model="Swift",
    )
    for k, v in result.items():
        print(f"{k.upper():<22}: {v}")
