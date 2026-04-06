"""
Extract nutrition data from data/nutrition.pdf into data/nutrition-facts.json.

The PDF has messy column-interleaved text, so we use GPT-4o-mini to parse
the raw extracted text into structured JSON.

Usage:
    python data/extract_nutrition.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(Path(__file__).parent.parent / ".env")

PDF_PATH = Path(__file__).parent / "nutrition.pdf"
OUT_FILE = Path(__file__).parent / "nutrition-facts.json"

client = OpenAI()


def main():
    reader = PdfReader(PDF_PATH)
    raw_text = "\n".join(page.extract_text() for page in reader.pages)
    print(f"Extracted {len(raw_text)} chars from {len(reader.pages)} page(s)")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data extraction assistant. Extract ALL menu items and their "
                    "nutrition information from the provided text. The text comes from a "
                    "Chick-fil-A nutrition PDF and may have messy formatting due to PDF "
                    "column extraction. Return ONLY a valid JSON array (no markdown, no "
                    "backticks). Each item should have this structure:\n"
                    '{"name": "...", "category": "...", "serving_size_g": 0, '
                    '"calories_kcal": 0, "total_fat_g": 0, "saturated_fat_g": 0, '
                    '"carbohydrate_g": 0, "sugars_g": 0, "protein_g": 0, '
                    '"salt_g": 0, "allergens": ["..."]}\n'
                    "For allergens, list only those that ARE present (not 'may contain'). "
                    "Use the category headers in the text (Salads, Sides, Entrées, etc.). "
                    "Extract EVERY item you can find."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
        max_tokens=16000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    items = json.loads(raw)
    OUT_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"Saved {len(items)} items to {OUT_FILE}")


if __name__ == "__main__":
    main()
