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

from data.cleaners import (
    normalize_integer,
    normalize_list_of_text,
    normalize_text,
)

load_dotenv(Path(__file__).parent.parent / ".env")

PDF_PATH = Path(__file__).parent / "nutrition.pdf"
OUT_FILE = Path(__file__).parent / "nutrition-facts.json"

client = OpenAI()


def normalize_nutrition_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None

    name = normalize_text(item.get("name"))
    category = normalize_text(item.get("category"))
    if not name or not category:
        return None

    serving_size_g = normalize_integer(item.get("serving_size_g"))
    calories_kcal = normalize_integer(item.get("calories_kcal"))
    if serving_size_g is None or calories_kcal is None:
        return None

    nutrients = {
        "total_fat_g": normalize_integer(item.get("total_fat_g")),
        "saturated_fat_g": normalize_integer(item.get("saturated_fat_g")),
        "carbohydrate_g": normalize_integer(item.get("carbohydrate_g")),
        "sugars_g": normalize_integer(item.get("sugars_g")),
        "protein_g": normalize_integer(item.get("protein_g")),
        "salt_g": normalize_integer(item.get("salt_g")),
    }

    return {
        "name": name,
        "category": category,
        "serving_size_g": serving_size_g,
        "calories_kcal": calories_kcal,
        "total_fat_g": nutrients["total_fat_g"],
        "saturated_fat_g": nutrients["saturated_fat_g"],
        "carbohydrate_g": nutrients["carbohydrate_g"],
        "sugars_g": nutrients["sugars_g"],
        "protein_g": nutrients["protein_g"],
        "salt_g": nutrients["salt_g"],
        "allergens": normalize_list_of_text(item.get("allergens")),
    }


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
    normalized_items = []
    skipped = 0
    for item in items:
        normalized = normalize_nutrition_item(item)
        if normalized is None:
            skipped += 1
            continue
        normalized_items.append(normalized)

    OUT_FILE.write_text(json.dumps(normalized_items, indent=2, ensure_ascii=False))
    print(
        f"Saved {len(normalized_items)} items to {OUT_FILE}"
        + (f" ({skipped} malformed items skipped)" if skipped else "")
    )


if __name__ == "__main__":
    main()
