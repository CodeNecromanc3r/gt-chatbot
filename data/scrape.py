"""
Chick-fil-A data scraper — uses the public WordPress REST API + JSON-LD.

  Menu   : /wp-json/wp/v2/menu-item  (549 items, paginated)
  Locations: /wp-json/wp/v2/location (3,405 locations, paginated)
             each location page has JSON-LD with address / phone / hours

Usage:
    python data/scrape.py
"""

import asyncio
import html as html_module
import json
import re
from pathlib import Path

import httpx

from data.cleaners import (
    normalize_address,
    normalize_hours_entry,
    normalize_integer,
    normalize_list_of_text,
    normalize_text,
)

OUT_FILE = Path(__file__).parent / "chickfila.json"
BASE     = "https://www.chick-fil-a.com"
API      = f"{BASE}/wp-json/wp/v2"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# max simultaneous HTTP connections when fetching location pages
LOCATION_CONCURRENCY = 50

NUTRITION_ALLERGENS_URL = f"{BASE}/nutrition-allergens"

# helpers

async def paginate(client: httpx.AsyncClient, endpoint: str, fields: str) -> list[dict]:
    """Fetch all pages of a WP REST endpoint."""
    results, page = [], 1
    while True:
        r = await client.get(
            endpoint,
            params={"per_page": 100, "page": page, "_fields": fields},
        )
        if r.status_code == 400:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        results.extend(batch)
        total_pages = int(r.headers.get("x-wp-totalpages", 1))
        if page >= total_pages:
            break
        page += 1
    return results


def extract_location_ld(html: str) -> dict | None:
    """Pull the Restaurant JSON-LD block out of a location page."""
    blocks = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL,
    )
    for raw in blocks:
        try:
            data = json.loads(raw.strip())
            graph = data.get("@graph", [data] if isinstance(data, dict) else data)
            for node in graph:
                if isinstance(node, dict) and node.get("@type") == "Restaurant":
                    return node
        except Exception:
            pass
    return None


def parse_hours(spec: list) -> list[dict]:
    hours = []
    for h in (spec or []):
        if not isinstance(h, dict):
            continue
        hours.append(normalize_hours_entry(h))
    return hours


def normalize_menu_item(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}

    all_categories = normalize_list_of_text(item.get("all_categories") or item.get("menu_taxonomy") or [])
    category = normalize_text(item.get("category") or (all_categories[0] if all_categories else ""))

    return {
        "name": normalize_text(item.get("name")),
        "slug": normalize_text(item.get("slug")),
        "category": category,
        "all_categories": all_categories,
        "calories": normalize_integer(item.get("calories")),
        "url": normalize_text(item.get("url")),
        "image_url": normalize_text(item.get("image_url")),
    }


def normalize_location(loc: dict) -> dict:
    if not isinstance(loc, dict):
        loc = {}
    return {
        "name": normalize_text(loc.get("name")),
        "url": normalize_text(loc.get("url")),
        "phone": normalize_text(loc.get("phone")),
        "image_url": normalize_text(loc.get("image_url")),
        "address": normalize_address(loc.get("address") or {}),
        "hours": [normalize_hours_entry(h) for h in (loc.get("hours") or []) if isinstance(h, dict)],
    }


# nutrition data from nutrition-allergens page

def _normalize_name(s: str) -> str:
    """Normalize item name for fuzzy matching."""
    s = html_module.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.lower()
    s = re.sub(r"[®™°]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def fetch_nutrition(client: httpx.AsyncClient) -> dict[str, dict]:
    """
    Fetch https://www.chick-fil-a.com/nutrition-allergens and extract the
    full nutrition table embedded in the page's JS state.
    Returns a dict of normalised_name -> {label: value, ...}.
    """
    r = await client.get(NUTRITION_ALLERGENS_URL, headers={**HEADERS, "Accept": "text/html"})
    r.raise_for_status()

    scripts = re.findall(r"<script[^>]*>(.*?)</script>", r.text, re.DOTALL)
    store = None
    for s in scripts:
        if "nutrition-allergens-table-store" in s:
            try:
                store = json.loads(s)["state"]["nutrition-allergens-table-store"]
                break
            except Exception:
                continue

    if not store:
        print("  WARNING: could not parse nutrition-allergens page")
        return {}

    nutrition_map: dict[str, dict] = {}
    for cat in store.get("activeTableData", []):
        for item in cat.get("items", []):
            fields = {f["label"]: f["value"] for f in item.get("fields", [])}
            nutrition_map[_normalize_name(item["title"])] = fields
            for sub in item.get("sub_items", []):
                sub_fields = {f["label"]: f["value"] for f in sub.get("fields", [])}
                nutrition_map[_normalize_name(sub["title"])] = sub_fields

    print(f"  nutrition data found for {len(nutrition_map)} items")
    return nutrition_map


def merge_nutrition(menu: list[dict], nutrition_map: dict[str, dict]) -> list[dict]:
    """Match menu items to nutrition data by name and attach the nutrition dict."""
    matched = 0
    for item in menu:
        norm = _normalize_name(item.get("name", ""))
        if norm in nutrition_map:
            item["nutrition"] = nutrition_map[norm]
            matched += 1
        else:
            # Try partial match
            for nk, nv in nutrition_map.items():
                if norm in nk or nk in norm:
                    item["nutrition"] = nv
                    matched += 1
                    break
    print(f"  matched nutrition for {matched}/{len(menu)} menu items")
    return menu


# menu

async def fetch_menu(client: httpx.AsyncClient) -> list[dict]:
    print("  fetching taxonomy map …")
    # build id -> category name map
    tax_terms = await paginate(client, f"{API}/menu_taxonomy", "id,name,slug")
    tax_map = {t["id"]: t["name"] for t in tax_terms}

    print("  fetching all menu items …")
    raw_items = await paginate(
        client, f"{API}/menu-item",
        "id,title,slug,link,featured_media,menu_taxonomy,menu_item_type",
    )

    # fetch media URLs for images in one batch
    media_ids = [i["featured_media"] for i in raw_items if i.get("featured_media")]
    media_map: dict[int, str] = {}
    for chunk_start in range(0, len(media_ids), 100):
        chunk = media_ids[chunk_start:chunk_start + 100]
        try:
            r = await client.get(
                f"{API}/media",
                params={"include": ",".join(map(str, chunk)), "per_page": 100, "_fields": "id,source_url"},
            )
            for m in r.json():
                media_map[m["id"]] = m.get("source_url", "")
        except Exception:
            pass

    menu = []
    for item in raw_items:
        tax_ids = item.get("menu_taxonomy") or []
        categories = [tax_map[t] for t in tax_ids if t in tax_map]
        menu.append({
            "name":           item["title"]["rendered"],
            "slug":           item["slug"],
            "category":       categories[0] if categories else None,
            "all_categories": categories,
            "url":            item.get("link", ""),
            "image_url":      media_map.get(item.get("featured_media", 0)),
        })

    return [normalize_menu_item(item) for item in menu]


# locations

async def fetch_one_location(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    loc_link: str,
name: str,
) -> dict | None:
    async with sem:
        try:
            r = await client.get(loc_link, headers={**HEADERS, "Accept": "text/html"})
            ld = extract_location_ld(r.text)
            if not ld:
                return normalize_location({"name": name, "url": loc_link})
            addr = ld.get("address", {})
            return normalize_location({
                "name":      ld.get("name", name),
                "url":       loc_link,
                "phone":     ld.get("telephone"),
                "image_url": ld.get("image"),
                "address": {
                    "street": addr.get("streetAddress"),
                    "city":   addr.get("addressLocality"),
                    "state":  addr.get("addressRegion"),
                    "zip":    addr.get("postalCode"),
                    "country": addr.get("addressCountry", "US"),
                },
                "hours": parse_hours(ld.get("openingHoursSpecification")),
            })
        except Exception:
            return normalize_location({"name": name, "url": loc_link})


async def fetch_locations(client: httpx.AsyncClient) -> list[dict]:
    print("  fetching location index …")
    raw_locs = await paginate(client, f"{API}/location", "id,title,link")
    print(f"  fetched {len(raw_locs)} location entries — downloading detail pages …")

    sem = asyncio.Semaphore(LOCATION_CONCURRENCY)
    tasks = [
        fetch_one_location(client, sem, loc["link"], loc["title"]["rendered"])
        for loc in raw_locs
    ]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        if result:
            results.append(result)
        if i % 200 == 0 or i == len(tasks):
            print(f"    {i}/{len(tasks)} locations processed …")

    return results


# main

async def main():
    print("=== Chick-fil-A Scraper ===\n")

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=20, follow_redirects=True,
        limits=httpx.Limits(max_connections=60, max_keepalive_connections=40),
    ) as client:
        print("[ 1/3 ] Menu")
        menu = await fetch_menu(client)
        print(f"  ✓ {len(menu)} menu items\n")

        print("[ 2/3 ] Locations")
        locations = await fetch_locations(client)
        print(f"  ✓ {len(locations)} locations\n")

        print("[ 3/3 ] Nutrition (nutrition-allergens page)")
        nutrition_map = await fetch_nutrition(client)
        menu = merge_nutrition(menu, nutrition_map)
        print()

    OUT_FILE.write_text(
        json.dumps({"menu": menu, "locations": locations}, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"=== Done — written to {OUT_FILE} ===")


if __name__ == "__main__":
    asyncio.run(main())
