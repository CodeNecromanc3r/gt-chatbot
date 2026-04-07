"""
Chick-fil-A data scraper — uses the public WordPress REST API + JSON-LD.

  Menu   : /wp-json/wp/v2/menu-item  (549 items, paginated)
  Locations: /wp-json/wp/v2/location (3,405 locations, paginated)
             each location page has JSON-LD with address / phone / hours

Usage:
    python data/scrape.py
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

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
LOCATION_CONCURRENCY = 20

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


# calorie scraping via Playwright

MENU_CATEGORIES = [
    "/menu/breakfast",
    "/menu/entrees",
    "/menu/salads",
    "/menu/sides",
    "/menu/kids-meals",
    "/menu/family-meals",
    "/menu/treats",
    "/menu/beverages",
    "/menu/sauces-dressings",
]

async def scrape_calories() -> dict[str, int]:
    """
    Load each menu category page with Playwright and extract calorie counts.
    Returns a dict of  normalised_name -> calories.
    """
    calorie_map: dict[str, int] = {}

    def normalise(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=HEADERS["User-Agent"])
        page = await ctx.new_page()

        for path in MENU_CATEGORIES:
            try:
                await page.goto(BASE + path, wait_until="networkidle", timeout=25000)
            except Exception:
                continue

            cards = await page.locator(
                "li[class*='menu' i], li[class*='card' i], article, [class*='MenuItem' i]"
            ).all()

            for card in cards:
                try:
                    name = (await card.locator("h2, h3, [class*='name' i]").first.inner_text()).strip()
                except Exception:
                    continue
                if not name:
                    continue

                cal_raw = ""
                try:
                    cal_raw = (await card.locator("[class*='cal' i]").first.inner_text()).strip()
                except Exception:
                    pass
                if not cal_raw:
                    try:
                        cal_raw = (await card.locator("p").first.inner_text()).strip()
                    except Exception:
                        pass

                m = re.search(r"(\d[\d,]*)\s*cal", cal_raw, re.IGNORECASE)
                if m:
                    calorie_map[normalise(name)] = int(m.group(1).replace(",", ""))

        await browser.close()

    print(f"  calorie data found for {len(calorie_map)} items")
    return calorie_map


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

    print("  scraping calorie data from menu pages …")
    calorie_map = await scrape_calories()

    def normalise(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    menu = []
    for item in raw_items:
        tax_ids = item.get("menu_taxonomy") or []
        categories = [tax_map[t] for t in tax_ids if t in tax_map]
        name = item["title"]["rendered"]
        menu.append({
            "name":           name,
            "slug":           item["slug"],
            "category":       categories[0] if categories else None,
            "all_categories": categories,
            "calories":       calorie_map.get(normalise(name)),
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
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=20),
    ) as client:
        print("[ 1/2 ] Menu")
        menu = await fetch_menu(client)
        print(f"  ✓ {len(menu)} menu items\n")

        print("[ 2/2 ] Locations")
        locations = await fetch_locations(client)
        print(f"  ✓ {len(locations)} locations\n")

    OUT_FILE.write_text(
        json.dumps({"menu": menu, "locations": locations}, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"=== Done — written to {OUT_FILE} ===")


if __name__ == "__main__":
    asyncio.run(main())
