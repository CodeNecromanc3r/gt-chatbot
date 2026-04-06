import os
from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = "chat"

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        import json
        from pathlib import Path
        from langchain_core.documents import Document
        from .views import get_or_build_store

        data_file = Path(__file__).resolve().parent.parent / "data" / "chickfila.json"
        if not data_file.exists():
            return

        data = json.loads(data_file.read_text())
        docs = []

        # menu items
        for item in data.get("menu", []):
            name = item.get("name", "")
            category = item.get("category") or ""
            all_cats = ", ".join(item.get("all_categories") or [])
            url = item.get("url") or ""
            calories = item.get("calories")
            text = f"Menu item: {name}. Category: {category}."
            if all_cats and all_cats != category:
                text += f" Also in: {all_cats}."
            if calories:
                text += f" Calories: {calories}."
            if url and "chick-fil-a.com/menu" in url:
                text += f" Details: {url}."
            docs.append(Document(page_content=text, metadata={"topic": "menu"}))

        # locations
        for loc in data.get("locations", []):
            name  = loc.get("name", "")
            phone = loc.get("phone") or ""
            addr  = loc.get("address") or {}
            street = addr.get("street") or ""
            city   = addr.get("city") or ""
            state  = addr.get("state") or ""
            zip_   = addr.get("zip") or ""

            address_str = ", ".join(filter(None, [street, city, state, zip_]))

            hours_parts = []
            for h in loc.get("hours") or []:
                days = h.get("day_of_week") or []
                if isinstance(days, list):
                    days = "/".join(days)
                opens  = h.get("opens", "")
                closes = h.get("closes", "")
                if opens.lower() == "closed":
                    hours_parts.append(f"{days}: Closed")
                else:
                    hours_parts.append(f"{days}: {opens}–{closes}")
            hours_str = "; ".join(hours_parts)

            text = f"Chick-fil-A location: {name}."
            if address_str:
                text += f" Address: {address_str}."
            if phone:
                text += f" Phone: {phone}."
            if hours_str:
                text += f" Hours: {hours_str}."

            docs.append(Document(page_content=text, metadata={"topic": "locations"}))

        get_or_build_store(docs)

        from .views import load_location_data
        load_location_data()
