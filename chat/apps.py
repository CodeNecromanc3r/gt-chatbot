import os
from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = "chat"

    def ready(self):
        # In development, runserver spawns two processes — only load in the
        # reloader child (RUN_MAIN=true) to avoid doing it twice.
        # In production (gunicorn), RUN_MAIN is never set, so always load.
        is_dev_server = "RUN_MAIN" in os.environ
        if is_dev_server and os.environ.get("RUN_MAIN") != "true":
            return

        from .views import get_or_build_store, load_location_data, _get_connection_string

        if _get_connection_string():
            # With pgvector, documents are already persisted in PostgreSQL.
            # Just connect to the existing collection — no need to re-embed.
            get_or_build_store()
        else:
            # Local dev with SQLite — need to load into memory each time.
            from .views import reload_knowledge_base
            reload_knowledge_base()

        load_location_data()
