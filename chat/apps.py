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

        from .views import reload_knowledge_base
        reload_knowledge_base()
