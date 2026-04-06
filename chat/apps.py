import os
from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = "chat"

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        from .views import reload_knowledge_base
        reload_knowledge_base()
