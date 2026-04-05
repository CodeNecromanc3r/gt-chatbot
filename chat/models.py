from django.db import models


class Conversation(models.Model):
    query = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation {self.id} @ {self.created_at}"
