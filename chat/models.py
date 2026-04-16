from django.db import models
from django.contrib.auth.models import User


class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_sessions")
    title = models.CharField(max_length=200, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"ChatSession {self.id} – {self.title}"


class Conversation(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="conversations")
    session = models.ForeignKey(ChatSession, null=True, blank=True, on_delete=models.CASCADE, related_name="messages")
    query = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    response_time_ms = models.IntegerField(null=True, blank=True)
    is_success = models.BooleanField(default=True)
    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Conversation {self.id} @ {self.created_at}"
