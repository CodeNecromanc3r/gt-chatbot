from django.db import models
from django.conf import settings
from project.app.retrievers import DocumentRetriever
from pgvector.django import VectorField
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
store = {}

def conversational_rag(condense_question_prompt_template, system_prompt_template,):
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [("system", condense_question_prompt_template), ("placeholder", "{chat_history}"), ("human", "{input}"),],
    )

class DocumentChunk(models.Model):
    source = models.CharField(max_length=500)
    content = models.TextField()
    embedding = VectorField(dimensions=1536)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        indexes = [
            models.Index(fields=['source']),
        ]

class Conversation(models.Model):
    query = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation {self.id} @ {self.created_at}"