import os
import json
import re
from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .models import Conversation

rag_store = None
location_data: list[dict] = []   # raw location records for keyword search

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "chickfila.json"

LOCATION_KEYWORDS = {
    "location", "locations", "address", "near", "nearby", "closest", "closest",
    "where", "hours", "open", "close", "closing", "opening", "phone", "directions",
    "drive", "drivethru", "drive-thru", "restaurant", "restaurants", "store", "stores",
}


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_or_build_store(documents):
    global rag_store
    if rag_store is None:
        embeddings = get_embeddings()
        rag_store = FAISS.from_documents(documents, embeddings)
    else:
        rag_store.add_documents(documents)
    return rag_store


def load_location_data():
    global location_data
    if DATA_FILE.exists():
        location_data = json.loads(DATA_FILE.read_text()).get("locations", [])


# ── location keyword search ───────────────────────────────────────────────────

def search_locations(query: str, max_results: int = 10) -> str:
    """
    Keyword search through raw location data.
    Scores each location by how many query tokens appear in its text fields,
    and returns the top matches formatted as plain text.
    """
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    # remove common stop words that would match everything
    stop = {"the", "a", "an", "in", "near", "around", "of", "is", "are",
            "where", "what", "any", "some", "chick", "fil", "chickfila", "location",
            "locations", "restaurant", "restaurants"}
    tokens -= stop

    scored = []
    for loc in location_data:
        addr = loc.get("address") or {}
        haystack = " ".join(filter(None, [
            loc.get("name", ""),
            addr.get("street", ""),
            addr.get("city", ""),
            addr.get("state", ""),
            addr.get("zip", ""),
        ])).lower()

        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            scored.append((score, loc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [loc for _, loc in scored[:max_results]]

    if not top:
        return "No matching Chick-fil-A locations found for this query."

    lines = []
    for loc in top:
        addr = loc.get("address") or {}
        address_str = ", ".join(filter(None, [
            addr.get("street"), addr.get("city"),
            addr.get("state"), addr.get("zip"),
        ]))
        hours_parts = []
        for h in loc.get("hours") or []:
            days = h.get("day_of_week", [])
            if isinstance(days, list):
                days = "/".join(days)
            opens = h.get("opens", "")
            if opens.lower() == "closed":
                hours_parts.append(f"{days}: Closed")
            else:
                hours_parts.append(f"{days}: {opens}–{h.get('closes', '')}")

        line = f"- {loc['name']}: {address_str}"
        if loc.get("phone"):
            line += f" | {loc['phone']}"
        if hours_parts:
            line += f" | Hours: {'; '.join(hours_parts)}"
        lines.append(line)

    return "\n".join(lines)


# ── prompt & chain ────────────────────────────────────────────────────────────

def _is_location_query(query: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    return bool(tokens & LOCATION_KEYWORDS)


def _build_prompt(location_query: bool = False):
    hint = (
        " Mention specific names, addresses, phone numbers, and hours when available."
        if location_query else ""
    )
    return ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful Chick-fil-A assistant."
            + hint
            + " Use the context below to answer the user's question. "
            "Only say you don't have information if the context is completely unrelated "
            "to the question.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}"),
    ])


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def create_rag_chain(query: str):
    if rag_store is None:
        raise ValueError("No knowledge base loaded yet.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    location_query = _is_location_query(query)

    if location_query:
        context_fn = RunnableLambda(search_locations)
    else:
        retriever = rag_store.as_retriever(search_kwargs={"k": 6})
        context_fn = retriever | _format_docs

    return (
        {"context": context_fn, "question": RunnablePassthrough()}
        | _build_prompt(location_query)
        | llm
        | StrOutputParser()
    )


def interface(request):
    return render(request, "chat/index.html")


@csrf_exempt
@require_POST
def ingest_documents(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    texts = []
    if "documents" in data and isinstance(data["documents"], list):
        for item in data["documents"]:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
    else:
        return JsonResponse(
            {"error": "Provide a \"documents\" list with {\"text\": \"...\"} entries."},
            status=400,
        )

    if not texts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for text in texts:
        chunks = splitter.split_text(text)
        docs.extend([Document(page_content=chunk) for chunk in chunks])

    get_or_build_store(docs)
    return JsonResponse({"status": "ok", "ingested_chunks": len(docs)})


@csrf_exempt
@require_POST
def query_chat(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    query = data.get("query")
    if not query:
        return JsonResponse({"error": "Missing query parameter"}, status=400)

    if rag_store is None:
        return JsonResponse({"error": "Knowledge base not loaded yet."}, status=503)

    try:
        chain = create_rag_chain(query=query)
        answer = chain.invoke(query)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    Conversation(query=query, answer=answer).save()

    return JsonResponse({"query": query, "answer": answer})
