import os
import json
import re
import time
import threading
import subprocess
from pathlib import Path
from datetime import timedelta
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Avg
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .models import Conversation

rag_store = None
location_data: list[dict] = []   # raw location records for keyword search
_scrape_state: dict = {"status": "idle", "started_at": None, "finished_at": None, "message": ""}
_pdf_state: dict = {"status": "idle", "started_at": None, "finished_at": None, "message": ""}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "chickfila.json"
NUTRITION_FILE = Path(__file__).resolve().parent.parent / "data" / "nutrition-facts.json"

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
        rag_store = InMemoryVectorStore.from_documents(documents, embeddings)
    else:
        rag_store.add_documents(documents)
    return rag_store


def load_location_data():
    global location_data
    if DATA_FILE.exists():
        location_data = json.loads(DATA_FILE.read_text(encoding='utf-8')).get("locations", [])


def build_documents():
    """Build all LangChain documents from data files. Used by apps.py and reload endpoint."""
    from langchain_core.documents import Document

    docs = []
    if DATA_FILE.exists():
        data = json.loads(DATA_FILE.read_text(encoding='utf-8'))

        for item in data.get("menu", []):
            name = item.get("name", "")
            category = item.get("category") or ""
            all_cats = ", ".join(item.get("all_categories") or [])
            url = item.get("url") or ""
            nutrition = item.get("nutrition") or {}
            calories = nutrition.get("Calories") or item.get("calories")
            serving_size = nutrition.get("Serving Size")
            text = f"Menu item: {name}. Category: {category}."
            if all_cats and all_cats != category:
                text += f" Also in: {all_cats}."
            if calories:
                serving_note = f" (per {serving_size} serving)" if serving_size else ""
                text += f" Calories: {calories}{serving_note}."
            if nutrition:
                parts = []
                for key in ("Fat (g)", "Sat. Fat (g)", "Cholesterol (mg)", "Sodium (mg)", "Carbohydrates (g)", "Fiber (g)", "Sugar (g)", "Protein (g)"):
                    if key in nutrition:
                        parts.append(f"{key}: {nutrition[key]}")
                if parts:
                    text += " Nutrition: " + ", ".join(parts) + "."
            if url and "chick-fil-a.com/menu" in url:
                text += f" Details: {url}."
            metadata = {"topic": "Menu", "title": name}
            if url:
                metadata["url"] = url
            docs.append(Document(page_content=text, metadata=metadata))

        # Build category summary documents so broad queries like
        # "what salads do you have" retrieve a single doc listing all items.
        category_items: dict[str, list[str]] = {}
        for item in data.get("menu", []):
            cat = item.get("category") or ""
            if cat:
                category_items.setdefault(cat, []).append(item.get("name", ""))
        for cat, names in category_items.items():
            # Deduplicate while preserving order
            seen = set()
            unique = [n for n in names if not (n in seen or seen.add(n))]
            text = f"Chick-fil-A {cat} menu items: {', '.join(unique)}."
            docs.append(Document(page_content=text, metadata={"topic": "Menu", "title": f"{cat} (category)"}))

        for loc in data.get("locations", []):
            name = loc.get("name", "")
            phone = loc.get("phone") or ""
            addr = loc.get("address") or {}
            address_str = ", ".join(filter(None, [
                addr.get("street", ""), addr.get("city", ""),
                addr.get("state", ""), addr.get("zip", ""),
            ]))
            hours_parts = []
            for h in loc.get("hours") or []:
                days = h.get("day_of_week") or []
                if isinstance(days, list):
                    days = "/".join(days)
                opens = h.get("opens", "")
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
            metadata = {"topic": "Location", "title": name}
            if addr.get("city"):
                metadata["city"] = addr.get("city")
            docs.append(Document(page_content=text, metadata=metadata))

    if NUTRITION_FILE.exists():
        # Second ingestion source: PDF document (nutrition.pdf parsed via pdfplumber).
        # This source provides allergen information and per-100g breakdowns that
        # the web-scraped nutrition data does not include.
        nutrition = json.loads(NUTRITION_FILE.read_text(encoding='utf-8'))
        for item in nutrition:
            name = item.get("name", "")
            category = item.get("category", "")
            serving = item.get("serving_size_g", "?")
            allergens = ", ".join(item.get("allergens", [])) or "None listed"
            text = (
                f"Allergen and detailed nutrition info for {name} (source: Chick-fil-A nutrition PDF)."
                f" Category: {category}."
                f" Allergens: {allergens}."
                f" Per 100g breakdown — Calories: {item.get('calories_kcal', '?')} kcal,"
                f" Total fat: {item.get('total_fat_g', '?')}g,"
                f" Saturated fat: {item.get('saturated_fat_g', '?')}g,"
                f" Carbohydrates: {item.get('carbohydrate_g', '?')}g,"
                f" Sugars: {item.get('sugars_g', '?')}g,"
                f" Protein: {item.get('protein_g', '?')}g,"
                f" Salt: {item.get('salt_g', '?')}g."
                f" Note: these are per-100g figures, not whole-meal totals."
            )
            docs.append(Document(page_content=text, metadata={"topic": "Allergens", "title": name}))

    return docs


def reload_knowledge_base():
    """Clear and rebuild the vector store and location data from disk."""
    global rag_store
    rag_store = None
    docs = build_documents()
    get_or_build_store(docs)
    load_location_data()
    return len(docs)


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


def search_locations_with_sources(query: str, max_results: int = 10) -> tuple:
    """
    Keyword search through raw location data.
    Returns tuple of (formatted_text, matched_locations) for source extraction.
    """
    tokens = set(re.findall(r"[a-z]+", query.lower()))
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
        return "No matching Chick-fil-A locations found for this query.", []

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

    return "\n".join(lines), top




def _is_location_query(query: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    return bool(tokens & LOCATION_KEYWORDS)


def _build_prompt(location_query: bool = False):
    hint = (
        " Mention specific names, addresses, phone numbers, and hours when available."
        if location_query else ""
    )
    source_instruction = (
        "When answering, always start with 'Based on the Chick-fil-A website/locations data, ...' to reference your source."
        if location_query else
        "When answering, always start with 'Based on the Chick-fil-A menu/nutrition data, ...' to reference your source."
    )
    return ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful Chick-fil-A assistant."
            + hint
            + " Use the context below to answer the user's question. "
            + source_instruction + " "
            "When answering nutrition questions, always state the serving size the figures apply to. "
            "Never present per-100g figures as whole-meal calories. "
            "If you only have per-serving data and the user asks about a whole meal, say so clearly. "
            "Only say you don't have information if the context is completely unrelated "
            "to the question.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}"),
    ])


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def _extract_sources(docs):
    sources = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title") or doc.metadata.get("topic", "Source")
        source = {
            "index": i,
            "content": doc.page_content[:200],
            "topic": title,
        }
        if doc.metadata.get("url"):
            source["url"] = doc.metadata["url"]
        sources.append(source)
    return sources


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


def create_rag_chain_with_sources(query: str):
    """Create RAG chain that returns both answer and source documents for all query types."""
    if rag_store is None:
        raise ValueError("No knowledge base loaded yet.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    location_query = _is_location_query(query)

    if location_query:
        context, matched_locations = search_locations_with_sources(query)
        
        prompt = _build_prompt(location_query)
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": query
        })
        
        sources = []
        for loc in matched_locations:
            source = {
                "topic": loc.get("name", "Unknown Location"),
                "content": f"{loc.get('address', {}).get('city', '')} - {loc.get('phone', '')}"[:100],
            }
            if loc.get("url"):
                source["url"] = loc.get("url")
            sources.append(source)
        
        return {"answer": answer, "sources": sources}
    else:
        retriever = rag_store.as_retriever(search_kwargs={"k": 6})
        
        def get_answer_with_sources(q):
            docs = retriever.invoke(q)
            context = _format_docs(docs)
            prompt = _build_prompt(location_query)
            answer = (prompt | llm | StrOutputParser()).invoke({
                "context": context,
                "question": q
            })
            sources = _extract_sources(docs)
            return {"answer": answer, "sources": sources}
        
        return get_answer_with_sources(query)


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

    start = time.perf_counter()
    try:
        result = create_rag_chain_with_sources(query=query)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        Conversation(
            query=query,
            answer=answer,
            response_time_ms=elapsed_ms,
            is_success=True,
        ).save()
        return JsonResponse({
            "query": query,
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        Conversation(
            query=query,
            answer="",
            response_time_ms=elapsed_ms,
            is_success=False,
            error_message=str(e),
        ).save()
        return JsonResponse({"error": str(e)}, status=500)


# ── admin dashboard ──────────────────────────────────────────────────────────

TIME_RANGES = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}

staff_required = user_passes_test(lambda u: u.is_staff, login_url="/admin/login/")


def _filtered_qs(time_range):
    qs = Conversation.objects.all()
    delta = TIME_RANGES.get(time_range)
    if delta:
        qs = qs.filter(created_at__gte=timezone.now() - delta)
    return qs


@login_required(login_url="/admin/login/")
@staff_required
def dashboard(request):
    time_range = request.GET.get("range", "24h")
    qs = _filtered_qs(time_range)

    total_queries = qs.count()
    failed_queries = qs.filter(is_success=False).count()
    failure_rate = (failed_queries / total_queries * 100) if total_queries else 0
    avg_response_time = qs.filter(
        response_time_ms__isnull=False
    ).aggregate(avg=Avg("response_time_ms"))["avg"] or 0

    queries_today = Conversation.objects.filter(
        created_at__date=timezone.now().date()
    ).count()

    return render(request, "chat/dashboard.html", {
        "total_queries": total_queries,
        "failed_queries": failed_queries,
        "failure_rate": round(failure_rate, 1),
        "avg_response_time": round(avg_response_time),
        "queries_today": queries_today,
        "current_range": time_range,
        "time_ranges": list(TIME_RANGES.keys()),
    })


@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_logs(request):
    time_range = request.GET.get("range", "24h")
    page = int(request.GET.get("page", 1))
    per_page = 20

    qs = _filtered_qs(time_range).order_by("-created_at")
    total = qs.count()
    start = (page - 1) * per_page
    logs = qs[start:start + per_page]

    return JsonResponse({
        "logs": [
            {
                "id": c.id,
                "query": c.query[:150],
                "answer": c.answer[:150],
                "created_at": c.created_at.isoformat(),
                "response_time_ms": c.response_time_ms,
                "is_success": c.is_success,
                "error_message": c.error_message or "",
            }
            for c in logs
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total else 1,
    })


@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_chart(request):
    time_range = request.GET.get("range", "24h")

    qs = _filtered_qs(time_range).filter(
        response_time_ms__isnull=False
    ).order_by("-created_at")

    entries = list(qs[:50].values_list("created_at", "response_time_ms", "is_success"))
    entries.reverse()

    return JsonResponse({
        "data": [
            {
                "timestamp": ts.isoformat(),
                "response_time_ms": rt,
                "is_success": ok,
            }
            for ts, rt, ok in entries
        ]
    })




@csrf_exempt
@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_reload(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        count = reload_knowledge_base()
        return JsonResponse({"status": "ok", "documents_loaded": count})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_clear(request):
    global rag_store, location_data
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    rag_store = None
    location_data = []
    return JsonResponse({"status": "ok"})


SCRAPE_ETA_SECONDS = 180  # ~3 minutes expected runtime
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"


def _run_scraper():
    global _scrape_state
    _scrape_state.update({"status": "running", "started_at": time.time(), "finished_at": None, "message": ""})
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python"
    try:
        result = subprocess.run(
            [python, "-m", "data.scrape"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            reload_knowledge_base()
            _scrape_state.update({"status": "done", "finished_at": time.time(), "message": "Scrape complete. Knowledge base reloaded."})
        else:
            raw = (result.stderr or result.stdout or "").strip()
            # Surface only the final error line, not the full traceback
            last_error = next((l.strip() for l in reversed(raw.splitlines()) if l.strip()), raw)
            _scrape_state.update({"status": "error", "finished_at": time.time(), "message": last_error})
    except Exception as e:
        _scrape_state.update({"status": "error", "finished_at": time.time(), "message": str(e)})


@csrf_exempt
@login_required(login_url="/admin/login/")
@staff_required
@require_POST
def dashboard_api_scrape(request):
    if _scrape_state["status"] == "running":
        return JsonResponse({"error": "Scrape already in progress"}, status=409)
    threading.Thread(target=_run_scraper, daemon=True).start()
    return JsonResponse({"status": "started"})


@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_scrape_status(request):
    state = dict(_scrape_state)
    if state["status"] == "running" and state["started_at"]:
        elapsed = time.time() - state["started_at"]
        state["elapsed_seconds"] = int(elapsed)
        state["eta_seconds"] = max(0, int(SCRAPE_ETA_SECONDS - elapsed))
    return JsonResponse(state)


PDF_ETA_SECONDS = 30
PDF_EXTRACTOR_PATH = PROJECT_ROOT / "data" / "extract_nutrition.py"


def _run_pdf_extractor():
    global _pdf_state
    _pdf_state.update({"status": "running", "started_at": time.time(), "finished_at": None, "message": ""})
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python"
    try:
        result = subprocess.run(
            [python, "-m", "data.extract_nutrition"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            reload_knowledge_base()
            _pdf_state.update({"status": "done", "finished_at": time.time(), "message": "PDF extracted. Knowledge base reloaded."})
        else:
            raw = (result.stderr or result.stdout or "").strip()
            last_error = next((l.strip() for l in reversed(raw.splitlines()) if l.strip()), raw)
            _pdf_state.update({"status": "error", "finished_at": time.time(), "message": last_error})
    except Exception as e:
        _pdf_state.update({"status": "error", "finished_at": time.time(), "message": str(e)})


@csrf_exempt
@login_required(login_url="/admin/login/")
@staff_required
@require_POST
def dashboard_api_extract_pdf(request):
    if _pdf_state["status"] == "running":
        return JsonResponse({"error": "PDF extraction already in progress"}, status=409)
    threading.Thread(target=_run_pdf_extractor, daemon=True).start()
    return JsonResponse({"status": "started"})


@login_required(login_url="/admin/login/")
@staff_required
def dashboard_api_extract_pdf_status(request):
    state = dict(_pdf_state)
    if state["status"] == "running" and state["started_at"]:
        elapsed = time.time() - state["started_at"]
        state["elapsed_seconds"] = int(elapsed)
        state["eta_seconds"] = max(0, int(PDF_ETA_SECONDS - elapsed))
    return JsonResponse(state)