from django.urls import path
from . import views

urlpatterns = [
    path('ingest/', views.ingest_documents, name='rag_ingest'),
    path('query/', views.query_chat, name='rag_query'),
    path('dashboard/logs/', views.dashboard_api_logs, name='dashboard_logs'),
    path('dashboard/chart/', views.dashboard_api_chart, name='dashboard_chart'),
    path('dashboard/reload/', views.dashboard_api_reload, name='dashboard_reload'),
    path('dashboard/clear/', views.dashboard_api_clear, name='dashboard_clear'),
    path('dashboard/scrape/', views.dashboard_api_scrape, name='dashboard_scrape'),
    path('dashboard/scrape/status/', views.dashboard_api_scrape_status, name='dashboard_scrape_status'),
    path('dashboard/extract-pdf/', views.dashboard_api_extract_pdf, name='dashboard_extract_pdf'),
    path('dashboard/extract-pdf/status/', views.dashboard_api_extract_pdf_status, name='dashboard_extract_pdf_status'),
]
