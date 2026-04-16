from django.contrib import admin
from django.urls import path, include
from chat.views import interface, dashboard, login_view, register_view, logout_view, history_view

urlpatterns = [
    path('', interface, name='interface'),
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('history/', history_view, name='history'),
    path('dashboard/', dashboard, name='dashboard'),
    path('admin/', admin.site.urls),
    path('api/chat/', include('chat.urls')),
]
