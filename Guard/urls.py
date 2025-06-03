from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path("admin/", admin.site.urls),
    path("guard/", views.guard, name="guard"),
    path("prevention/", views.prevention, name="prevention"),
    path('chat/', views.chat_api, name='chat_api'),  # 新增chat api路由
    path('broadcast/', views.broadcast_case, name='broadcast_case'),
    path('record_exception/', views.record_exception, name='record_exception'),
]
