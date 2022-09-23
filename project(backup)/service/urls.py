from django.urls import path
from . import views

urlpatterns = [
    path('mem_info/', views.mem_info, name='mem_info'),
]
