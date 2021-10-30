from django.urls import path
from . import views

urlpatterns = [
    path('update/<int:num>/', views.update, name='update'),
    path('info/<int:num>/', views.info, name='info'),
    path('list/', views.list, name='list'),
    path("write/",views.write,name="write"),
    path('delete/<int:num>/', views.delete, name='delete'),
]
