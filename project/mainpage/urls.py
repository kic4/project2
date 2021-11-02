from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.main, name='main'),
    path('stock/', views.search, name='search'),

]
