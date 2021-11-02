from django.urls import path
from . import views

urlpatterns = [
    path('stock/', views.stock, name='stock'),
    path('stock1/', views.stock1, name='stock1'),
    path('stock2/', views.stock2, name='stock2'),
    path('stock3/', views.stock3, name='stock3'),
    path('search/', views.search, name='search'),
]
