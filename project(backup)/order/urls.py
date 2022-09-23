from django.urls import path
from . import views

urlpatterns = [
    path('buy/', views.buy, name='buy'),
    path('sell/', views.sell, name='sell')
]
