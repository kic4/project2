from django.urls import path
from . import views

urlpatterns = [
    path('join/', views.join, name='join'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('main/', views.main, name='main'),
    path('info/<str:id>/', views.info, name='info'),
    path('update/<str:id>/', views.update, name='update'),
    path('password/<str:id>/', views.password, name='password'),
    path('delete/<str:id>/', views.delete, name='delete'),
    path('picture/', views.picture, name='picture'),
    path("list/", views.list, name='list'),
]
