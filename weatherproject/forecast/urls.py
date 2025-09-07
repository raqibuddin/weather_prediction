from django.urls import path
from . import views

urlpatterns =[
    path('', views.enter_api_key, name="enter_api_key"),
    path('weather/', views.weather_view, name='Weather View'),
]