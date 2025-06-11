from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static 
from . import views
urlpatterns = [
    path("", views.home, name='home'),
    path("upload/", views.upload, name='upload'),
    path('train/', views.train, name='train'),
    path('training/', views.training, name='training'),
    path('testing/', views.testing, name='testing'),
    path('test/', views.test_image, name='test'),
    path('delete_cow/<int:id>', views.delete_cow, name="delete_cow"),
]