from django.conf.urls import include
from django.urls import re_path as url

urlpatterns = [
    url(r'^', include('tutorials.urls')),
]