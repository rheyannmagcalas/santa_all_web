from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include
from django.views.generic.base import TemplateView

from documentation import views

urlpatterns =[
    url(r'^admin/', admin.site.urls),
    url(r"^accounts/", include("django.contrib.auth.urls")),
    path('', views.index, name='home'),
    url(r'^documentation/',include('documentation.urls')),
    url(r'^wishlist/',include('wishlist.urls'))
    # url(r'^logout/$', views.user_logout, name='logout'),
]