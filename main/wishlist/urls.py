from django.conf.urls import url
from wishlist import views

app_name = 'wishlist'

urlpatterns=[
    url(r'^$',views.index, name='index'),
    url(r'^register/$',views.register, name='register'),
    url(r'^forgot-password/$',views.forgot_password, name='forgot-password'),
    url(r'^my-wishlist/$',views.mywishlist, name='my-wishlist'),
    url(r'^community/$',views.community, name='community'),
    url(r'^search/$', views.SearchWishlist.as_view(), name="search-wishlist"),
]