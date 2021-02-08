from django.db import models
from django.contrib.auth.models import User

# Create your models here.



class MyWishlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    wishlist_item = models.CharField(max_length=256, null=True, blank=True))
    date_added = models.DateTimeField(null=True, blank=True) 

    class Meta:
        app_label = 'ssa'

class MyWishlistItems(models.Model):
    wishlist_id = models.ForeignKey(MyWishlist, on_delete=models.CASCADE)
    product_name = models.CharField(max_length=256, null=True, blank=True))

    class Meta:
        app_label = 'ssa'