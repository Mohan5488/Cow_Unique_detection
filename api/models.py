from django.db import models

# Create your models here.
class Hombenai(models.Model):
    unique_id = models.CharField(max_length=10)
    name = models.CharField(max_length=50)
    location = models.CharField(max_length=200)
    upload = models.ImageField(upload_to='images')
