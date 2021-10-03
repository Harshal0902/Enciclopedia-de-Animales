from django.db import models

# Create your models here.

class modelimg(models.Model):
    predicton = models.CharField(max_length=100)
    img = models.CharField(max_length=1000)
    probability = models.CharField(max_length=100)
    