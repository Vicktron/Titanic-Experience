from django.db import models


class Prediction(models.Model):
    name = models.CharField(max_length=50)
    pclass = models.IntegerField()
    gender = models.CharField(max_length=6)
    realage = models.IntegerField()
    realfamilysize = models.IntegerField()
