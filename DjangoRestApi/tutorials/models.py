from django.db import models


class Tutorial(models.Model):
    title = models.CharField(max_length=300, blank=False, default='')
    description = models.CharField(max_length=2000,blank=False, default='')
    published = models.BooleanField(default=False)


class Scenario(models.Model):
    seqtactic = models.CharField(max_length=2000, blank=False, default='')
    iduser=models.CharField(max_length=50, blank=False, default='')
    seqtec = models.CharField(max_length=2000,blank=False, default='')
    software = models.CharField(max_length=70, blank=False, default='')

    score=models.CharField(max_length=50, blank=False, default='')

