#pip install pymysql 필요
#pip install mysqlclient 필요
#settings.py 마리아db로 변경
#python manage.py migrate
#python manage.py runserver
# 브라우저 : http://localhost:8000/board/write
import pymysql
#conn = pymysql.connect(host="localhost",port=3306, user='kic',passwd='1234',db='kicdb',charset="utf8")

from django.db import models
# Create your models here.
class Board(models.Model):
    num = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)
    pass1 = models.CharField(max_length=20)
    subject = models.CharField(max_length=100)
    content = models.CharField(max_length=4000)
    regdate = models.DateTimeField()
    readcnt = models.IntegerField(default=0)
    file1 = models.CharField(max_length=100)

    def __str__(self):
        return str(self.num)+':' + self.subject