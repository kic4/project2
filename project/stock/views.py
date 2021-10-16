from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import auth

def stock(request):
    return render(request, 'stock/stock.html')