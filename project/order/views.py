from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

def buy(request):
    return render(request, 'order/buy.html')

def sell(request):
    return render(request, 'order/sell.html')
