from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

def mem_info(request):
    return render(request, 'service/mem_info.html')