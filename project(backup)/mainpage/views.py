from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import auth

def main(request):
    return render(request, 'mainpage/index.html')