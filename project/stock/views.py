from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.http import StreamingHttpResponse

class Echo:
    """An object that implements just the write method of the file-like
    interface.
    """
    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        print("stock의 views.py")
        return value

def stock(request):
    return render(request, 'stock/stock.html')

def stock1(request):
    return render(request, 'stock/stock1.html')

def stock2(request):
    return render(request, 'stock/stock2.html')

def stock3(request):
    return render(request, 'stock/stock3.html')

def search(request):
    if request.method != 'POST':
        return render(request, 'stock/stock.html')
    else:
        search = request.POST['search'].replace(" ","")
        search1 = search.upper()

        if search1 == 'HMM' or search1 == '해운':
            return render(request, 'stock/stock1.html')
        elif search1 == '알테오젠' or search1 == '생물공학':
            return render(request, 'stock/stock2.html')
        elif search1 == '씨젠' or search1 == '생명과학도구및서비스':
            return render(request, 'stock/stock3.html')
        elif search1 == "":
            context = {"msg": "기업명 혹은 종목을 입력해주세요", "url": "../../stock/stock"}
            return render(request, 'alert.html', context)
        else:
            context = {"msg": "입력하신 종목 및 기업명과 일치하는 항목이 없습니다.", "url": "../../stock/stock"}
            return render(request, 'alert.html', context)
