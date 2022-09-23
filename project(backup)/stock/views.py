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

def streaming_graph(request):
    rows = (["Row {}".format(idx), str(idx)] for idx in range(65536))
    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)
    return StreamingHttpResponse(
        (writer.writerow(row) for row in rows),
        content_type="text/csv",
        headers={'Content-Disposition': 'attachment; filename="somefilename.csv"'},
    )