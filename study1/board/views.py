from django.shortcuts import render
from . import models
from django.http import HttpResponseRedirect
from .models import Board
from django.utils import timezone
from django.core.paginator import  Paginator

# Create your views here.
def handle_upload(f) :
# 파일을 서버의 지정된 위치에 저장
    with open("file/board/" +f.name, "wb+") as destination :
        for ch in f.chunks() :
            destination.write(ch)

# http://localhost:8000/board/write/
def write(request) :
    if request.method != 'POST' :  #GET 방식 요청
        return render(request,'board/writeform.html')
    else :  #POST 방식 요청
        try :
#   request.FILES["file1"] : 업로드된 파일
            filename = request.FILES["file1"].name #업로드된 파일 이름
            handle_upload(request.FILES["file1"])
        except :
            filename = ""
        print("filename=",filename)

        b = Board(name=request.POST['name'],pass1=request.POST['pass'],\
              subject=request.POST['subject'], content=request.POST['content'],\
              regdate= timezone.now(),readcnt = 0,file1=filename)
        b.save()  #해당 레코드가 없는 경우 num 컬럼을 자동 증가시키고,  db에 추가함.

        return HttpResponseRedirect("../list/")


#http://localhost:8000/board/list/?pageNum=2
def list(request) :
    pageNum = int(request.GET.get('pageNum', 1)) #pageNum파라미터값 정수로 변경. pageNum이 없으면 1로 설정
    all_boards = Board.objects.all().order_by("-num") # 모든 데이터 조회, 내림차순(-표시) 조회
    paginator = Paginator(all_boards, 10) #10개씩 분리하여 페이징함.
    board_list = paginator.get_page(pageNum) #pageNum=1 경우 첫번째 페이지 정보만 board_list 저장
    listcount = Board.objects.count() #게시물 등록 건수
    return render(request, 'board/list.html', {'board':board_list,'listcount':listcount})

# http://localhost:8000/board/info/10/
def info(request, num):
    board = Board.objects.get(num=num) #num 컬럼의 값이 num값인 데이터 한건 조회
    board.readcnt += 1  #조회수 증가.
    board.save()        #BOARD 데이터를 DB에서 수정.
    return render(request, 'board/info.html', {'b': board})

def update(request, num):
    if request.method != 'POST':
        board = Board.objects.get(num=num)
        context = {'b': board}
        return render(request, 'board/updateform.html',context)
    else :
        try:
            handle_upload(request.FILES['file1'])
            filename = request.FILES['file1'].name
        except:
            filename = ''

        if filename == "" :
            filename = request.POST["file2"]

        b = Board(num=request.POST['num'],name=request.POST['name'],pass1=request.POST['pass'],\
                subject=request.POST['subject'], content=request.POST['content'],\
                regdate= timezone.now(),readcnt = 0,file1=filename)
        b.save()
        return HttpResponseRedirect("../../list/")

def delete (request, num):
    if request.method != 'POST':
        return render(request, 'board/deleteform.html',{"num":num})
    else :
        board = Board.objects.get(num=num)
        pass1 = request.POST["pass"]
        if board.pass1 == pass1:
           board.delete()
           return HttpResponseRedirect("../../list/")
        else :
           return render(request, 'board/deleteform.html', {"num": num})
