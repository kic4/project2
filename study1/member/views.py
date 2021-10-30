from django.shortcuts import render
from .models import Member
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import auth

# Create your views here.
def join(request):
    if request.method != 'POST':
        return render(request, 'member/joinform.html')
    else :
        member = Member(id=request.POST['id'],pass1=request.POST['pass'],name=request.POST['name'],\
                    gender=request.POST['gender'],tel=request.POST['tel'],email=request.POST['email'],\
                        picture=request.POST['picture'])
        member.save()
        return HttpResponseRedirect("../login/")

def login(request):
    if request.method != 'POST':
        return render(request, 'member/loginform.html')
    else :
        id1 = request.POST['id'];
        pass1 = request.POST['pass'];
        try :
           member = Member.objects.get(id=id1) #id의 db 정보 조회. 없는 id인 경우 예외 발생.
           #member.pass1 : member 테이블의 pass1 컬럼의 값
           #pass1 : 입력한 비밀번호. 파라미터 값
           if member.pass1 == pass1:  # 정상 로그인
               #request.session : session 객체.
               session_id = request.session.session_key
               request.session['login'] = id1 #로그인정보 session에 저장
               return HttpResponseRedirect("../main/")
           else: #비밀번호 틀린경우
               context = {"msg":"비밀번호가 틀립니다.", "url":"../login/"}
               return render(request, 'alert.html',context)
        except :
#           context = {"msg": "아이디가 틀립니다.", "url": "../login/"}
            return render(request, 'member/loginform.html',{"errormsg":"아이디가 틀립니다."})

def main(request):
        return render(request, 'member/main.html')

def logout(request):
    auth.logout(request)  #로그아웃
    return HttpResponseRedirect("../login/")

'''
  1. 로그인된 경우만 조회가능
  2. 내정보만 조회가능.
'''
def info(request,id):
    try :
        login = request.session["login"]
    except :
        login = ""
    if login != "" : #로그인 된경우
       if login == id or login == 'admin': #정상적인 기능
          member = Member.objects.get(id=id) #id에 해당하는 회원 정보
          return render(request, 'member/info.html',{"mem":member})
       else : #다른 사용자의 정보를 조회
          context = {"msg": "본인 정보만 조회가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)
'''
  수정 검증 : 
    1. 로그인 필요
    2. 본인 정보만 수정 가능
    3. 비밀번호 확인
'''
def update(request,id):
    try :
        login = request.session["login"]
    except :
        login = ""
    if login != "" :
       if login == id :
           return update_rtn(request,id)
       else :
          context = {"msg": "본인 정보만 조회가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)

def update_rtn(request,id) :
    if request.method != 'POST':
       member = Member.objects.get(id=id)  # id에 해당하는 회원 정보
       return render(request, 'member/updateform.html', {"mem": member})
    else :
# 수정 완료 후 /info/ 페이지 호출
        member = Member.objects.get(id=id)
        if member.pass1 == request.POST['pass'] :
           member = Member(id=request.POST['id'],name=request.POST['name'],pass1=request.POST['pass'],\
                gender=request.POST['gender'],tel=request.POST['tel'],email=request.POST['email'],\
                picture=request.POST['picture'])
           member.save()
           return HttpResponseRedirect("../../info/"+id+"/")
        else :
           context = {"msg": "회원 정보 수정 실패. \\n비밀번호 오류 입니다.",\
                      "url": "../../update/"+id+"/"}
           return render(request, 'alert.html', context)

'''
  비밀번호 수정 검증 : 
    1. 로그인 필요
    2. 본인 정보만 수정 가능
    3. GET : passwordform.html 출력
       POST :password.html 출력
          - 비밀번호 수정 완료 
             opener 창의 url을 /info/id/로 변경.
             현재창은 종료
          - 비밀번호입력 오류
             현재창으로  passwordform.html 출력
'''
def password(request,id) :
    try :
        login = request.session["login"]
    except :
        login = ""

    if login != "" :
       if login == id :
           return password_rtn(request,id)
       else :
          context = {"msg": "본인 정보만 조회가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)

def password_rtn(request,id) :
    if request.method != 'POST':
       return render(request, 'member/passwordform.html', {"id":id})
    else :
        member = Member.objects.get(id=id)
        if member.pass1 == request.POST['pass'] :
            member.pass1 = request.POST['chgpass']
            member.save()
            context = {"msg": "비밀번호 수정이 완료 되었습니다.",\
                       "url": "../../info/" + id + "/","closer":True}
            return render(request, 'member/password.html', context)
        else :
           context = {"msg": "비밀번호 오류 입니다.", "url": "../../password/"+id+"/",\
                      "closer":False}
           return render(request, 'member/password.html', context)

def delete(request,id) :
    try :
        login = request.session["login"]
    except :
        login = ""

    if login != "" :
       if login == id :
           return delete_rtn(request,id)
       else :
          context = {"msg": "3.본인만 탈퇴 가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "3.로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)

def delete_rtn(request,id) :
    if request.method != 'POST':
       return render(request, 'member/deleteform.html', {"id":id})
    else :
        member = Member.objects.get(id=id)
        if member.pass1 == request.POST['pass'] :
            member.delete()
            auth.logout(request)  #로그아웃
            context = {"msg": "회원님 탈퇴처리가 완료 되었습니다.", "url": "../../login/"}
            return render(request, 'alert.html', context)
        else :
           context = {"msg": "비밀번호 오류 입니다.", "url": "../../delete/"+id+"/"}
           return render(request, 'alert.html', context)

def picture(request) :
    if request.method != 'POST':
       return render(request, 'member/pictureform.html')
    else :
        fname = request.FILES['picture'].name
        handle_upload(request.FILES['picture'])
        return render(request, 'member/picture.html', {'fname': fname})

def handle_upload(f):
    with open("file/picture/" + f.name, 'wb+') as destination:
        for ch in f.chunks():
            destination.write(ch)

def list(request) :
    try :
        login = request.session["login"]
    except :
        login = ""
    if login != "" :
       if login == 'admin' :
           member = Member.objects.all()
           return render(request, 'member/list.html', {"mlist": member})
       else :
          context = {"msg": "관리자만 조회합니다.", "url": "../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../login/"}
        return render(request, 'alert.html', context)
